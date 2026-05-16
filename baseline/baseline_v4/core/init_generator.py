"""可行解生成器：多种策略生成初始可行解池。"""

from typing import List, Optional
import numpy as np
from numpy import ndarray
import time
from scipy.optimize import linprog


class FeasibleSolutionGenerator:
    """可行解生成器。

    策略组合:
        1. TrivialSolver: 全零解（必定可行的基准）
        2. GreedyConstruction: 按边际贡献逐变量构造
        3. RandomFeasible: 随机采样 + 可行性修复
        4. RoundingLP: LP 松弛取整
        5. SingleFlipImprove: 单变量翻转局部搜索
    """

    def __init__(self, evaluator, repairer):
        self.evaluator = evaluator
        self.repairer = repairer
        self.inst = evaluator.inst

    def generate_pool(self, pool_size: int, time_limit: float) -> List:
        """生成初始可行解池。

        时间分配:
            [0, 0.05T]:   TrivialSolver → 1 解
            [0.05T, 0.3T]: GreedyConstruction × 5 → ~5 解
            [0.3T, 0.6T]: RandomFeasible × 10 → ~7 解（含修复失败）
            [0.6T, 0.8T]: RoundingLP → 2 解
            [0.8T, T]:    SingleFlipImprove(best) → 1 解

        增强: n ≤ 15 时直接穷举所有可行解（精确全局最优初始化）。
        """
        t0 = time.perf_counter()
        T = time_limit
        results = []

        # === 增强: 小实例穷举初始化 ===
        brute_result = self._brute_force_init(max_n=15, max_feasible=3000)
        if brute_result is not None:
            results.append(brute_result)
            # 穷举已找到全局最优，可以跳过其他初始化以节省时间
            # 但对于 n=15，穷举很快，仍可继续其他策略增加多样性
            if self.inst.n <= 12:
                # n <= 12 时穷举几乎瞬间完成，直接返回最优解即可
                return results

        elapsed = time.perf_counter() - t0
        if elapsed >= T:
            return results

        # Phase 1: TrivialSolver (全零解)
        if not results:
            x_zero = np.zeros(self.inst.n)
            r_zero = self.evaluator.evaluate(x_zero)
            if r_zero.is_feasible:
                results.append(r_zero)

        elapsed = time.perf_counter() - t0
        if elapsed >= T:
            return results

        # Phase 2: GreedyConstruction
        n_greedy = min(5, pool_size - len(results))
        for _ in range(n_greedy):
            if time.perf_counter() - t0 >= 0.3 * T:
                break
            x_greedy = self._greedy_construction()
            r_greedy = self.evaluator.evaluate(x_greedy)
            if r_greedy.is_feasible:
                results.append(r_greedy)

        # Phase 3: RandomFeasible
        n_random = min(10, pool_size - len(results))
        for _ in range(n_random):
            if time.perf_counter() - t0 >= 0.6 * T:
                break
            x_rand = np.random.randint(0, 2, self.inst.n).astype(float)
            # 尝试修复
            repair_result = self.repairer.repair(x_rand, list(range(self.inst.n)), x_rand)
            if repair_result.is_feasible:
                r_rand = self.evaluator.evaluate(repair_result.x_repaired)
                if r_rand.is_feasible:
                    results.append(r_rand)

        # Phase 4: RoundingLP
        if time.perf_counter() - t0 < 0.8 * T and len(results) < pool_size:
            x_round = self._rounding_lp()
            if x_round is not None:
                repair_result = self.repairer.repair(x_round, list(range(self.inst.n)), x_round)
                if repair_result.is_feasible:
                    r_round = self.evaluator.evaluate(repair_result.x_repaired)
                    if r_round.is_feasible:
                        results.append(r_round)
                        # 再做一次扰动
                        x_round2 = np.where(np.random.rand(self.inst.n) > 0.5, 1 - x_round, x_round)
                        rr2 = self.repairer.repair(x_round2, list(range(self.inst.n)), x_round2)
                        if rr2.is_feasible:
                            r2 = self.evaluator.evaluate(rr2.x_repaired)
                            if r2.is_feasible:
                                results.append(r2)

        # Phase 5: SingleFlipImprove on best so far
        if results and time.perf_counter() - t0 < T:
            best = max(results, key=lambda r: r.objective)
            x_improved = self._single_flip_improve(best.x)
            r_imp = self.evaluator.evaluate(x_improved)
            if r_imp.is_feasible and r_imp.objective > best.objective:
                results.append(r_imp)

        # 去重并按目标值排序
        unique_results = self._deduplicate(results)
        unique_results.sort(key=lambda r: r.objective, reverse=True)
        return unique_results[:pool_size]

    def _greedy_construction(self) -> ndarray:
        """贪心构造可行解。

        按单变量边际贡献降序排列，逐个尝试设为 1，
        仅当保持可行性时才保留。
        """
        inst = self.inst
        scores = inst.c + np.diag(inst.Q)  # 单变量边际贡献
        order = np.argsort(-scores)

        x = np.zeros(inst.n)
        for i in order:
            x_try = x.copy()
            x_try[i] = 1
            # 快速二元可行性检查
            if inst.m2 > 0 and np.any(inst.B @ x_try > inst.b_prime + 1e-10):
                continue
            # LP 可行性检查
            lp_r = self.evaluator.solve_lp(x_try)
            if lp_r is not None:
                x = x_try
        return x

    def _rounding_lp(self) -> Optional[ndarray]:
        """LP 松弛取整。

        求解 LP 松弛后按概率取整，再修复。
        对于二次目标，使用线性近似或 QP 求解器。
        """
        inst = self.inst
        # 简化：使用线性近似 max c^T x + h^T y
        # 未来可扩展为使用 QP 求解器处理二次项
        c_lin = inst.c.copy()
        rhs = inst.b.copy()

        try:
            result = linprog(
                c=-c_lin,
                A_ub=inst.A,
                b_ub=rhs,
                A_eq=inst.B if inst.m2 > 0 else None,
                b_eq=inst.b_prime if inst.m2 > 0 else None,
                bounds=[(0, 1)] * inst.n,
                method='highs',
            )
        except Exception:
            return None

        if result.success:
            x_frac = result.x[:inst.n]
            # 按概率取整
            x_rounded = (np.random.rand(inst.n) < x_frac).astype(float)
            return x_rounded
        return None

    def _single_flip_improve(self, x: ndarray) -> ndarray:
        """单变量翻转局部搜索。"""
        best_x = x.copy()
        best_obj = self.evaluator.evaluate(x).objective
        improved = True
        max_iter = self.inst.n
        iteration = 0

        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            for i in range(self.inst.n):
                x_flip = best_x.copy()
                x_flip[i] = 1 - x_flip[i]
                r = self.evaluator.evaluate(x_flip)
                if r.is_feasible and r.objective > best_obj:
                    best_x = x_flip
                    best_obj = r.objective
                    improved = True
                    break
        return best_x

    def _brute_force_init(self, max_n: int = 15, max_feasible: int = 3000):
        """小实例穷举初始化。

        对 n <= max_n 的实例，枚举所有 2^n 个二元解，
        通过向量化二元约束过滤后，逐个 evaluate 找到全局最优。

        若二元约束过滤后可行解数量 > max_feasible，则回退到 None
        （避免约束过松的实例耗时过长）。

        Returns:
            EvalResult or None
        """
        n = self.inst.n
        if n > max_n:
            return None

        N = 2 ** n
        # 生成所有位串: (N, n)
        bits = ((np.arange(N)[:, None] >> np.arange(n)) & 1).astype(np.float64)

        # 向量化二元约束过滤
        if self.inst.m2 > 0:
            feasible_mask = np.all(
                self.inst.B @ bits.T <= self.inst.b_prime[:, None] + 1e-10, axis=0
            )
            feasible_indices = np.where(feasible_mask)[0]
        else:
            feasible_indices = np.arange(N)

        if len(feasible_indices) > max_feasible:
            return None  # 可行解太多，放弃穷举

        # 逐个 evaluate 找最优
        best_obj = -np.inf
        best_result = None
        for idx in feasible_indices:
            x = bits[idx]
            r = self.evaluator.evaluate(x)
            if r.is_feasible and r.objective > best_obj:
                best_obj = r.objective
                best_result = r

        return best_result

    def _deduplicate(self, results: List) -> List:
        """按 Hamming 距离去重。"""
        if not results:
            return results
        unique = [results[0]]
        for r in results[1:]:
            is_dup = False
            for u in unique:
                hd = np.sum(r.x != u.x) / self.inst.n
                if hd < 0.05:  # 5% 以下为重复
                    is_dup = True
                    break
            if not is_dup:
                unique.append(r)
        return unique
