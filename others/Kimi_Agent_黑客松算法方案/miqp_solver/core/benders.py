"""
Benders分解主循环 - 量子增强的Benders分解

核心思想：
1. 外层：Benders分解，处理二元变量x和连续变量y的分离
2. 内层：subQUBO用QAOA求解，LP用经典求解器
3. 分块策略：大问题分解为多个subQUBO，每块≤20量子比特
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import config
from utils.blocking import (
    extract_subqubo, merge_solution,
    coupling_strength_blocks, random_blocks
)
from core.quantum_solver import create_quantum_solver, solve_qubo_exact
from core.lp_solver import solve_y_subproblem
from utils.validator import check_feasibility, compute_objective, quick_validation_report


class QuantumBendersSolver:
    """
    量子Benders分解求解器

    适用于MIQP：max x^T Q x + c^T x + h^T y
                  s.t. Ax + Gy <= b
                       Bx <= bp
                       x in {0,1}^n, y >= 0
    """

    def __init__(self, data: dict, block_size: int = None,
                 n_qaoa_blocks: int = 1, backend: str = None):
        """
        Args:
            data: 问题数据字典
            block_size: subQUBO块大小（默认从config读取）
            n_qaoa_blocks: 每轮Benders迭代中随机选择几个块用QAOA求解
        """
        self.data = data
        self.Q = data["Q"]
        self.c = data["c"]
        self.n = data["n"]
        self.p = data["p"]
        self.block_size = block_size or config.BLOCK_SIZE
        self.n_qaoa_blocks = n_qaoa_blocks

        # 初始化量子求解器
        self.qsolver = create_quantum_solver(backend=backend)

        # 分块
        if self.n <= self.block_size:
            self.blocks = [list(range(self.n))]
        else:
            self.blocks = coupling_strength_blocks(self.Q, self.block_size)

        print(f"[INFO] Benders Solver: n={self.n}, p={self.p}, "
              f"blocks={len(self.blocks)}, block_size={self.block_size}")

        # 记录求解历史
        self.history = {
            "iterations": [],
            "objectives": [],
            "best_obj": float("-inf"),
            "best_x": None,
            "best_y": None,
        }

    def _solve_subqubo_block(self, block_vars: List[int],
                             x_current: np.ndarray,
                             warm_start: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        求解单个subQUBO块

        Args:
            block_vars: 块内变量索引
            x_current: 当前全局解
            warm_start: 热启动参数

        Returns:
            x_sub: 块内最优解
            obj_sub: 块内目标值
        """
        # 构建subQUBO
        fixed_vars = x_current.copy()
        for bv in block_vars:
            fixed_vars[bv] = -1  # 标记为未固定

        Q_sub, c_sub, var_map, constant = extract_subqubo(
            self.Q, self.c, fixed_vars, block_vars
        )

        k = len(block_vars)

        # 小规模用精确求解
        if k <= 15:
            x_sub, obj_sub = solve_qubo_exact(Q_sub, c_sub)
            obj_sub += constant
            return x_sub, obj_sub

        # 大规模用QAOA
        if warm_start is not None and len(warm_start) == 2 * config.QAOA_REPS:
            x_sub, obj_sub = self.qsolver.solve(Q_sub, c_sub, initial_point=warm_start)
        else:
            x_sub, obj_sub = self.qsolver.solve(Q_sub, c_sub)

        obj_sub += constant
        return x_sub, obj_sub

    def _solve_all_blocks(self, x_current: np.ndarray,
                          use_quantum: bool = True) -> np.ndarray:
        """
        求解所有subQUBO块

        Args:
            x_current: 当前解
            use_quantum: 是否对选中的块使用QAOA

        Returns:
            x_new: 更新后的全局解
        """
        x_new = x_current.copy()

        # 随机选择用QAOA求解的块（减少量子计算时间）
        n_total = len(self.blocks)
        if use_quantum and self.n > self.block_size:
            # 选择目标值改善潜力最大的块
            block_scores = []
            for block in self.blocks:
                # 简单启发式：当前解中变化潜力大的块
                score = np.sum(np.abs(self.Q[np.ix_(block, block)]))
                block_scores.append(score)

            block_order = np.argsort(block_scores)[::-1]
            quantum_blocks = block_order[:self.n_qaoa_blocks]
        else:
            quantum_blocks = list(range(n_total))

        for bidx, block in enumerate(self.blocks):
            try:
                use_q = use_quantum and (bidx in quantum_blocks)

                if use_q or self.n <= self.block_size:
                    x_sub, _ = self._solve_subqubo_block(block, x_new)
                else:
                    # 用经典穷举求解这个小块（作为fallback）
                    fixed_vars = x_new.copy()
                    for bv in block:
                        fixed_vars[bv] = -1
                    Q_sub, c_sub, var_map, _ = extract_subqubo(
                        self.Q, self.c, fixed_vars, block
                    )
                    x_sub, _ = solve_qubo_exact(Q_sub, c_sub)

                # 合并解
                x_new = merge_solution(x_new, x_sub, np.array(block))

            except Exception as e:
                print(f"[WARNING] Block {bidx} solve failed: {e}")
                continue

        return x_new

    def solve(self, max_iter: int = None, tol: float = None,
              time_limit: float = None) -> Dict:
        """
        主求解循环

        Args:
            max_iter: 最大迭代次数
            tol: 收敛容差
            time_limit: 时间限制（秒）

        Returns:
            {
                "x": 最优二元解,
                "y": 最优连续解,
                "objective": 最优目标值,
                "history": 迭代历史,
                "time": 总求解时间
            }
        """
        max_iter = max_iter or config.BENDERS_MAX_ITER
        tol = tol or config.BENDERS_TOLERANCE
        time_limit = time_limit or config.MAX_TIME_PER_TEST

        start_time = time.time()

        # 初始解：随机或全0
        x_best = np.random.randint(0, 2, self.n).astype(float)
        y_best, success, _ = solve_y_subproblem(self.data, x_best)

        best_obj = compute_objective(self.data, x_best, y_best)
        print(f"[INIT] Initial obj={best_obj:.4f}, feasible={success}")

        # 如果初始解不可行，尝试全0解
        if not success:
            x_alt = np.zeros(self.n)
            y_alt, success_alt, _ = solve_y_subproblem(self.data, x_alt)
            if success_alt:
                obj_alt = compute_objective(self.data, x_alt, y_alt)
                if obj_alt > best_obj:
                    x_best, y_best, best_obj = x_alt, y_alt, obj_alt

        self.history["best_x"] = x_best.copy()
        self.history["best_y"] = y_best.copy() if y_best is not None else None
        self.history["best_obj"] = best_obj

        # Benders迭代
        for iteration in range(max_iter):
            if time.time() - start_time > time_limit:
                print(f"[TIMEOUT] Iteration {iteration}, time limit reached")
                break

            # 求解LP子问题
            y_new, success, y_obj = solve_y_subproblem(self.data, x_best)

            if not success:
                # 如果LP不可行，稍微扰动x
                x_best = np.random.randint(0, 2, self.n).astype(float)
                continue

            # 固定y，求解关于x的subQUBO块
            x_new = self._solve_all_blocks(x_best, use_quantum=True)

            # 评估新解
            y_new, success, y_obj = solve_y_subproblem(self.data, x_new)
            if not success:
                # 保持当前解
                obj_new = float("-inf")
            else:
                obj_new = compute_objective(self.data, x_new, y_new)

            # 记录
            self.history["iterations"].append(iteration)
            self.history["objectives"].append(obj_new)

            # 更新最优解
            if success and obj_new > best_obj:
                improvement = obj_new - best_obj
                best_obj = obj_new
                x_best = x_new.copy()
                y_best = y_new.copy()
                self.history["best_x"] = x_best.copy()
                self.history["best_y"] = y_best.copy()
                self.history["best_obj"] = best_obj

                print(f"[ITER {iteration}] obj={best_obj:.6f}, improvement={improvement:.6f}")

                if improvement < tol:
                    print(f"[CONVERGED] Improvement < {tol}")
                    break
            else:
                print(f"[ITER {iteration}] obj={obj_new:.4f}, success={success}")

        total_time = time.time() - start_time
        self.history["time"] = total_time

        # 最终验证
        feas = check_feasibility(self.data, x_best, y_best)
        print(quick_validation_report(self.data, x_best, y_best))

        return {
            "x": x_best,
            "y": y_best if y_best is not None else np.zeros(self.p),
            "objective": best_obj,
            "feasible": feas["feasible"],
            "history": self.history,
            "time": total_time,
            "iterations": len(self.history["iterations"])
        }


class FallbackSolver:
    """
    Fallback求解器：当量子求解超时时使用
    使用贪心+局部搜索的经典启发式
    """

    def __init__(self, data: dict):
        self.data = data
        self.Q = data["Q"]
        self.c = data["c"]
        self.n = data["n"]

    def solve(self, time_limit: float = 60) -> Dict:
        """
        贪心+局部搜索求解QUBO部分
        """
        import time
        start = time.time()

        # 贪心构造
        x = np.zeros(self.n)
        current_obj = 0

        for _ in range(self.n):
            best_i = -1
            best_gain = 0
            for i in range(self.n):
                if x[i] == 1:
                    continue
                # 尝试设置x[i]=1
                x[i] = 1
                gain = x @ self.Q @ x + self.c @ x - current_obj
                x[i] = 0
                if gain > best_gain:
                    best_gain = gain
                    best_i = i

            if best_i >= 0 and best_gain > 0:
                x[best_i] = 1
                current_obj += best_gain
            else:
                break

        # 局部搜索（1-opt）
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            for i in range(self.n):
                x[i] = 1 - x[i]
                new_obj = x @ self.Q @ x + self.c @ x
                if new_obj > current_obj:
                    current_obj = new_obj
                    improved = True
                else:
                    x[i] = 1 - x[i]

        # 求解LP
        y, success, _ = solve_y_subproblem(self.data, x)

        total_obj = compute_objective(self.data, x, y)

        return {
            "x": x,
            "y": y,
            "objective": total_obj,
            "feasible": success,
            "history": {},
            "time": time.time() - start,
            "method": "fallback_greedy"
        }
