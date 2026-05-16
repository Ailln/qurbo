"""目标函数评价器：给定二元解 x，求解连续子问题并提取对偶信息。"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy import ndarray
from scipy.optimize import linprog


@dataclass
class EvalResult:
    """评价结果。"""
    objective: float        # 完整目标值 x^TQx + c^Tx + h^Ty*
    x: ndarray              # 二元解
    y: Optional[ndarray]    # 连续最优解
    dual: Optional[ndarray] # 对偶价格 u* (m1,)
    is_feasible: bool       # 是否可行
    lp_status: str          # 'optimal' | 'infeasible' | 'unbounded' | 'binary_infeasible'
    solve_time: float       # LP 求解耗时


@dataclass
class LPResult:
    """LP 求解结果。"""
    y: ndarray          # 最优连续解
    dual: Optional[ndarray]  # 对偶价格
    obj_val: float      # 连续部分目标值


class ObjectiveEvaluator:
    """目标函数评价器。

    核心方法:
        evaluate(x): 给定二元解，求解连续子问题，返回完整目标值和对偶信息
        solve_lp(x): 求解 LP(x)
        compute_benders_linear(dual): 计算 Benders 线性项 l_cont = -A^T u*
    """

    def __init__(self, instance, cache_size: int = 5000):
        self.inst = instance
        self._cache = {}
        self._cache_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0

    def evaluate(self, x: ndarray) -> EvalResult:
        """评价给定二元解 x。

        流程:
            1. 检查缓存
            2. 检查纯二元约束可行性 (Bx <= b')
            3. 求解连续子问题 LP(x)
            4. 计算完整目标值
            5. 提取对偶价格

        单次调用性能目标: <= 50ms (n <= 120, p <= 100)
        """
        import time
        t0 = time.perf_counter()

        # Step 0: 缓存检查
        key = tuple(x.astype(int).tolist())
        cached = self._cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1

        # Step 1: 纯二元约束检查
        if self.inst.m2 > 0 and np.any(self.inst.B @ x > self.inst.b_prime + 1e-10):
            result = EvalResult(
                objective=-np.inf,
                x=x.copy(),
                y=None,
                dual=None,
                is_feasible=False,
                lp_status='binary_infeasible',
                solve_time=time.perf_counter() - t0,
            )
            self._add_to_cache(key, result)
            return result

        # Step 2: 求解连续子问题
        lp_result = self.solve_lp(x)
        solve_time = time.perf_counter() - t0

        if lp_result is None:
            result = EvalResult(
                objective=-np.inf,
                x=x.copy(),
                y=None,
                dual=None,
                is_feasible=False,
                lp_status='infeasible',
                solve_time=solve_time,
            )
            self._add_to_cache(key, result)
            return result

        # Step 3: 计算完整目标值
        quad = float(x @ self.inst.Q @ x)
        lin_bin = float(self.inst.c @ x)
        lin_cont = float(self.inst.h @ lp_result.y) if lp_result.y is not None else 0.0
        total_obj = quad + lin_bin + lin_cont

        result = EvalResult(
            objective=total_obj,
            x=x.copy(),
            y=lp_result.y,
            dual=lp_result.dual,
            is_feasible=True,
            lp_status='optimal',
            solve_time=solve_time,
        )
        self._add_to_cache(key, result)
        return result

    def _add_to_cache(self, key, result):
        """添加结果到缓存，LRU 淘汰。"""
        if len(self._cache) >= self._cache_size:
            # 简单淘汰：移除最早的 10% 条目
            n_remove = max(1, self._cache_size // 10)
            for _k in list(self._cache.keys())[:n_remove]:
                del self._cache[_k]
        self._cache[key] = result

    def solve_lp(self, x: ndarray) -> Optional[LPResult]:
        """求解连续子问题 LP(x)。

        问题: max h^T y  s.t.  Gy <= b - Ax, y >= 0

        使用 SciPy HiGHS 求解器（最快开源 LP 求解器）。
        """
        inst = self.inst
        rhs = inst.b - inst.A @ x

        # 处理可能的负 rhs（某些约束已被违反）
        if np.any(rhs < -1e-10):
            return None

        # LP: max h^T y  =>  min (-h)^T y
        c_obj = -inst.h

        # 约束: G y <= rhs
        # 变量边界: y >= 0
        bounds = [(0, None) for _ in range(inst.p)]

        try:
            result = linprog(
                c=c_obj,
                A_ub=inst.G,
                b_ub=rhs,
                bounds=bounds,
                method='highs',
            )
        except Exception:
            return None

        if result.success:
            # 提取对偶价格（影子价格）
            dual = self._extract_dual(result)
            return LPResult(
                y=result.x,
                dual=dual,
                obj_val=-result.fun,  # 转回 max
            )
        return None

    def _extract_dual(self, lp_result) -> Optional[ndarray]:
        """从 linprog 结果提取对偶价格。

        scipy >= 1.9.0: result.ineqlin.marginals 直接获取
        旧版本: 退化方案
        """
        if hasattr(lp_result, 'ineqlin') and hasattr(lp_result.ineqlin, 'marginals'):
            dual = np.array(lp_result.ineqlin.marginals)
            # 确保非负（对应 <= 约束的影子价格）
            return np.maximum(dual, 0.0)
        # 退化：无法直接提取，返回 None
        return None

    def compute_benders_linear(self, dual: ndarray) -> ndarray:
        """计算 Benders 对偶引导线性项。

        l_cont = -A^T u*
        l_cont[i] > 0 意味着增大 x_i 对连续子问题有正面影响
        """
        return -self.inst.A.T @ dual
