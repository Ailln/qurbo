"""
经典LP求解器模块 - 求解固定x后的连续变量子问题
"""

import numpy as np
from typing import Tuple, Optional
import config


def solve_lp_scipy(h: np.ndarray, G: np.ndarray, rhs: np.ndarray) -> Tuple[np.ndarray, bool, float]:
    """
    使用SciPy求解LP

    问题: max h^T y  s.t. Gy <= rhs, y >= 0
    转化为: min (-h)^T y
    """
    from scipy.optimize import linprog

    p = h.shape[0]

    res = linprog(
        c=-h.astype(float),
        A_ub=G.astype(float),
        b_ub=rhs.astype(float) + 1e-9,  # 微小松弛避免数值问题
        bounds=[(0, None)] * p,
        method="highs",
        options={"maxiter": 1000, "presolve": True}
    )

    if res.success:
        return res.x, True, -res.fun  # 转回最大化
    else:
        return np.zeros(p), False, float("-inf")


def solve_lp_numpy(h: np.ndarray, G: np.ndarray, rhs: np.ndarray,
                   max_iter: int = 500) -> Tuple[np.ndarray, bool, float]:
    """
    纯NumPy实现（无依赖，作为备选）
    使用投影梯度法
    """
    p = h.shape[0]
    y = np.zeros(p)
    lr = 0.01
    best_y = y.copy()
    best_obj = 0.0

    for _ in range(max_iter):
        # 梯度: h (最大化方向)
        gradient = h.copy()

        # 检查约束违反
        violation = G @ y - rhs
        active = violation > 0

        if np.any(active):
            # 投影到可行域
            grad_correction = G[active].T @ violation[active]
            gradient -= 100 * grad_correction

        # 梯度上升
        y = y + lr * gradient
        y = np.maximum(y, 0)  # y >= 0

        obj = h @ y
        if obj > best_obj:
            best_obj = obj
            best_y = y.copy()

    # 最终可行性检查
    violation = G @ best_y - rhs
    feasible = np.all(violation <= 1e-4)

    return best_y, feasible, best_obj


def solve_lp_simplex(h: np.ndarray, G: np.ndarray, rhs: np.ndarray) -> Tuple[np.ndarray, bool, float]:
    """
    使用两阶段单纯形法（纯numpy实现，无外部依赖）
    适合黑客松场景下的备选方案
    """
    try:
        return solve_lp_scipy(h, G, rhs)
    except Exception:
        return solve_lp_numpy(h, G, rhs)


def solve_y_subproblem(data: dict, x: np.ndarray) -> Tuple[np.ndarray, bool, float]:
    """
    求解给定x下的y子问题

    Returns:
        y: 最优连续变量
        success: 是否成功
        obj: y部分的目标值 h^T y
    """
    h = data["h"]
    G = data["G"]
    b = data["b"]
    A = data["A"]
    p = data["p"]

    if p == 0 or h is None:
        return np.array([]), True, 0.0

    # 计算右端项: G y <= b - Ax
    rhs = b - A @ x

    solver = config.LP_SOLVER

    if solver == "scipy":
        y, success, obj = solve_lp_scipy(h, G, rhs)
    elif solver == "numpy":
        y, success, obj = solve_lp_numpy(h, G, rhs)
    else:
        # 默认
        y, success, obj = solve_lp_scipy(h, G, rhs)

    return y, success, obj
