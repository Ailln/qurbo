"""
解验证模块 - 快速检查解的可行性和目标值
"""

import numpy as np
from typing import Dict, Tuple


def check_feasibility(data: dict, x: np.ndarray, y: np.ndarray,
                      tol: float = 1e-6) -> Dict:
    """
    检查MIQP解的可行性

    Returns:
        {
            "feasible": bool,
            "binary_violation": float,  # 二元变量违反度
            "ineq_violation": float,   # 混合约束违反度
            "bineq_violation": float,  # 二元约束违反度
            "nonneg_violation": float, # y>=0违反度
            "details": dict
        }
    """
    Q = data["Q"]
    c = data["c"]
    h = data["h"]
    A = data["A"]
    G = data["G"]
    b = data["b"]
    B = data["B"]
    bp = data["bp"]

    results = {}

    # 1. 检查二元变量
    binary_violation = np.max(np.abs(x * (1 - x)))
    results["binary_violation"] = float(binary_violation)

    # 2. 检查混合约束 Ax + Gy <= b
    if A is not None and G is not None:
        lhs = A @ x + G @ y
        ineq_violation = np.max(np.maximum(0, lhs - b))
    else:
        ineq_violation = 0.0
    results["ineq_violation"] = float(ineq_violation)

    # 3. 检查二元约束 Bx <= bp
    if B is not None:
        blhs = B @ x
        bineq_violation = np.max(np.maximum(0, blhs - bp))
    else:
        bineq_violation = 0.0
    results["bineq_violation"] = float(bineq_violation)

    # 4. 检查y>=0
    if y is not None:
        nonneg_violation = np.max(np.maximum(0, -y))
    else:
        nonneg_violation = 0.0
    results["nonneg_violation"] = float(nonneg_violation)

    # 总可行性判断
    feasible = (binary_violation < tol and
                ineq_violation < tol and
                bineq_violation < tol and
                nonneg_violation < tol)
    results["feasible"] = feasible

    return results


def compute_objective(data: dict, x: np.ndarray, y: np.ndarray) -> float:
    """计算MIQP目标函数值"""
    Q = data["Q"]
    c = data["c"]
    h = data["h"]

    obj = x.T @ Q @ x + c @ x
    if h is not None and y is not None:
        obj += h.T @ y

    return float(obj)


def quick_validation_report(data: dict, x: np.ndarray, y: np.ndarray) -> str:
    """生成快速验证报告"""
    feas = check_feasibility(data, x, y)
    obj = compute_objective(data, x, y)

    report = f"""
{'='*50}
解验证报告 [{data['prefix']}]
{'='*50}
目标函数值: {obj:.6f}
可行性: {'✓ FEASIBLE' if feas['feasible'] else '✗ INFEASIBLE'}
  二元违反: {feas['binary_violation']:.2e}
  混合约束违反: {feas['ineq_violation']:.2e}
  二元约束违反: {feas['bineq_violation']:.2e}
  非负违反: {feas['nonneg_violation']:.2e}
{'='*50}
"""
    return report


def solve_lp_for_y(data: dict, x: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    固定x，求解关于y的LP子问题

    子问题：max h^T y
           s.t. G y <= b - Ax
                y >= 0

    Returns:
        y_opt: 最优y
        success: 是否成功
    """
    h = data["h"]
    G = data["G"]
    b = data["b"]
    A = data["A"]
    p = data["p"]

    if p == 0 or h is None:
        return np.array([]), True

    rhs = b - A @ x  # G y <= rhs

    try:
        from scipy.optimize import linprog
        # linprog做最小化，所以目标取负
        res = linprog(
            c=-h,  # 最大化 h^T y -> 最小化 -h^T y
            A_ub=G,
            b_ub=rhs,
            bounds=[(0, None)] * p,
            method="highs",
            options={"maxiter": 1000}
        )
        if res.success:
            return res.x, True
        else:
            return np.zeros(p), False
    except Exception as e:
        print(f"[WARNING] LP求解失败: {e}")
        return np.zeros(p), False
