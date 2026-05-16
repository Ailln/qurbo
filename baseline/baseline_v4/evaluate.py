#!/usr/bin/env python3
"""统一评估脚本：对比多种求解方法的结果。

用法:
    python -m baseline.baseline_v4.evaluate --instance data/alpha-test/miqp_sample_A.npz \
        --sol path/to/solution_A.npz path/to/solution_B.npz

输出:
    - 每个解的目标值、可行性、约束违反量
    - 方法对比表格
    - 与最优值（若已知）的 gap
"""

import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linprog

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_instance(path: str):
    raw = np.load(path)
    data = {k: raw[k] for k in raw.files}
    for key in ["n", "p", "m1", "m2"]:
        if key in data:
            data[key] = int(np.array(data[key]).item())
    return data


def evaluate_solution(x, data):
    """精确评估一个二元解 x。

    返回:
        feasible: bool
        objective: float
        y: ndarray 或 None
        binary_violation: float
        mixed_violation: float
        lp_status: str
    """
    Q = data["Q"]
    c = data["c"]
    h = data["h"]
    A = data["A"]
    G = data["G"]
    b = data["b"]
    B = data["B"]
    b_prime = data["b_prime"]
    p = data["p"]

    # 1. 纯二元约束违反
    binary_violation = 0.0
    if B.size > 0:
        v = B @ x - b_prime
        binary_violation = float(np.sum(np.maximum(v, 0.0)))

    if binary_violation > 1e-8:
        return False, -np.inf, None, binary_violation, 0.0, "binary_infeasible"

    # 2. 连续子问题
    rhs = b - A @ x
    if np.any(rhs < -1e-10):
        return False, -np.inf, None, binary_violation, 0.0, "mixed_infeasible"

    res = linprog(
        c=-h,
        A_ub=G,
        b_ub=rhs,
        bounds=[(0, None)] * p,
        method="highs",
    )

    if not res.success:
        return False, -np.inf, None, binary_violation, 0.0, "lp_infeasible"

    y = res.x
    objective = float(x @ Q @ x + c @ x + h @ y)

    # 3. 混合约束违反（在最优 y 下应满足，但再检查一下）
    mixed_violation = 0.0
    if A.size > 0 and G.size > 0:
        v = A @ x + G @ y - b
        mixed_violation = float(np.sum(np.maximum(v, 0.0)))

    return True, objective, y, binary_violation, mixed_violation, "optimal"


def load_solution(path: str):
    sol = np.load(path)
    x = sol["x"].astype(float)
    y = sol["y"] if "y" in sol else None
    obj = float(sol["objective"]) if "objective" in sol else None
    feasible = bool(sol["feasible"]) if "feasible" in sol else None
    return {
        "path": path,
        "x": x,
        "y": y,
        "claimed_obj": obj,
        "claimed_feasible": feasible,
    }


def print_separator():
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="统一评估 MIQP 解")
    parser.add_argument("--instance", required=True, help="实例 .npz 文件路径")
    parser.add_argument("--sol", nargs="+", required=True, help="解文件列表（.npz）")
    parser.add_argument("--optimal", type=float, default=None, help="已知最优值（用于计算 gap）")
    args = parser.parse_args()

    data = load_instance(args.instance)
    instance_name = Path(args.instance).stem
    print_separator()
    print(f"实例: {instance_name}")
    print(f"n={data['n']}, p={data['p']}, m1={data['m1']}, m2={data['m2']}")
    print_separator()

    results = []
    for sol_path in args.sol:
        sol = load_solution(sol_path)
        feasible, obj, y, bin_viol, mix_viol, status = evaluate_solution(sol["x"], data)

        method_name = Path(sol_path).stem
        results.append({
            "method": method_name,
            "feasible": feasible,
            "objective": obj,
            "claimed_obj": sol["claimed_obj"],
            "binary_violation": bin_viol,
            "mixed_violation": mix_viol,
            "status": status,
            "path": sol_path,
        })

    # 排序：按目标值降序（最大化问题）
    results.sort(key=lambda r: r["objective"] if r["feasible"] else -np.inf, reverse=True)

    # 打印对比表格
    print(f"\n{'方法':<25} {'目标值':>12} {'声称值':>12} {'可行':>6} {'二元违反':>12} {'混合违反':>12} {'状态':>15}")
    print("-" * 100)
    for r in results:
        obj_str = f"{r['objective']:.4f}" if r["feasible"] else "INFEASIBLE"
        claimed_str = f"{r['claimed_obj']:.4f}" if r["claimed_obj"] is not None else "N/A"
        feas_str = "YES" if r["feasible"] else "NO"
        print(
            f"{r['method']:<25} {obj_str:>12} {claimed_str:>12} "
            f"{feas_str:>6} {r['binary_violation']:>12.6f} {r['mixed_violation']:>12.6f} {r['status']:>15}"
        )

    # 与最优值对比
    if args.optimal is not None:
        print_separator()
        print(f"已知最优值: {args.optimal:.6f}")
        for r in results:
            if r["feasible"]:
                gap = (args.optimal - r["objective"]) / abs(args.optimal) * 100
                print(f"  {r['method']:<25} gap = {gap:+.4f}%")
        print_separator()

    # 最佳方法摘要
    best = results[0]
    print(f"\n[BEST] 最佳方法: {best['method']}")
    print(f"       目标值: {best['objective']:.6f}")
    print(f"       可行: {'YES' if best['feasible'] else 'NO'}")
    print_separator()


if __name__ == "__main__":
    main()
