"""
主入口 - 端到端求解Pipeline
用法：
    python main.py --data_dir ./data --output_dir ./output --mode sample
    python main.py --data_dir ./data --output_dir ./output --mode test
"""

import numpy as np
import argparse
import os
import sys
import time
import json

import config
from utils.data_loader import load_data, save_solution
from utils.blocking import coupling_strength_blocks, random_blocks
from core.benders import QuantumBendersSolver, FallbackSolver
from utils.validator import quick_validation_report, check_feasibility, compute_objective


def solve_single_instance(data: dict, output_dir: str, time_limit: float = None) -> dict:
    """
    求解单个MIQP实例

    Args:
        data: 问题数据
        output_dir: 输出目录
        time_limit: 时间限制（秒）

    Returns:
        结果字典
    """
    prefix = data["prefix"]
    print(f"\n{'='*60}")
    print(f"求解实例: {prefix}")
    print(f"规模: n={data['n']}, p={data['p']}, m={data['m']}, m2={data['m2']}")
    print(f"{'='*60}")

    time_limit = time_limit or config.MAX_TIME_PER_TEST

    # 对于小规模问题，直接用穷举
    if data["n"] <= 15:
        result = solve_small_instance(data)
    else:
        # 中大规模：Benders分解
        try:
            solver = QuantumBendersSolver(
                data,
                block_size=min(config.BLOCK_SIZE, 15),
                n_qaoa_blocks=min(3, max(1, data["n"] // 30))
            )
            result = solver.solve(
                max_iter=min(50, max(10, 200 // data["n"])),
                time_limit=time_limit
            )
        except Exception as e:
            print(f"[ERROR] Benders求解失败: {e}")
            print("[FALLBACK] 使用贪心启发式")
            solver = FallbackSolver(data)
            result = solver.solve(time_limit=min(time_limit, 300))

    # 保存结果
    x = result["x"]
    y = result["y"]
    obj = result["objective"]

    save_solution(output_dir, prefix, x, y, obj, {
        "time": result.get("time", 0),
        "iterations": result.get("iterations", 0),
        "method": result.get("method", "benders_qaoa"),
        "feasible": result.get("feasible", False)
    })

    # 打印验证报告
    print(quick_validation_report(data, x, y))

    return result


def solve_small_instance(data: dict) -> dict:
    """
    求解小规模问题(n<=15)：穷举+LP
    """
    from utils.blocking import evaluate_partial_objective
    from core.lp_solver import solve_y_subproblem
    from core.quantum_solver import solve_qubo_exact

    Q, c = data["Q"], data["c"]
    n = data["n"]

    print(f"[SMALL] n={n}, using exact enumeration")

    # 穷举所有2^n个x
    best_obj = float("-inf")
    best_x = None
    best_y = None
    best_feasible = False

    for i in range(2 ** n):
        x = np.array([(i >> j) & 1 for j in range(n)], dtype=float)
        y, success, y_obj = solve_y_subproblem(data, x)

        if not success:
            continue

        obj = compute_objective(data, x, y)

        if obj > best_obj:
            best_obj = obj
            best_x = x.copy()
            best_y = y.copy()
            best_feasible = True

    if best_x is None:
        # 无可行解，返回全0
        best_x = np.zeros(n)
        best_y, _, _ = solve_y_subproblem(data, best_x)
        best_obj = compute_objective(data, best_x, best_y)

    return {
        "x": best_x,
        "y": best_y if best_y is not None else np.zeros(data["p"]),
        "objective": best_obj,
        "feasible": best_feasible,
        "history": {},
        "time": 0,
        "method": "exact_enumeration"
    }


def batch_solve(data_dir: str, output_dir: str, prefixes: list,
                time_per_instance: float = 1800) -> dict:
    """
    批量求解多个实例

    Args:
        data_dir: 数据目录
        output_dir: 输出目录
        prefixes: 实例前缀列表
        time_per_instance: 每个实例的时间限制

    Returns:
        汇总结果
    """
    results = {}
    total_start = time.time()

    for prefix in prefixes:
        instance_start = time.time()

        # 加载数据
        data = load_data(data_dir, prefix)

        # 动态调整时间预算
        remaining = len(prefixes) - len(results)
        if remaining > 1:
            time_budget = min(time_per_instance,
                            max(300, (config.MAX_TIME_PER_TEST * 5 - (time.time() - total_start)) / remaining))
        else:
            time_budget = time_per_instance

        print(f"[BATCH] {prefix}: time_budget={time_budget:.0f}s")

        # 求解
        result = solve_single_instance(data, output_dir, time_limit=time_budget)
        results[prefix] = {
            "objective": result["objective"],
            "feasible": result.get("feasible", False),
            "time": time.time() - instance_start,
            "method": result.get("method", "unknown")
        }

    # 保存汇总
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("批量求解完成")
    print(f"总时间: {time.time() - total_start:.1f}s")
    print(f"结果汇总: {summary_path}")
    for p, r in results.items():
        print(f"  {p}: obj={r['objective']:.6f}, feasible={r['feasible']}, time={r['time']:.1f}s")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="MIQP Quantum Solver")
    parser.add_argument("--data_dir", default="./data", help="数据目录")
    parser.add_argument("--output_dir", default="./output", help="输出目录")
    parser.add_argument("--mode", choices=["sample", "test", "single"], default="sample")
    parser.add_argument("--prefix", default=None, help="单实例模式下的前缀")
    parser.add_argument("--block_size", type=int, default=None, help="分块大小")
    parser.add_argument("--backend", default=None, choices=["qiskit", "pennylane"])
    parser.add_argument("--qaoa_reps", type=int, default=None)
    parser.add_argument("--max_time", type=float, default=None, help="每个实例最大时间(秒)")

    args = parser.parse_args()

    # 更新配置
    if args.block_size:
        config.BLOCK_SIZE = args.block_size
    if args.backend:
        config.QUANTUM_BACKEND = args.backend
    if args.qaoa_reps:
        config.QAOA_REPS = args.qaoa_reps
    if args.max_time:
        config.MAX_TIME_PER_TEST = args.max_time

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "single" and args.prefix:
        # 单实例模式
        data = load_data(args.data_dir, args.prefix)
        solve_single_instance(data, args.output_dir)

    elif args.mode == "sample":
        # Sample模式
        prefixes = ["sample_A", "sample_B"]
        batch_solve(args.data_dir, args.output_dir, prefixes,
                    time_per_instance=config.MAX_TIME_PER_TEST)

    elif args.mode == "test":
        # Test模式 - 自动发现所有test文件
        prefixes = []
        for f in sorted(os.listdir(args.data_dir)):
            if f.startswith("test_") and f.endswith("_Q.npy"):
                prefix = f.replace("_Q.npy", "")
                prefixes.append(prefix)

        if not prefixes:
            print("[WARNING] 未找到test数据，尝试默认列表")
            prefixes = ["test_1", "test_2", "test_3", "test_4", "test_5"]

        # 动态时间分配
        total_time = 3 * 3600  # 3小时
        time_per = max(300, (total_time - 600) / len(prefixes))  # 留10分钟给Paper

        batch_solve(args.data_dir, args.output_dir, prefixes,
                    time_per_instance=time_per)

    print("\n[OK] 全部完成！")


if __name__ == "__main__":
    main()
