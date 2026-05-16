#!/usr/bin/env python3
"""baseline_v4.run - 一键运行 MIQP 求解器。"""

import argparse
from pathlib import Path
import sys
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from baseline_v4.data.instance import MIQPInstance
    from baseline_v4.core.evaluator import ObjectiveEvaluator
    from baseline_v4.config import auto_config
    from baseline_v4.solver import HybridMIQPSolver
else:
    from .data.instance import MIQPInstance
    from .core.evaluator import ObjectiveEvaluator
    from .config import auto_config
    from .solver import HybridMIQPSolver


def main():
    parser = argparse.ArgumentParser(description='Hybrid MIQP Solver')
    parser.add_argument('--instance', type=str, required=True,
                       help='Path to .npz instance file')
    parser.add_argument('--time-limit', type=float, default=None,
                       help='Total time limit (seconds)')
    parser.add_argument('--max-qubits', type=int, default=20,
                       help='Maximum QAOA qubits (CPU simulation)')
    parser.add_argument('--sa-only', action='store_true',
                       help='Use Simulated Annealing only (no QAOA)')
    parser.add_argument('--device', type=str, default='CPU', choices=['CPU', 'GPU'],
                       help='AerSimulator device for QAOA (CPU or GPU)')
    parser.add_argument('--output', type=str, default='result.json',
                       help='Output result file')
    args = parser.parse_args()

    # 加载实例
    print(f"Loading instance from {args.instance}...")
    instance = MIQPInstance()
    instance.load(args.instance)
    instance.validate()

    diag = instance.diagnose()
    print(f"Instance: n={instance.n}, p={instance.p}, "
          f"m1={instance.m1}, m2={instance.m2}")
    print(f"Q density: {diag.Q_density:.3f}, "
          f"Recommended subQUBO: {diag.recommended_sub_qubo_size}")

    # 自动配置
    config = auto_config(instance)
    if args.time_limit:
        config.time_limit = args.time_limit
    if args.max_qubits:
        config.max_qubits = args.max_qubits
    if args.sa_only:
        config.max_qubits = 0  # 禁用 QAOA
    config.qaoa_device = args.device

    print(f"Config: time_limit={config.time_limit}s, "
          f"sub_qubo={config.sub_qubo_size}, max_qubits={config.max_qubits}")

    # 求解
    print("\nStarting solver...")
    solver = HybridMIQPSolver(instance, config)
    result = solver.solve()

    # 用 evaluator 重新精确验证最终解（与 baseline 评估标准一致）
    evaluator = ObjectiveEvaluator(instance)
    eval_r = evaluator.evaluate(result.best_x)
    final_obj = eval_r.objective if eval_r.is_feasible else -np.inf
    final_y = eval_r.y if eval_r.is_feasible and eval_r.y is not None else np.zeros(instance.p)
    final_feasible = eval_r.is_feasible

    # 输出结果
    print(f"\n{'='*50}")
    print(f"BEST OBJECTIVE: {final_obj:.6f}")
    print(f"TOTAL TIME: {result.total_time:.2f}s")
    print(f"ITERATIONS: {result.iterations_completed}")
    print(f"ELITE POOL: {result.elite_pool_final_size}")
    print(f"{'='*50}")

    # 保存结果（.npz 格式与 baseline 兼容）
    if args.output.endswith('.json'):
        import json
        output = {
            'best_objective': float(final_obj),
            'best_x': result.best_x.tolist(),
            'best_y': final_y.tolist(),
            'total_time': result.total_time,
            'iterations': result.iterations_completed,
            'convergence': [(t, float(obj)) for t, obj in result.convergence_history],
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
    else:
        # 保存为 .npz（与 baseline 统一格式）
        np.savez(
            args.output,
            x=result.best_x.astype(int),
            y=final_y.astype(float),
            objective=np.array(final_obj),
            feasible=np.array(final_feasible),
        )
    print(f"Result saved to {args.output}")


if __name__ == '__main__':
    main()
