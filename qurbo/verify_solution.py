import argparse
import numpy as np


def load_npz(path):
    return np.load(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--solution", required=True)
    args = parser.parse_args()

    inst = load_npz(args.input)
    sol = load_npz(args.solution)
    x = sol["x"].astype(float)
    y = sol["y"].astype(float)

    obj = float(x @ inst["Q"] @ x + inst["c"] @ x + inst["h"] @ y)
    max_binary = float(np.max(inst["B"] @ x - inst["b_prime"])) if inst["B"].size else 0.0
    max_mixed = float(np.max(inst["A"] @ x + inst["G"] @ y - inst["b"])) if inst["A"].size else 0.0
    min_y = float(np.min(y)) if y.size else 0.0
    feasible = max_binary <= 1e-7 and max_mixed <= 1e-7 and min_y >= -1e-8

    print(f"objective={obj:.12f}")
    print(f"stored_objective={float(sol['objective']):.12f}" if "objective" in sol.files else "stored_objective=NA")
    print(f"feasible={feasible}")
    print(f"max_binary_violation={max_binary:.12e}")
    print(f"max_mixed_violation={max_mixed:.12e}")
    print(f"min_y={min_y:.12e}")
    if "optimal_value" in inst.files:
        opt = float(inst["optimal_value"])
        gap = (opt - obj) / abs(opt)
        print(f"optimal_value={opt:.12f}")
        print(f"optimality_gap={gap:.12f}")
        print(f"optimality_gap_percent={100.0 * gap:.6f}")
    for key in [
        "qaoa_calls",
        "exact_calls",
        "sa_calls",
        "max_qubits",
        "iterations_done",
        "lp_eval_count",
        "accepted_count",
        "restart_count",
        "qaoa_improvement_count",
        "exact_improvement_count",
        "classical_improvement_count",
        "qaoa_agreement_mean",
        "elapsed_seconds",
    ]:
        if key in sol.files:
            val = sol[key]
            if np.asarray(val).shape == ():
                print(f"{key}={val.item()}")
            else:
                print(f"{key}={val}")


if __name__ == "__main__":
    main()
