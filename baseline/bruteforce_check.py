import argparse
import numpy as np
from itertools import product
from scipy.optimize import linprog


def load_instance(path):
    raw = np.load(path)
    data = {k: raw[k] for k in raw.files}
    for key in ["n", "p", "m1", "m2"]:
        data[key] = int(np.array(data[key]).item())
    return data


def objective_value(x, y, data):
    return float(x @ data["Q"] @ x + data["c"] @ x + data["h"] @ y)


def solve_y(x, data):
    A, G, b = data["A"], data["G"], data["b"]
    B, b_prime = data["B"], data["b_prime"]
    h = data["h"]
    p = data["p"]

    if B.size > 0:
        if np.any(B @ x - b_prime > 1e-8):
            return False, None, -np.inf

    rhs = b - A @ x

    res = linprog(
        c=-h,
        A_ub=G,
        b_ub=rhs,
        bounds=[(0, None)] * p,
        method="highs",
    )

    if not res.success:
        return False, None, -np.inf

    y = res.x
    obj = objective_value(x, y, data)
    return True, y, obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    data = load_instance(args.input)
    n = data["n"]

    if n > 25:
        raise ValueError(f"n: {n} 太大，不建议暴力枚举。")

    best_x = None
    best_y = None
    best_obj = -np.inf
    feasible_count = 0

    for bits in product([0, 1], repeat=n):
        x = np.array(bits, dtype=int)
        feasible, y, obj = solve_y(x, data)

        if feasible:
            feasible_count += 1

        if feasible and obj > best_obj:
            best_x = x.copy()
            best_y = y.copy()
            best_obj = obj

    print("feasible_count =", feasible_count)
    print("best_obj =", best_obj)
    print("best_x =", best_x)
    print("best_y =", best_y)


if __name__ == "__main__":
    main()