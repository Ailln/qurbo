import argparse
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


@dataclass
class SubCandidate:
    bits: np.ndarray
    energy: float
    count: int
    source: str


def load_instance(path: str):
    raw = np.load(path)
    data = {k: raw[k] for k in raw.files}

    for key in ["n", "p", "m1", "m2"]:
        if key in data:
            data[key] = int(np.array(data[key]).item())

    return data


def objective_value(x, y, data):
    return float(x @ data["Q"] @ x + data["c"] @ x + data["h"] @ y)


def positive_violation(matrix, rhs, x):
    if matrix.size == 0:
        return 0.0, 0.0

    violation = np.maximum(matrix @ x - rhs, 0.0)
    return float(np.sum(violation)), float(np.max(violation))


def binary_constraint_violation(x, data):
    total, _max_v = positive_violation(data["B"], data["b_prime"], x)
    return total


def solve_continuous_subproblem(x, data):
    """
    固定二元变量 x 后，求连续变量 y。
    """
    A = data["A"]
    G = data["G"]
    b = data["b"]
    h = data["h"]
    p = data["p"]

    if binary_constraint_violation(x, data) > 1e-8:
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


def add_squared_linear_penalty(linear, const, penalty, l, pair):
    """
    给 QUBO 加 penalty * (const + sum_i linear_i z_i)^2。
    """
    k = len(linear)

    for i in range(k):
        l[i] += penalty * (2.0 * const * linear[i] + linear[i] ** 2)

    for i in range(k):
        for j in range(i + 1, k):
            pair[(i, j)] = pair.get((i, j), 0.0) + penalty * 2.0 * linear[i] * linear[j]


def add_active_inequality_penalties(
    matrix,
    rhs,
    x,
    subset,
    penalty,
    active_margin,
    l,
    pair,
):
    """
    只对当前已违反或接近边界的 inequality 加平方罚项。

    这样仍是一个近似 QUBO，但避免把深度可行的约束当成等式强行拉到边界。
    """
    if matrix.size == 0 or penalty <= 0:
        return 0

    active_count = 0
    x_subset = x[subset]
    lhs = matrix @ x

    for row, row_lhs, row_rhs in zip(matrix, lhs, rhs):
        slack = float(row_rhs - row_lhs)
        if slack > active_margin:
            continue

        linear = row[subset]
        const = float(row_lhs - linear @ x_subset - row_rhs)
        add_squared_linear_penalty(linear, const, penalty, l, pair)
        active_count += 1

    return active_count


def normalize_qubo(l, pair):
    max_abs = 0.0
    if len(l) > 0:
        max_abs = max(max_abs, float(np.max(np.abs(l))))

    if pair:
        max_abs = max(max_abs, max(abs(v) for v in pair.values()))

    if max_abs <= 0:
        return l.copy(), dict(pair), 1.0

    scaled_pair = {key: value / max_abs for key, value in pair.items()}
    return l / max_abs, scaled_pair, max_abs


def build_subqubo(
    x,
    subset,
    data,
    penalty=10.0,
    active_margin=1.0,
    mixed_penalty_scale=0.0,
):
    """
    构造局部 subQUBO。

    v2 的主要区别：
    - 纯二元约束只在当前违反或接近边界时加入罚项；
    - 混合约束默认不加入 QUBO 罚项，交给 LP 回代做真实可行性检查。
    """
    Q = data["Q"]
    c = data["c"]
    A = data["A"]
    b = data["b"]
    B = data["B"]
    b_prime = data["b_prime"]

    subset = np.array(subset, dtype=int)
    subset_set = set(subset.tolist())
    rest = np.array([i for i in range(len(x)) if i not in subset_set], dtype=int)

    k = len(subset)
    l = np.zeros(k, dtype=float)
    pair = {}

    Qss = Q[np.ix_(subset, subset)]

    if len(rest) > 0:
        Qsr = Q[np.ix_(subset, rest)]
        fixed_linear = 2.0 * Qsr @ x[rest]
    else:
        fixed_linear = np.zeros(k)

    for i in range(k):
        l[i] += -Qss[i, i]

    for i in range(k):
        for j in range(i + 1, k):
            pair[(i, j)] = pair.get((i, j), 0.0) - 2.0 * Qss[i, j]

    l += -(c[subset] + fixed_linear)

    active_binary = add_active_inequality_penalties(
        B,
        b_prime,
        x,
        subset,
        penalty,
        active_margin,
        l,
        pair,
    )

    active_mixed = add_active_inequality_penalties(
        A,
        b,
        x,
        subset,
        penalty * mixed_penalty_scale,
        active_margin,
        l,
        pair,
    )

    return l, pair, {"active_binary": active_binary, "active_mixed": active_mixed}


def qubo_energy(bits, l, pair):
    e = float(l @ bits)
    for (i, j), v in pair.items():
        e += float(v * bits[i] * bits[j])
    return e


def apply_cost_layer(qc, gamma, l, pair):
    for i, coeff in enumerate(l):
        qc.rz(-gamma * coeff, i)

    for (i, j), coeff in pair.items():
        qc.rz(-gamma * coeff / 2.0, i)
        qc.rz(-gamma * coeff / 2.0, j)
        qc.rzz(gamma * coeff / 2.0, i, j)


def apply_mixer_layer(qc, beta, k):
    for i in range(k):
        qc.rx(2.0 * beta, i)


def qaoa_like_collect_candidates(
    l,
    pair,
    shots=512,
    layers=1,
    device="CPU",
    top_k=20,
    candidate_pool=20,
):
    """
    采集多个 QAOA-like 候选，而不是只返回单个 bitstring。
    """
    k = len(l)
    if k == 0:
        return [SubCandidate(np.array([], dtype=int), 0.0, 1, "empty")]

    l_run, pair_run, _scale = normalize_qubo(l, pair)
    gammas = [0.2, 0.8, 1.4, 2.0]
    betas = [0.2, 0.6, 1.0]
    sim = AerSimulator(method="statevector", device=device)

    by_bits = {}

    for gamma in gammas:
        for beta in betas:
            qc = QuantumCircuit(k, k)
            for i in range(k):
                qc.h(i)

            for _ in range(layers):
                apply_cost_layer(qc, gamma, l_run, pair_run)
                apply_mixer_layer(qc, beta, k)

            qc.measure(range(k), range(k))

            result = sim.run(qc, shots=shots).result()
            counts = result.get_counts()
            sampled = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:candidate_pool]

            for bitstr, count in sampled:
                bits = np.array([int(v) for v in bitstr[::-1]], dtype=int)
                key = tuple(bits.tolist())
                energy = qubo_energy(bits, l_run, pair_run)
                current = by_bits.get(key)

                if current is None:
                    by_bits[key] = SubCandidate(bits=bits, energy=energy, count=count, source="qaoa")
                else:
                    current.count += count
                    current.energy = min(current.energy, energy)

    ranked = sorted(by_bits.values(), key=lambda item: (item.energy, -item.count))
    return ranked[:top_k]


def add_classical_candidates(candidates, current_bits, l, pair, rng, random_candidates):
    by_bits = {tuple(candidate.bits.tolist()): candidate for candidate in candidates}

    def add(bits, source):
        key = tuple(bits.tolist())
        if key not in by_bits:
            by_bits[key] = SubCandidate(
                bits=bits.copy(),
                energy=qubo_energy(bits, l, pair),
                count=1,
                source=source,
            )

    add(current_bits, "current")

    k = len(current_bits)
    for i in range(k):
        bits = current_bits.copy()
        bits[i] = 1 - bits[i]
        add(bits, "one_flip")

    for _ in range(random_candidates):
        bits = rng.integers(0, 2, size=k, dtype=int)
        add(bits, "random")

    return sorted(by_bits.values(), key=lambda item: (item.energy, -item.count))


def initialize_solution(data, random_trials=200, seed=42):
    rng = np.random.default_rng(seed)
    n = data["n"]
    candidates = [np.zeros(n, dtype=int)]

    for _ in range(random_trials):
        candidates.append(rng.integers(0, 2, size=n, dtype=int))

    best_x = None
    best_y = None
    best_obj = -np.inf
    best_feasible = False

    for x in candidates:
        feasible, y, obj = solve_continuous_subproblem(x, data)
        if feasible and obj > best_obj:
            best_x = x.copy()
            best_y = y.copy()
            best_obj = obj
            best_feasible = True

    if best_x is None:
        best_x = np.zeros(n, dtype=int)
        feasible, y, obj = solve_continuous_subproblem(best_x, data)
        if y is None:
            y = np.zeros(data["p"])
            obj = -np.inf

        best_y = y
        best_obj = obj
        best_feasible = feasible

    return best_x, best_y, best_obj, best_feasible


def normalized(values):
    values = np.asarray(values, dtype=float)
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    if max_abs <= 0:
        return np.zeros_like(values)
    return values / max_abs


def choose_subset(x, data, sub_size, rng):
    """
    用目标梯度、Q 耦合强度和约束压力混合选变量。
    """
    Q = data["Q"]
    c = data["c"]
    A = data["A"]
    b = data["b"]
    B = data["B"]
    b_prime = data["b_prime"]
    n = data["n"]

    gradient = np.abs(2.0 * Q @ x + c)
    coupling = np.sum(np.abs(Q), axis=1)
    pressure = np.zeros(n, dtype=float)

    if B.size > 0:
        residual = B @ x - b_prime
        active = residual >= -1.0
        if np.any(active):
            pressure += np.sum(np.abs(B[active]), axis=0)

    if A.size > 0:
        residual = A @ x - b
        active = residual >= -1.0
        if np.any(active):
            pressure += 0.25 * np.sum(np.abs(A[active]), axis=0)

    score = 0.50 * normalized(gradient) + 0.35 * normalized(coupling) + 0.15 * normalized(pressure)

    exploit_count = min(sub_size, max(1, int(np.ceil(0.7 * sub_size))))
    explore_count = sub_size - exploit_count
    pool_size = min(n, max(sub_size * 3, sub_size))
    pool = np.argsort(-score)[:pool_size]

    selected = []
    if exploit_count > 0:
        selected.extend(rng.choice(pool, size=exploit_count, replace=False).tolist())

    if explore_count > 0:
        remaining = np.array([i for i in range(n) if i not in set(selected)], dtype=int)
        selected.extend(rng.choice(remaining, size=explore_count, replace=False).tolist())

    return np.array(selected, dtype=int)


def evaluate_candidates(x, subset, candidates, data):
    best = {
        "x": None,
        "y": None,
        "obj": -np.inf,
        "feasible": False,
        "source": None,
        "energy": np.inf,
    }
    seen = set()
    evaluated = 0
    feasible_count = 0

    for candidate in candidates:
        new_x = x.copy()
        new_x[subset] = candidate.bits
        key = tuple(new_x.tolist())
        if key in seen:
            continue

        seen.add(key)
        evaluated += 1
        feasible, new_y, obj = solve_continuous_subproblem(new_x, data)

        if feasible:
            feasible_count += 1

        if feasible and obj > best["obj"]:
            best.update(
                {
                    "x": new_x,
                    "y": new_y,
                    "obj": obj,
                    "feasible": feasible,
                    "source": candidate.source,
                    "energy": candidate.energy,
                }
            )

    return best, evaluated, feasible_count


def run_baseline_v2(
    path,
    output_path="solution_baseline_v2.npz",
    iterations=20,
    sub_size=12,
    shots=512,
    layers=1,
    penalty=10.0,
    device="CPU",
    seed=42,
    top_k=20,
    candidate_pool=20,
    random_candidates=8,
    init_trials=200,
    active_margin=1.0,
    mixed_penalty_scale=0.0,
):
    data = load_instance(path)
    rng = np.random.default_rng(seed)
    n = data["n"]

    if sub_size > 30:
        raise ValueError("sub_size 不能超过 30。比赛要求单次量子调用不超过 30 qubit。")

    if sub_size > n:
        sub_size = n

    print(f"[INFO] instance = {path}")
    print(f"[INFO] n={data['n']}, p={data['p']}, m1={data['m1']}, m2={data['m2']}")
    print(
        f"[INFO] sub_size={sub_size}, iterations={iterations}, shots={shots}, "
        f"layers={layers}, device={device}"
    )
    print(
        f"[INFO] top_k={top_k}, candidate_pool={candidate_pool}, "
        f"random_candidates={random_candidates}, init_trials={init_trials}"
    )

    x, y, best_obj, feasible = initialize_solution(data, random_trials=init_trials, seed=seed)
    accepted_count = 0
    total_evaluated = 0
    total_feasible_candidates = 0

    print(f"[INIT] feasible={feasible}, objective={best_obj:.6f}")

    for it in range(1, iterations + 1):
        subset = choose_subset(x, data, sub_size, rng)
        l, pair, penalty_stats = build_subqubo(
            x,
            subset,
            data,
            penalty=penalty,
            active_margin=active_margin,
            mixed_penalty_scale=mixed_penalty_scale,
        )

        candidates = qaoa_like_collect_candidates(
            l,
            pair,
            shots=shots,
            layers=layers,
            device=device,
            top_k=top_k,
            candidate_pool=candidate_pool,
        )
        candidates = add_classical_candidates(
            candidates,
            x[subset],
            l,
            pair,
            rng,
            random_candidates=random_candidates,
        )
        candidates = candidates[: max(top_k, sub_size + random_candidates + 1)]

        best_candidate, evaluated, feasible_candidates = evaluate_candidates(
            x,
            subset,
            candidates,
            data,
        )
        total_evaluated += evaluated
        total_feasible_candidates += feasible_candidates

        candidate_obj = best_candidate["obj"]
        accepted = False
        source = best_candidate["source"] or "none"

        if best_candidate["feasible"] and candidate_obj > best_obj:
            x = best_candidate["x"]
            y = best_candidate["y"]
            best_obj = candidate_obj
            feasible = True
            accepted = True
            accepted_count += 1

        print(
            f"[ITER {it:03d}] "
            f"evaluated={evaluated} "
            f"feasible_candidates={feasible_candidates} "
            f"candidate_obj={candidate_obj:.6f} "
            f"best_obj={best_obj:.6f} "
            f"accepted={accepted} "
            f"source={source} "
            f"active_B={penalty_stats['active_binary']} "
            f"active_A={penalty_stats['active_mixed']}"
        )

    np.savez(
        output_path,
        x=x.astype(int),
        y=y.astype(float),
        objective=np.array(best_obj),
        feasible=np.array(feasible),
        accepted_count=np.array(accepted_count),
        evaluated_count=np.array(total_evaluated),
        feasible_candidate_count=np.array(total_feasible_candidates),
    )

    print(f"[DONE] saved to {output_path}")
    print(f"[DONE] best objective = {best_obj:.6f}")
    print(f"[DONE] feasible = {feasible}")
    print(f"[DONE] accepted_count = {accepted_count}")
    print(f"[DONE] evaluated_count = {total_evaluated}")
    print(f"[DONE] feasible_candidate_count = {total_feasible_candidates}")

    return x, y, best_obj, feasible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to miqp_xxx.npz")
    parser.add_argument("--output", default="solution_baseline_v2.npz")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--sub-size", type=int, default=12)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--penalty", type=float, default=10.0)
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-pool", type=int, default=20)
    parser.add_argument("--random-candidates", type=int, default=8)
    parser.add_argument("--init-trials", type=int, default=200)
    parser.add_argument("--active-margin", type=float, default=1.0)
    parser.add_argument("--mixed-penalty-scale", type=float, default=0.0)

    args = parser.parse_args()

    run_baseline_v2(
        path=args.input,
        output_path=args.output,
        iterations=args.iterations,
        sub_size=args.sub_size,
        shots=args.shots,
        layers=args.layers,
        penalty=args.penalty,
        device=args.device,
        seed=args.seed,
        top_k=args.top_k,
        candidate_pool=args.candidate_pool,
        random_candidates=args.random_candidates,
        init_trials=args.init_trials,
        active_margin=args.active_margin,
        mixed_penalty_scale=args.mixed_penalty_scale,
    )


if __name__ == "__main__":
    main()
