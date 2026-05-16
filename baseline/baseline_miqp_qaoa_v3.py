import argparse
from dataclasses import dataclass, field

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


@dataclass
class EvalResult:
    feasible: bool
    y: np.ndarray | None
    obj: float
    recourse_value: float
    dual: np.ndarray | None = None


@dataclass
class RecourseModel:
    n: int
    ridge: float = 1e-3
    max_samples: int = 300
    dual_linear: np.ndarray = field(init=False)
    regression_linear: np.ndarray = field(init=False)
    samples_x: list[np.ndarray] = field(default_factory=list)
    samples_value: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.dual_linear = np.zeros(self.n, dtype=float)
        self.regression_linear = np.zeros(self.n, dtype=float)

    def update(self, x, recourse_value, dual, data):
        self.samples_x.append(x.astype(float).copy())
        self.samples_value.append(float(recourse_value))

        if len(self.samples_x) > self.max_samples:
            self.samples_x.pop(0)
            self.samples_value.pop(0)

        if dual is not None and len(dual) == data["m1"]:
            estimate = -(data["A"].T @ dual)
            self.dual_linear = 0.8 * self.dual_linear + 0.2 * estimate

        if len(self.samples_x) >= min(self.n + 1, 20):
            self._fit_regression()

    def _fit_regression(self):
        x_mat = np.vstack(self.samples_x)
        values = np.asarray(self.samples_value, dtype=float)
        centered_x = x_mat - np.mean(x_mat, axis=0, keepdims=True)
        centered_y = values - float(np.mean(values))

        gram = centered_x.T @ centered_x
        rhs = centered_x.T @ centered_y
        reg = self.ridge * (float(np.trace(gram)) / max(1, self.n) + 1.0)

        try:
            self.regression_linear = np.linalg.solve(
                gram + reg * np.eye(self.n),
                rhs,
            )
        except np.linalg.LinAlgError:
            self.regression_linear = np.linalg.lstsq(
                gram + reg * np.eye(self.n),
                rhs,
                rcond=None,
            )[0]

    def linear(self):
        if np.any(np.abs(self.regression_linear) > 1e-12):
            return 0.7 * self.dual_linear + 0.3 * self.regression_linear
        return self.dual_linear.copy()


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


def solve_continuous_subproblem_with_dual(x, data):
    A = data["A"]
    G = data["G"]
    b = data["b"]
    h = data["h"]
    p = data["p"]

    if binary_constraint_violation(x, data) > 1e-8:
        return EvalResult(False, None, -np.inf, -np.inf, None)

    rhs = b - A @ x
    res = linprog(
        c=-h,
        A_ub=G,
        b_ub=rhs,
        bounds=[(0, None)] * p,
        method="highs",
    )

    if not res.success:
        return EvalResult(False, None, -np.inf, -np.inf, None)

    y = res.x
    recourse_value = float(h @ y)
    obj = objective_value(x, y, data)
    dual = None

    if hasattr(res, "ineqlin") and hasattr(res.ineqlin, "marginals"):
        # linprog minimizes -h@y; for the original max LP, lambda = -marginals.
        dual = np.maximum(-np.asarray(res.ineqlin.marginals, dtype=float), 0.0)

    return EvalResult(True, y, obj, recourse_value, dual)


def add_squared_linear_penalty(linear, const, penalty, l, pair):
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


def normalize_vector(values):
    values = np.asarray(values, dtype=float)
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    if max_abs <= 0:
        return np.zeros_like(values)
    return values / max_abs


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


def build_surrogate_subqubo(
    x,
    subset,
    data,
    recourse_linear,
    penalty=10.0,
    active_margin=1.0,
    mixed_penalty_scale=0.05,
):
    Q = data["Q"]
    c_eff = data["c"] + recourse_linear
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

    l += -(c_eff[subset] + fixed_linear)

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
    top_k=30,
    candidate_pool=30,
):
    k = len(l)
    if k == 0:
        return [SubCandidate(np.array([], dtype=int), 0.0, 1, "empty")]

    l_run, pair_run, _scale = normalize_qubo(l, pair)
    gammas = [0.15, 0.45, 0.9, 1.4, 2.0]
    betas = [0.15, 0.4, 0.75, 1.1]
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

    two_flip_limit = min(k * 2, 40)
    for _ in range(two_flip_limit):
        if k < 2:
            break
        i, j = rng.choice(k, size=2, replace=False)
        bits = current_bits.copy()
        bits[i] = 1 - bits[i]
        bits[j] = 1 - bits[j]
        add(bits, "two_flip")

    for _ in range(random_candidates):
        bits = rng.integers(0, 2, size=k, dtype=int)
        add(bits, "random")

    return sorted(by_bits.values(), key=lambda item: (item.energy, -item.count))


def build_interaction_graph(data, q_weight=0.65, a_weight=0.20, b_weight=0.15):
    Q = data["Q"]
    A = data["A"]
    B = data["B"]
    n = data["n"]

    graph = q_weight * np.abs(Q)

    if A.size > 0:
        graph += a_weight * normalize_matrix(np.abs(A.T @ A))

    if B.size > 0:
        graph += b_weight * normalize_matrix(np.abs(B.T @ B))

    graph = normalize_matrix(graph)
    graph[np.diag_indices(n)] = 0.0
    return graph


def normalize_matrix(matrix):
    max_abs = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    if max_abs <= 0:
        return np.zeros_like(matrix, dtype=float)
    return matrix / max_abs


def compute_variable_scores(x, data, recourse_linear, success_score):
    Q = data["Q"]
    A = data["A"]
    b = data["b"]
    B = data["B"]
    b_prime = data["b_prime"]
    n = data["n"]

    gradient = np.abs(2.0 * Q @ x + data["c"] + recourse_linear)
    coupling = np.sum(np.abs(Q), axis=1)
    pressure = np.zeros(n, dtype=float)

    if B.size > 0:
        residual = B @ x - b_prime
        active = residual >= -1.0
        violated = residual > 0.0
        if np.any(active):
            pressure += np.sum(np.abs(B[active]), axis=0)
        if np.any(violated):
            pressure += 2.0 * np.sum(np.abs(B[violated]), axis=0)

    if A.size > 0:
        residual = A @ x - b
        active = residual >= -1.0
        if np.any(active):
            pressure += 0.35 * np.sum(np.abs(A[active]), axis=0)

    score = (
        0.42 * normalize_vector(gradient)
        + 0.25 * normalize_vector(coupling)
        + 0.23 * normalize_vector(pressure)
        + 0.10 * normalize_vector(success_score)
    )
    return score


def choose_cluster_subset(x, data, graph, recourse_linear, success_score, sub_size, rng, iteration):
    n = data["n"]
    if sub_size >= n:
        return np.arange(n, dtype=int)

    score = compute_variable_scores(x, data, recourse_linear, success_score)
    pool_size = min(n, max(sub_size * 4, sub_size))
    pool = np.argsort(-score)[:pool_size]

    if iteration % 7 == 0:
        center = int(rng.integers(0, n))
    else:
        weights = np.maximum(score[pool], 0.0) + 1e-9
        weights = weights / np.sum(weights)
        center = int(rng.choice(pool, p=weights))

    selected = [center]
    selected_set = {center}

    while len(selected) < sub_size:
        candidates = np.array([i for i in range(n) if i not in selected_set], dtype=int)
        if candidates.size == 0:
            break

        affinity = np.max(graph[np.ix_(candidates, selected)], axis=1)
        candidate_scores = 0.65 * normalize_vector(affinity) + 0.35 * normalize_vector(score[candidates])
        best_idx = int(candidates[int(np.argmax(candidate_scores))])
        selected.append(best_idx)
        selected_set.add(best_idx)

    if iteration % 5 == 0 and len(selected) > 2:
        replace_count = max(1, sub_size // 5)
        kept = selected[:-replace_count]
        kept_set = set(kept)
        remaining = np.array([i for i in range(n) if i not in kept_set], dtype=int)
        random_tail = rng.choice(remaining, size=min(replace_count, len(remaining)), replace=False).tolist()
        selected = kept + random_tail

    return np.array(selected[:sub_size], dtype=int)


def repair_candidate(x, data, recourse_linear, max_steps=30):
    repaired = x.copy()
    steps = 0

    while steps < max_steps:
        total_b, max_b = positive_violation(data["B"], data["b_prime"], repaired)
        rhs = data["b"] - data["A"] @ repaired
        min_rhs = float(np.min(rhs)) if rhs.size else 0.0

        if total_b <= 1e-8 and min_rhs >= -1e-8:
            break

        one_indices = np.flatnonzero(repaired == 1)
        if one_indices.size == 0:
            break

        release = np.zeros_like(one_indices, dtype=float)
        if data["B"].size > 0:
            residual_b = data["B"] @ repaired - data["b_prime"]
            active_b = residual_b > -1e-8
            if np.any(active_b):
                release += np.sum(np.maximum(data["B"][active_b][:, one_indices], 0.0), axis=0)

        if data["A"].size > 0:
            rhs = data["b"] - data["A"] @ repaired
            tight_a = rhs < 0.5
            if np.any(tight_a):
                release += 0.5 * np.sum(np.maximum(data["A"][tight_a][:, one_indices], 0.0), axis=0)

        objective_loss = np.abs(2.0 * (data["Q"][one_indices] @ repaired) + data["c"][one_indices] + recourse_linear[one_indices])
        ratio = release / (objective_loss + 1e-6)

        if float(np.max(release)) <= 0.0:
            flip = int(one_indices[int(np.argmin(objective_loss))])
        else:
            flip = int(one_indices[int(np.argmax(ratio))])

        repaired[flip] = 0
        steps += 1

    return repaired, steps


class EvaluationCache:
    def __init__(self, data, recourse_model):
        self.data = data
        self.recourse_model = recourse_model
        self.cache = {}
        self.lp_evals = 0

    def evaluate(self, x):
        key = tuple(x.astype(int).tolist())
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        result = solve_continuous_subproblem_with_dual(x, self.data)
        self.cache[key] = result
        self.lp_evals += 1

        if result.feasible:
            self.recourse_model.update(x, result.recourse_value, result.dual, self.data)

        return result


def initialize_solution(data, evaluator, random_trials=300, seed=42):
    rng = np.random.default_rng(seed)
    n = data["n"]
    candidates = [np.zeros(n, dtype=int)]

    for density in (0.25, 0.50, 0.75):
        for _ in range(max(1, random_trials // 6)):
            candidates.append((rng.random(n) < density).astype(int))

    for _ in range(random_trials - len(candidates) + 1):
        candidates.append(rng.integers(0, 2, size=n, dtype=int))

    best_x = None
    best_result = None

    for x in candidates:
        repaired, _steps = repair_candidate(x, data, np.zeros(n), max_steps=max(10, n // 2))
        for candidate in (x, repaired):
            result = evaluator.evaluate(candidate)
            if result.feasible and (best_result is None or result.obj > best_result.obj):
                best_x = candidate.copy()
                best_result = result

    if best_x is None:
        best_x = np.zeros(n, dtype=int)
        best_result = evaluator.evaluate(best_x)

    if best_result.y is None:
        best_result = EvalResult(False, np.zeros(data["p"]), -np.inf, -np.inf, None)

    return best_x, best_result


def evaluate_candidate_pool(
    x,
    subset,
    candidates,
    data,
    evaluator,
    recourse_linear,
    repair_steps,
):
    best = {
        "x": None,
        "result": EvalResult(False, None, -np.inf, -np.inf, None),
        "source": None,
        "energy": np.inf,
        "repair_steps": 0,
    }
    seen = set()
    evaluated = 0
    feasible_count = 0
    repaired_count = 0

    for candidate in candidates:
        new_x = x.copy()
        new_x[subset] = candidate.bits

        full_candidates = [(new_x, 0)]
        repaired_x, steps = repair_candidate(new_x, data, recourse_linear, max_steps=repair_steps)
        if steps > 0:
            full_candidates.append((repaired_x, steps))
            repaired_count += 1

        for full_x, used_steps in full_candidates:
            key = tuple(full_x.tolist())
            if key in seen:
                continue

            seen.add(key)
            evaluated += 1
            result = evaluator.evaluate(full_x)

            if result.feasible:
                feasible_count += 1

            if result.feasible and result.obj > best["result"].obj:
                best.update(
                    {
                        "x": full_x.copy(),
                        "result": result,
                        "source": candidate.source,
                        "energy": candidate.energy,
                        "repair_steps": used_steps,
                    }
                )

    return best, evaluated, feasible_count, repaired_count


def update_elite_pool(elite_pool, x, result, max_elites=8):
    if not result.feasible:
        return elite_pool

    by_key = {tuple(item[0].tolist()): item for item in elite_pool}
    by_key[tuple(x.tolist())] = (x.copy(), result)
    ranked = sorted(by_key.values(), key=lambda item: item[1].obj, reverse=True)
    return ranked[:max_elites]


def choose_restart_solution(elite_pool, data, evaluator, rng):
    if elite_pool and rng.random() < 0.75:
        base_x = elite_pool[int(rng.integers(0, len(elite_pool)))][0].copy()
        flips = max(1, data["n"] // 20)
        indices = rng.choice(data["n"], size=flips, replace=False)
        base_x[indices] = 1 - base_x[indices]
        repaired, _steps = repair_candidate(base_x, data, np.zeros(data["n"]), max_steps=max(10, data["n"] // 3))
        result = evaluator.evaluate(repaired)
        if result.feasible:
            return repaired, result

    random_x = rng.integers(0, 2, size=data["n"], dtype=int)
    repaired, _steps = repair_candidate(random_x, data, np.zeros(data["n"]), max_steps=max(10, data["n"] // 2))
    result = evaluator.evaluate(repaired)
    return repaired, result


def run_baseline_v3(
    path,
    output_path="solution_baseline_v3.npz",
    iterations=120,
    sub_size=20,
    shots=512,
    layers=1,
    penalty=10.0,
    device="CPU",
    seed=42,
    top_k=30,
    candidate_pool=30,
    random_candidates=10,
    init_trials=300,
    active_margin=1.0,
    mixed_penalty_scale=0.05,
    repair_steps=30,
    stagnation_limit=25,
    temperature=0.01,
):
    data = load_instance(path)
    rng = np.random.default_rng(seed)
    n = data["n"]

    if sub_size > 30:
        raise ValueError("sub_size 不能超过 30。比赛要求单次量子调用不超过 30 qubit。")

    if sub_size > n:
        sub_size = n

    recourse_model = RecourseModel(n=n)
    evaluator = EvaluationCache(data, recourse_model)
    graph = build_interaction_graph(data)
    success_score = np.zeros(n, dtype=float)

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

    x, current_result = initialize_solution(data, evaluator, random_trials=init_trials, seed=seed)
    best_x = x.copy()
    best_result = current_result
    elite_pool = update_elite_pool([], best_x, best_result)
    best_trace = [best_result.obj]

    accepted_count = 0
    restart_count = 0
    total_evaluated = 0
    total_feasible_candidates = 0
    total_repaired_candidates = 0
    stale = 0

    print(f"[INIT] feasible={best_result.feasible}, objective={best_result.obj:.6f}")

    for it in range(1, iterations + 1):
        recourse_linear = recourse_model.linear()
        subset = choose_cluster_subset(
            x,
            data,
            graph,
            recourse_linear,
            success_score,
            sub_size,
            rng,
            it,
        )
        l, pair, penalty_stats = build_surrogate_subqubo(
            x,
            subset,
            data,
            recourse_linear,
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
        candidates = candidates[: max(top_k, sub_size + random_candidates + 10)]

        best_candidate, evaluated, feasible_candidates, repaired_candidates = evaluate_candidate_pool(
            x,
            subset,
            candidates,
            data,
            evaluator,
            recourse_linear,
            repair_steps=repair_steps,
        )
        total_evaluated += evaluated
        total_feasible_candidates += feasible_candidates
        total_repaired_candidates += repaired_candidates

        candidate_result = best_candidate["result"]
        accepted = False
        improved_best = False
        source = best_candidate["source"] or "none"

        if candidate_result.feasible:
            candidate_x = best_candidate["x"]
            is_new_state = candidate_x is not None and not np.array_equal(candidate_x, x)
            delta_current = candidate_result.obj - current_result.obj
            accept_worse = False
            accept_equal = False
            if is_new_state and delta_current < -1e-9 and temperature > 0:
                scale = max(1.0, abs(current_result.obj))
                accept_worse = rng.random() < np.exp(delta_current / (temperature * scale))
            elif is_new_state and abs(delta_current) <= 1e-9:
                accept_equal = rng.random() < 0.2

            if delta_current > 1e-9 or accept_worse or accept_equal:
                old_x = x.copy()
                x = candidate_x.copy()
                current_result = candidate_result
                accepted = True
                accepted_count += 1
                changed = np.flatnonzero(old_x != x)
                success_score[changed] += 1.0
                success_score *= 0.98

            if candidate_result.obj > best_result.obj + 1e-9:
                best_x = best_candidate["x"].copy()
                best_result = candidate_result
                elite_pool = update_elite_pool(elite_pool, best_x, best_result)
                stale = 0
                improved_best = True
            else:
                stale += 1
        else:
            stale += 1

        if stale >= stagnation_limit:
            restart_x, restart_result = choose_restart_solution(elite_pool, data, evaluator, rng)
            if restart_result.feasible:
                x = restart_x
                current_result = restart_result
            else:
                x = best_x.copy()
                current_result = best_result
            restart_count += 1
            stale = 0

        best_trace.append(best_result.obj)

        print(
            f"[ITER {it:03d}] "
            f"evaluated={evaluated} "
            f"feasible_candidates={feasible_candidates} "
            f"repaired_candidates={repaired_candidates} "
            f"candidate_obj={candidate_result.obj:.6f} "
            f"current_obj={current_result.obj:.6f} "
            f"best_obj={best_result.obj:.6f} "
            f"accepted={accepted} "
            f"improved={improved_best} "
            f"source={source} "
            f"active_B={penalty_stats['active_binary']} "
            f"active_A={penalty_stats['active_mixed']} "
            f"stale={stale}"
        )

    y_out = best_result.y if best_result.y is not None else np.zeros(data["p"])
    np.savez(
        output_path,
        x=best_x.astype(int),
        y=y_out.astype(float),
        objective=np.array(best_result.obj),
        feasible=np.array(best_result.feasible),
        accepted_count=np.array(accepted_count),
        restart_count=np.array(restart_count),
        evaluated_count=np.array(total_evaluated),
        feasible_candidate_count=np.array(total_feasible_candidates),
        repaired_candidate_count=np.array(total_repaired_candidates),
        lp_eval_count=np.array(evaluator.lp_evals),
        best_trace=np.asarray(best_trace, dtype=float),
        seed=np.array(seed),
    )

    print(f"[DONE] saved to {output_path}")
    print(f"[DONE] best objective = {best_result.obj:.6f}")
    print(f"[DONE] feasible = {best_result.feasible}")
    print(f"[DONE] accepted_count = {accepted_count}")
    print(f"[DONE] restart_count = {restart_count}")
    print(f"[DONE] evaluated_count = {total_evaluated}")
    print(f"[DONE] feasible_candidate_count = {total_feasible_candidates}")
    print(f"[DONE] repaired_candidate_count = {total_repaired_candidates}")
    print(f"[DONE] lp_eval_count = {evaluator.lp_evals}")

    return best_x, y_out, best_result.obj, best_result.feasible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to miqp_xxx.npz")
    parser.add_argument("--output", default="solution_baseline_v3.npz")
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--sub-size", type=int, default=20)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--penalty", type=float, default=10.0)
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--candidate-pool", type=int, default=30)
    parser.add_argument("--random-candidates", type=int, default=10)
    parser.add_argument("--init-trials", type=int, default=300)
    parser.add_argument("--active-margin", type=float, default=1.0)
    parser.add_argument("--mixed-penalty-scale", type=float, default=0.05)
    parser.add_argument("--repair-steps", type=int, default=30)
    parser.add_argument("--stagnation-limit", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=0.01)

    args = parser.parse_args()

    run_baseline_v3(
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
        repair_steps=args.repair_steps,
        stagnation_limit=args.stagnation_limit,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
