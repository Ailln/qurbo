import argparse
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog, minimize

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


@dataclass
class EvalResult:
    feasible: bool
    y: np.ndarray
    obj: float
    recourse_value: float
    dual: np.ndarray | None = None
    max_binary_violation: float = 0.0
    max_mixed_violation: float = 0.0


@dataclass
class Candidate:
    bits: np.ndarray
    energy: float
    source: str
    count: int = 1


@dataclass
class SolverStats:
    qaoa_calls: int = 0
    qaoa_calls_small: int = 0
    qaoa_calls_mid: int = 0
    qaoa_calls_large: int = 0
    exact_calls: int = 0
    sa_calls: int = 0
    lp_evals: int = 0
    cache_hits: int = 0
    qaoa_time: float = 0.0
    exact_time: float = 0.0
    sa_time: float = 0.0
    repair_count: int = 0
    accepted_count: int = 0
    restart_count: int = 0
    evaluated_count: int = 0
    feasible_candidate_count: int = 0
    qaoa_improvement_count: int = 0
    exact_improvement_count: int = 0
    classical_improvement_count: int = 0
    qaoa_agreement_rates: list[float] = field(default_factory=list)
    qubit_counts: list[int] = field(default_factory=list)

    def qaoa_agreement_mean(self):
        if not self.qaoa_agreement_rates:
            return 0.0
        return float(np.mean(self.qaoa_agreement_rates))

    def max_qubits(self):
        if not self.qubit_counts:
            return 0
        return int(np.max(self.qubit_counts))


def load_instance(path: str):
    raw = np.load(path)
    data = {key: raw[key] for key in raw.files}
    for key in ("n", "p", "m1", "m2"):
        if key in data:
            data[key] = int(np.array(data[key]).item())
    data["Q"] = 0.5 * (data["Q"] + data["Q"].T)
    return data


def objective_value(x, y, data):
    return float(x @ data["Q"] @ x + data["c"] @ x + data["h"] @ y)


def max_positive_violation(lhs_minus_rhs):
    if lhs_minus_rhs.size == 0:
        return 0.0
    return float(np.max(lhs_minus_rhs))


def is_integer_matrix(matrix, tol=1e-9):
    if matrix.size == 0:
        return True
    return bool(np.allclose(matrix, np.round(matrix), atol=tol))


class LPEvaluator:
    def __init__(self, data, cache_size=5000, dual_tol=1e-6):
        self.data = data
        self.cache_size = int(cache_size)
        self.dual_tol = float(dual_tol)
        self.cache: OrderedDict[tuple[int, ...], EvalResult] = OrderedDict()
        self.stats = SolverStats()
        self.integer_B = is_integer_matrix(data["B"])
        if self.integer_B:
            self.binary_rhs_for_repair = np.floor(data["b_prime"] + 1e-9)
        else:
            self.binary_rhs_for_repair = data["b_prime"].copy()

    def binary_violation(self, x, use_repair_rhs=False):
        B = self.data["B"]
        if B.size == 0:
            return 0.0
        rhs = self.binary_rhs_for_repair if use_repair_rhs else self.data["b_prime"]
        return max_positive_violation(B @ x - rhs)

    def evaluate(self, x):
        key = tuple(np.asarray(x, dtype=int).tolist())
        cached = self.cache.get(key)
        if cached is not None:
            self.cache.move_to_end(key)
            self.stats.cache_hits += 1
            return cached

        result = self._evaluate_uncached(np.asarray(x, dtype=int))
        self.cache[key] = result
        self.stats.lp_evals += 1
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return result

    def _evaluate_uncached(self, x):
        data = self.data
        B = data["B"]
        if B.size > 0:
            max_b = max_positive_violation(B @ x - data["b_prime"])
            if max_b > 1e-8:
                return EvalResult(False, np.zeros(data["p"]), -np.inf, -np.inf, None, max_b, 0.0)

        rhs = data["b"] - data["A"] @ x
        res = linprog(
            c=-data["h"],
            A_ub=data["G"],
            b_ub=rhs,
            bounds=[(0, None)] * data["p"],
            method="highs",
        )
        if not res.success:
            return EvalResult(False, np.zeros(data["p"]), -np.inf, -np.inf, None, 0.0, np.inf)

        y = np.asarray(res.x, dtype=float)
        recourse = float(data["h"] @ y)
        obj = objective_value(x, y, data)
        dual = None
        if hasattr(res, "ineqlin") and hasattr(res.ineqlin, "marginals"):
            dual_candidate = -np.asarray(res.ineqlin.marginals, dtype=float)
            if self._validate_dual(y, dual_candidate, x, recourse):
                dual = np.maximum(dual_candidate, 0.0)

        max_mixed = max_positive_violation(data["A"] @ x + data["G"] @ y - data["b"])
        max_binary = self.binary_violation(x, use_repair_rhs=False)
        feasible = max_binary <= 1e-8 and max_mixed <= 1e-7 and float(np.min(y)) >= -1e-8
        return EvalResult(feasible, y, obj, recourse, dual, max_binary, max_mixed)

    def _validate_dual(self, y, u, x, primal_value):
        data = self.data
        if u is None or u.shape[0] != data["m1"]:
            return False
        if np.any(u < -self.dual_tol):
            return False
        if np.any(data["G"].T @ u < data["h"] - 1e-5):
            return False
        rhs = data["b"] - data["A"] @ x
        dual_value = float(rhs @ u)
        scale = 1.0 + abs(float(primal_value))
        return abs(dual_value - float(primal_value)) <= 1e-4 * scale


class DualSurrogate:
    def __init__(self, n, m1, eta_ema=0.3, eta_resc=0.5):
        self.n = int(n)
        self.m1 = int(m1)
        self.eta_ema = float(eta_ema)
        self.eta_resc = float(eta_resc)
        self.u_ema = np.zeros(self.m1, dtype=float)
        self.has_dual = False

    def update(self, result):
        if result.dual is None:
            return
        if not self.has_dual:
            self.u_ema = result.dual.astype(float).copy()
            self.has_dual = True
        else:
            self.u_ema = self.eta_ema * result.dual + (1.0 - self.eta_ema) * self.u_ema

    def linear(self, data):
        if not self.has_dual:
            return np.zeros(self.n, dtype=float)
        return -(data["A"].T @ self.u_ema)

    def omega(self, data):
        if not self.has_dual:
            return np.ones(self.n, dtype=float)
        sens = np.abs(data["A"].T @ self.u_ema)
        max_sens = float(np.max(sens)) if sens.size else 0.0
        if max_sens <= 1e-12:
            return np.ones(self.n, dtype=float)
        return 1.0 + self.eta_resc * sens / max_sens


def normalize(values):
    values = np.asarray(values, dtype=float)
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    if max_abs <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return values / max_abs


def build_coupling_graph(data):
    Q = data["Q"]
    B = data["B"]
    W = normalize(np.abs(Q))
    if B.size:
        W = W + 0.5 * normalize(B.T @ B)
    W = normalize(W)
    np.fill_diagonal(W, 0.0)
    return W


def elite_frequency(elite_pool, n):
    if not elite_pool:
        return np.full(n, 0.5, dtype=float)
    xs = np.vstack([item[0] for item in elite_pool]).astype(float)
    return np.mean(xs, axis=0)


def surrogate_value(x, data, linear, omega=None):
    if omega is None:
        Q_use = data["Q"]
    else:
        scale = np.sqrt(np.outer(omega, omega))
        Q_use = data["Q"] * scale
    return float(x @ Q_use @ x + (data["c"] + linear) @ x)


def flip_gain_scores(x, data, linear, omega):
    Q_use = data["Q"] * np.sqrt(np.outer(omega, omega))
    base_grad = 2.0 * Q_use @ x + data["c"] + linear
    gains = np.where(x == 0, base_grad + np.diag(Q_use), -base_grad + np.diag(Q_use))
    return gains


def grow_subset(seed, W, scores, q_size, forbidden=None):
    n = W.shape[0]
    selected = [int(seed)]
    selected_set = {int(seed)}
    forbidden = set() if forbidden is None else set(int(v) for v in forbidden)
    while len(selected) < min(q_size, n):
        candidates = np.array([i for i in range(n) if i not in selected_set and i not in forbidden], dtype=int)
        if candidates.size == 0:
            candidates = np.array([i for i in range(n) if i not in selected_set], dtype=int)
        if candidates.size == 0:
            break
        affinity = np.max(W[np.ix_(candidates, selected)], axis=1)
        combined = 0.65 * normalize(affinity) + 0.35 * normalize(scores[candidates])
        best = int(candidates[int(np.argmax(combined))])
        selected.append(best)
        selected_set.add(best)
    return np.array(selected, dtype=int)


def generate_neighborhoods(x, data, W, linear, omega, elite_pool, q_size, rng, stuck):
    n = data["n"]
    q_size = min(q_size, n)
    gains = flip_gain_scores(x, data, linear, omega)
    freq = elite_frequency(elite_pool, n)
    uncertainty = 4.0 * freq * (1.0 - freq)
    score = 0.55 * normalize(np.maximum(gains, 0.0)) + 0.30 * normalize(uncertainty) + 0.15 * normalize(np.sum(W, axis=1))

    neighborhoods = []
    seed1 = int(np.argmax(score))
    neighborhoods.append(("exploit", grow_subset(seed1, W, score, q_size)))

    overlap_forbidden = set(neighborhoods[0][1].tolist())
    uncertainty_scores = uncertainty.copy()
    uncertainty_scores[list(overlap_forbidden)] *= 0.2
    seed2 = int(np.argmax(uncertainty_scores))
    neighborhoods.append(("uncertainty", grow_subset(seed2, W, uncertainty_scores, q_size, forbidden=overlap_forbidden)))

    random_subset = rng.choice(n, size=q_size, replace=False)
    neighborhoods.append(("random", np.asarray(random_subset, dtype=int)))

    if stuck >= 5:
        if elite_pool:
            distances = [np.sum(np.abs(x - item[0])) for item in elite_pool]
            far = elite_pool[int(np.argmax(distances))][0]
            diff = np.flatnonzero(x != far)
            if diff.size == 0:
                seed4 = int(rng.integers(0, n))
            else:
                seed4 = int(diff[int(rng.integers(0, diff.size))])
        else:
            seed4 = int(rng.integers(0, n))
        neighborhoods.append(("valley", grow_subset(seed4, W, score + rng.random(n) * 0.05, q_size)))

    deduped = []
    seen = set()
    for name, subset in neighborhoods:
        key = tuple(sorted(int(v) for v in subset.tolist()))
        if key not in seen:
            seen.add(key)
            deduped.append((name, subset))
    return deduped


def local_energy_terms(x, subset, data, linear, omega, lambda_B):
    subset = np.asarray(subset, dtype=int)
    rest = np.array([i for i in range(data["n"]) if i not in set(subset.tolist())], dtype=int)
    scale = np.sqrt(np.outer(omega, omega))
    Qhat = data["Q"] * scale
    Qss = Qhat[np.ix_(subset, subset)]
    fixed = np.zeros(len(subset), dtype=float)
    if rest.size:
        fixed = 2.0 * Qhat[np.ix_(subset, rest)] @ x[rest]
    d = data["c"][subset] + linear[subset] + fixed
    if data["B"].size and lambda_B.size:
        d = d - data["B"][:, subset].T @ lambda_B

    M = -Qss
    r = -d
    l = np.diag(M).copy() + r
    pair = {}
    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            coeff = 2.0 * M[i, j]
            if abs(coeff) > 1e-12:
                pair[(i, j)] = float(coeff)
    return l, pair


def qubo_energy(bits, l, pair):
    bits = np.asarray(bits, dtype=float)
    e = float(l @ bits)
    for (i, j), coeff in pair.items():
        e += float(coeff * bits[i] * bits[j])
    return e


def vectorized_bruteforce(l, pair, top_k=20):
    start = time.time()
    q = len(l)
    total = 1 << q
    ints = np.arange(total, dtype=np.uint32)
    bits = ((ints[:, None] >> np.arange(q, dtype=np.uint32)) & 1).astype(np.float64)
    energies = bits @ l
    if pair:
        for (i, j), coeff in pair.items():
            energies += coeff * bits[:, i] * bits[:, j]
    top_k = min(int(top_k), total)
    indices = np.argpartition(energies, top_k - 1)[:top_k]
    indices = indices[np.argsort(energies[indices])]
    candidates = [
        Candidate(bits=bits[idx].astype(int), energy=float(energies[idx]), source="exact", count=1)
        for idx in indices
    ]
    return candidates, time.time() - start


def qubo_to_ising(l, pair):
    q = len(l)
    h = -0.5 * np.asarray(l, dtype=float)
    J = {}
    const = 0.5 * float(np.sum(l))
    for (i, j), coeff in pair.items():
        J[(i, j)] = coeff / 4.0
        h[i] -= coeff / 4.0
        h[j] -= coeff / 4.0
        const += coeff / 4.0
    return h, J, const


def build_qaoa_circuit(q, l, pair, params, init_probs, measure=False):
    layers = len(params) // 2
    gammas = params[:layers]
    betas = params[layers:]
    h, J, _const = qubo_to_ising(l, pair)
    qc = QuantumCircuit(q, q if measure else 0)
    init_probs = np.clip(init_probs, 1e-4, 1.0 - 1e-4)
    for i, prob in enumerate(init_probs):
        qc.ry(2.0 * np.arcsin(np.sqrt(prob)), i)
    for gamma, beta in zip(gammas, betas):
        for i, coeff in enumerate(h):
            if abs(coeff) > 1e-12:
                qc.rz(2.0 * gamma * coeff, i)
        for (i, j), coeff in J.items():
            if abs(coeff) > 1e-12:
                qc.rzz(2.0 * gamma * coeff, i, j)
        for i in range(q):
            qc.rx(2.0 * beta, i)
    if measure:
        qc.measure(range(q), range(q))
    return qc


def all_energies_for_q(l, pair):
    q = len(l)
    total = 1 << q
    ints = np.arange(total, dtype=np.uint32)
    bits = ((ints[:, None] >> np.arange(q, dtype=np.uint32)) & 1).astype(np.float64)
    energies = bits @ l
    if pair:
        for (i, j), coeff in pair.items():
            energies += coeff * bits[:, i] * bits[:, j]
    return energies


def qaoa_expectation(params, q, l, pair, init_probs, energies):
    qc = build_qaoa_circuit(q, l, pair, params, init_probs, measure=False)
    probs = np.abs(Statevector.from_instruction(qc).data) ** 2
    return float(probs @ energies)


def qaoa_candidates(
    l,
    pair,
    init_probs,
    depth,
    shots,
    opt_steps,
    multistart,
    device,
    top_k,
    rng,
    exact_best=None,
    rhobeg=0.35,
):
    start = time.time()
    q = len(l)
    if q == 0:
        return [Candidate(np.array([], dtype=int), 0.0, "qaoa", 1)], 0.0, 0.0
    energies = all_energies_for_q(l, pair) if q <= 16 and opt_steps > 0 else None
    initial_params = []
    base_gamma = np.linspace(0.35, 0.85, depth)
    base_beta = np.linspace(0.25, 0.55, depth)
    initial_params.append(np.concatenate([base_gamma, base_beta]))
    for _ in range(max(0, multistart - 1)):
        initial_params.append(rng.uniform(0.05, 1.2, size=2 * depth))

    best_params = initial_params[0]
    best_value = np.inf
    if energies is not None:
        for params0 in initial_params:
            if opt_steps > 0:
                res = minimize(
                    lambda p: qaoa_expectation(p, q, l, pair, init_probs, energies),
                    params0,
                    method="COBYLA",
                    options={"maxiter": int(opt_steps), "rhobeg": float(rhobeg), "disp": False},
                )
                value = float(res.fun)
                params = np.asarray(res.x, dtype=float)
            else:
                params = params0
                value = qaoa_expectation(params, q, l, pair, init_probs, energies)
            if value < best_value:
                best_value = value
                best_params = params

    sim = AerSimulator(method="statevector", device=device)
    qc = build_qaoa_circuit(q, l, pair, best_params, init_probs, measure=True)
    try:
        counts = sim.run(qc, shots=shots).result().get_counts()
    except Exception:
        sim = AerSimulator(method="statevector", device="CPU")
        counts = sim.run(qc, shots=shots).result().get_counts()

    candidates_by_key = {}
    exact_count = 0
    total_count = 0
    exact_key = None if exact_best is None else tuple(exact_best.bits.tolist())
    for bitstr, count in counts.items():
        bits = np.array([int(v) for v in bitstr[::-1]], dtype=int)
        key = tuple(bits.tolist())
        energy = qubo_energy(bits, l, pair)
        total_count += int(count)
        if exact_key is not None and key == exact_key:
            exact_count += int(count)
        current = candidates_by_key.get(key)
        if current is None:
            candidates_by_key[key] = Candidate(bits=bits, energy=energy, source="qaoa", count=int(count))
        else:
            current.count += int(count)
            current.energy = min(current.energy, energy)
    ranked = sorted(candidates_by_key.values(), key=lambda item: (item.energy, -item.count))
    agreement = float(exact_count / total_count) if total_count and exact_key is not None else 0.0
    return ranked[:top_k], time.time() - start, agreement


def simulated_annealing(l, pair, rng, top_k=20, sweeps=400, starts=16):
    start = time.time()
    q = len(l)
    best = {}
    for _ in range(starts):
        bits = rng.integers(0, 2, size=q, dtype=int)
        energy = qubo_energy(bits, l, pair)
        temp0 = max(1.0, abs(energy))
        for step in range(sweeps):
            i = int(rng.integers(0, q))
            trial = bits.copy()
            trial[i] = 1 - trial[i]
            trial_energy = qubo_energy(trial, l, pair)
            temp = temp0 * (0.995 ** step) + 1e-6
            if trial_energy <= energy or rng.random() < np.exp((energy - trial_energy) / temp):
                bits = trial
                energy = trial_energy
        key = tuple(bits.tolist())
        if key not in best or energy < best[key].energy:
            best[key] = Candidate(bits=bits.copy(), energy=energy, source="sa", count=1)
    return sorted(best.values(), key=lambda item: item.energy)[:top_k], time.time() - start


def solve_subqubo_hybrid(l, pair, init_probs, args, rng, stats):
    q = len(l)
    if q > args.qaoa_qubits:
        raise ValueError(f"QAOA qubit count {q} exceeds limit {args.qaoa_qubits}")
    stats.qubit_counts.append(q)

    candidates = []
    exact_candidates = []
    exact_best = None

    if q <= 18:
        exact_candidates, exact_time = vectorized_bruteforce(l, pair, top_k=args.top_k)
        stats.exact_calls += 1
        stats.exact_time += exact_time
        exact_best = exact_candidates[0] if exact_candidates else None

    if q <= 12:
        depth, opt_steps, shots = args.qaoa_depth_small, args.qaoa_opt_steps, args.shots_small
        stats.qaoa_calls_small += 1
    elif q <= 16:
        depth, opt_steps, shots = args.qaoa_depth_medium, args.qaoa_opt_steps, args.shots_small
        stats.qaoa_calls_mid += 1
    else:
        depth, opt_steps, shots = args.qaoa_depth_large, args.qaoa_opt_steps_large, args.shots_large
        stats.qaoa_calls_large += 1

    qaoa, qaoa_time, agreement = qaoa_candidates(
        l,
        pair,
        init_probs,
        depth=depth,
        shots=shots,
        opt_steps=opt_steps,
        multistart=args.qaoa_multistart,
        device=args.device,
        top_k=args.top_k,
        rng=rng,
        exact_best=exact_best,
        rhobeg=args.qaoa_rhobeg,
    )
    stats.qaoa_calls += 1
    stats.qaoa_time += qaoa_time
    if exact_best is not None:
        stats.qaoa_agreement_rates.append(agreement)

    if q <= 12:
        candidates = qaoa + exact_candidates[: args.top_k]
    elif q <= 16:
        qaoa_best = qaoa[0] if qaoa else None
        if exact_best is not None and qaoa_best is not None:
            denom = max(1.0, abs(exact_best.energy))
            gap = (qaoa_best.energy - exact_best.energy) / denom
            if gap > 0.02:
                candidates = exact_candidates[: args.top_k] + qaoa[:10]
            else:
                candidates = qaoa + exact_candidates[:5]
        else:
            candidates = qaoa + exact_candidates[: args.top_k]
    else:
        candidates = exact_candidates[: args.top_k] + qaoa[:10]

    deduped = {}
    for candidate in candidates:
        key = tuple(candidate.bits.tolist())
        if key not in deduped or candidate.energy < deduped[key].energy:
            deduped[key] = candidate
    return sorted(deduped.values(), key=lambda item: (item.energy, -item.count))[: args.top_k]


def repair_binary(x, data, evaluator, linear, omega, max_flips):
    repaired = x.copy()
    flips = 0
    B = data["B"]
    if B.size == 0:
        return repaired, flips
    rhs = evaluator.binary_rhs_for_repair
    while flips < max_flips:
        violation = B @ repaired - rhs
        if float(np.max(violation)) <= 1e-8:
            break
        active_rows = violation > 1e-8
        ones = np.flatnonzero(repaired == 1)
        if ones.size == 0:
            break
        Q_use = data["Q"] * np.sqrt(np.outer(omega, omega))
        losses = []
        for i in ones:
            trial = repaired.copy()
            trial[i] = 0
            loss = surrogate_value(repaired, data, linear, omega) - surrogate_value(trial, data, linear, omega)
            release = np.sum(np.maximum(B[active_rows, i], 0.0))
            losses.append((loss / (release + 1e-6), i))
        _ratio, idx = min(losses, key=lambda item: item[0])
        repaired[idx] = 0
        flips += 1
    return repaired, flips


def repair_mixed_if_needed(x, data, evaluator, linear, omega, max_flips):
    result = evaluator.evaluate(x)
    if result.feasible:
        return x, result, 0
    repaired = x.copy()
    flips = 0
    while flips < max_flips:
        ones = np.flatnonzero(repaired == 1)
        if ones.size == 0:
            break
        losses = []
        for i in ones:
            trial = repaired.copy()
            trial[i] = 0
            loss = surrogate_value(repaired, data, linear, omega) - surrogate_value(trial, data, linear, omega)
            release = 0.0
            if data["A"].size:
                rhs = data["b"] - data["A"] @ repaired
                tight = rhs < 1e-8
                if np.any(tight):
                    release += np.sum(np.maximum(data["A"][tight, i], 0.0))
            losses.append((loss / (release + 1e-6), i))
        _ratio, idx = min(losses, key=lambda item: item[0])
        repaired[idx] = 0
        flips += 1
        result = evaluator.evaluate(repaired)
        if result.feasible:
            return repaired, result, flips
    return repaired, evaluator.evaluate(repaired), flips


def evaluate_candidates(base_x, subset, candidates, data, evaluator, linear, omega, args, stats):
    best_x = None
    best_result = EvalResult(False, np.zeros(data["p"]), -np.inf, -np.inf)
    best_source = "none"
    evaluated = 0
    feasible_count = 0
    seen = set()
    for candidate in candidates:
        x = base_x.copy()
        x[subset] = candidate.bits
        x, flips1 = repair_binary(x, data, evaluator, linear, omega, max_flips=args.repair_flip_limit)
        x, result, flips2 = repair_mixed_if_needed(x, data, evaluator, linear, omega, max_flips=args.repair_flip_limit)
        stats.repair_count += int(flips1 + flips2 > 0)
        key = tuple(x.tolist())
        if key in seen:
            continue
        seen.add(key)
        evaluated += 1
        if result.feasible:
            feasible_count += 1
            if result.obj > best_result.obj:
                best_x = x.copy()
                best_result = result
                best_source = candidate.source
    stats.evaluated_count += evaluated
    stats.feasible_candidate_count += feasible_count
    return best_x, best_result, best_source, evaluated, feasible_count


def update_elite(elite_pool, x, result, max_size=20):
    if not result.feasible:
        return elite_pool
    by_key = {tuple(item[0].tolist()): item for item in elite_pool}
    by_key[tuple(x.tolist())] = (x.copy(), result)
    ranked = sorted(by_key.values(), key=lambda item: item[1].obj, reverse=True)
    return ranked[:max_size]


def initialize_solution(data, evaluator, surrogate, args, rng):
    n = data["n"]
    if n <= args.exact_init_limit:
        candidates = []
        total = 1 << n
        ints = np.arange(total, dtype=np.uint32)
        bits_matrix = ((ints[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(int)
        for x in bits_matrix:
            if evaluator.binary_violation(x, use_repair_rhs=False) <= 1e-8:
                candidates.append(x.copy())
    else:
        candidates = [np.zeros(n, dtype=int)]
        zero_result = evaluator.evaluate(candidates[0])
        surrogate.update(zero_result)
        linear = surrogate.linear(data)
        score = data["c"] + np.diag(data["Q"]) + linear
        x_greedy = np.zeros(n, dtype=int)
        for idx in np.argsort(-score):
            trial = x_greedy.copy()
            trial[idx] = 1
            if evaluator.binary_violation(trial, use_repair_rhs=True) <= 1e-8:
                x_greedy = trial
        candidates.append(x_greedy)
        random_count = args.init_random_large if n > 40 else args.init_random_small
        for _ in range(random_count):
            order = rng.permutation(n)
            x = np.zeros(n, dtype=int)
            for idx in order:
                if rng.random() < 0.5:
                    trial = x.copy()
                    trial[idx] = 1
                    if evaluator.binary_violation(trial, use_repair_rhs=True) <= 1e-8:
                        x = trial
            candidates.append(x)

    best_x = np.zeros(n, dtype=int)
    best_result = evaluator.evaluate(best_x)
    elite_pool = update_elite([], best_x, best_result, args.elite_size)
    for x in candidates:
        result = evaluator.evaluate(x)
        if result.feasible:
            surrogate.update(result)
            elite_pool = update_elite(elite_pool, x, result, args.elite_size)
            if result.obj > best_result.obj:
                best_x = x.copy()
                best_result = result
    return best_x, best_result, elite_pool


def update_lambda(lambda_B, data, x, violation_streak, scale):
    if data["B"].size == 0:
        return lambda_B, violation_streak
    violation = data["B"] @ x - data["b_prime"]
    current = violation > 1e-8
    violation_streak[current] += 1
    violation_streak[~current] = 0
    lambda_B[violation_streak >= 2] += 0.1 * scale
    return lambda_B, violation_streak


def run_hybrid_v4(args):
    data = load_instance(args.input)
    rng = np.random.default_rng(args.seed)
    evaluator = LPEvaluator(data, cache_size=args.cache_size)
    stats = evaluator.stats
    surrogate = DualSurrogate(data["n"], data["m1"], eta_ema=args.eta_ema, eta_resc=args.eta_resc)
    W = build_coupling_graph(data)
    lambda_B = np.zeros(data["m2"], dtype=float)
    violation_streak = np.zeros(data["m2"], dtype=int)

    start_time = time.time()
    best_x, best_result, elite_pool = initialize_solution(data, evaluator, surrogate, args, rng)
    current_x = best_x.copy()
    current_result = best_result
    surrogate.update(current_result)
    best_trace = [best_result.obj]
    current_trace = [current_result.obj]
    source_trace = ["init"]
    q_size = min(args.initial_sub_size, args.q_max, data["n"])
    stuck = 0
    T0 = args.temperature_scale * max(1.0, abs(current_result.obj))

    print(f"[INFO] instance={args.input}")
    print(f"[INFO] n={data['n']}, p={data['p']}, m1={data['m1']}, m2={data['m2']}")
    print(f"[INFO] q_max={args.q_max}, qaoa_qubits={args.qaoa_qubits}, time_limit={args.time_limit_seconds}s")
    print(f"[INIT] feasible={best_result.feasible}, objective={best_result.obj:.6f}")

    iterations_done = 0
    for iteration in range(1, args.iterations + 1):
        elapsed = time.time() - start_time
        if args.time_limit_seconds > 0 and elapsed >= args.time_limit_seconds:
            print(f"[STOP] time limit reached at iteration {iteration}")
            break

        linear = surrogate.linear(data)
        omega = surrogate.omega(data)
        neighborhoods = generate_neighborhoods(current_x, data, W, linear, omega, elite_pool, q_size, rng, stuck)
        improved_this_iter = False
        best_iter_source = "none"
        best_iter_obj = -np.inf
        accepted = False

        remaining = args.time_limit_seconds - elapsed if args.time_limit_seconds > 0 else np.inf
        if remaining < 0.10 * args.time_limit_seconds:
            neighborhoods = neighborhoods[:2]
        if remaining < 0.05 * args.time_limit_seconds:
            neighborhoods = [item for item in neighborhoods if item[0] != "random"] or neighborhoods[:1]

        for neigh_name, subset in neighborhoods:
            subset = np.asarray(subset[: min(q_size, args.q_max)], dtype=int)
            if subset.size == 0:
                continue
            l, pair = local_energy_terms(current_x, subset, data, linear, omega, lambda_B)
            freq = elite_frequency(elite_pool, data["n"])
            init_probs = np.clip(freq[subset], 0.1, 0.9)
            candidates = solve_subqubo_hybrid(l, pair, init_probs, args, rng, stats)
            cand_x, cand_result, cand_source, evaluated, feasible_count = evaluate_candidates(
                current_x,
                subset,
                candidates,
                data,
                evaluator,
                linear,
                omega,
                args,
                stats,
            )
            if cand_result.feasible:
                elite_pool = update_elite(elite_pool, cand_x, cand_result, args.elite_size)
                surrogate.update(cand_result)
                if cand_result.obj > best_iter_obj:
                    best_iter_obj = cand_result.obj
                    best_iter_source = cand_source
                delta = cand_result.obj - current_result.obj
                temperature = max(args.temperature_min, T0 * (args.cooling_rate ** iteration))
                accept = delta > 1e-9
                if not accept and temperature > 1e-12:
                    accept = rng.random() < np.exp(delta / temperature)
                if accept:
                    current_x = cand_x.copy()
                    current_result = cand_result
                    stats.accepted_count += 1
                    accepted = True
                    if cand_source == "qaoa":
                        stats.qaoa_improvement_count += int(delta > 1e-9)
                    elif cand_source == "exact":
                        stats.exact_improvement_count += int(delta > 1e-9)
                    else:
                        stats.classical_improvement_count += int(delta > 1e-9)
                if cand_result.obj > best_result.obj + 1e-9:
                    best_x = cand_x.copy()
                    best_result = cand_result
                    improved_this_iter = True
                    stuck = 0
                    q_size = max(args.min_sub_size, q_size - args.q_shrink_step)
                    elapsed_fraction = elapsed / args.time_limit_seconds if args.time_limit_seconds > 0 else 1.0
                    if elapsed_fraction > args.late_break_fraction:
                        break

        if not improved_this_iter:
            stuck += 1
            if stuck >= args.no_improve_expand_threshold:
                q_size = min(args.q_max, q_size + args.q_growth_step)
            if stuck >= args.no_improve_restart_threshold and elite_pool:
                distances = [np.sum(np.abs(current_x - item[0])) for item in elite_pool]
                restart_idx = int(np.argmax(distances))
                current_x = elite_pool[restart_idx][0].copy()
                current_result = elite_pool[restart_idx][1]
                stats.restart_count += 1
                stuck = 0

        scale = max(1.0, abs(best_result.obj))
        lambda_B, violation_streak = update_lambda(lambda_B, data, current_x, violation_streak, scale)
        best_trace.append(best_result.obj)
        current_trace.append(current_result.obj)
        source_trace.append(best_iter_source)
        iterations_done = iteration
        print(
            f"[ITER {iteration:03d}] q={q_size:02d} neighborhoods={len(neighborhoods)} "
            f"best_iter={best_iter_obj:.6f} current={current_result.obj:.6f} "
            f"best={best_result.obj:.6f} accepted={accepted} improved={improved_this_iter} "
            f"source={best_iter_source} stuck={stuck} qaoa_calls={stats.qaoa_calls}"
        )

    final = evaluator.evaluate(best_x)
    if not final.feasible:
        raise RuntimeError("Best solution is infeasible; refusing to save.")
    y_out = final.y
    opt = float(data["optimal_value"]) if "optimal_value" in data else np.nan
    gap = (opt - final.obj) / abs(opt) if np.isfinite(opt) and abs(opt) > 0 else np.nan

    np.savez(
        args.output,
        x=best_x.astype(int),
        y=y_out.astype(float),
        objective=np.array(final.obj),
        feasible=np.array(final.feasible),
        optimal_value=np.array(opt),
        optimality_gap=np.array(gap),
        iterations_done=np.array(iterations_done),
        best_trace=np.asarray(best_trace, dtype=float),
        current_trace=np.asarray(current_trace, dtype=float),
        qaoa_calls=np.array(stats.qaoa_calls),
        qaoa_calls_small=np.array(stats.qaoa_calls_small),
        qaoa_calls_mid=np.array(stats.qaoa_calls_mid),
        qaoa_calls_large=np.array(stats.qaoa_calls_large),
        exact_calls=np.array(stats.exact_calls),
        sa_calls=np.array(stats.sa_calls),
        qaoa_time=np.array(stats.qaoa_time),
        exact_time=np.array(stats.exact_time),
        sa_time=np.array(stats.sa_time),
        lp_eval_count=np.array(stats.lp_evals),
        cache_hits=np.array(stats.cache_hits),
        evaluated_count=np.array(stats.evaluated_count),
        feasible_candidate_count=np.array(stats.feasible_candidate_count),
        repair_count=np.array(stats.repair_count),
        accepted_count=np.array(stats.accepted_count),
        restart_count=np.array(stats.restart_count),
        qaoa_improvement_count=np.array(stats.qaoa_improvement_count),
        exact_improvement_count=np.array(stats.exact_improvement_count),
        classical_improvement_count=np.array(stats.classical_improvement_count),
        qaoa_agreement_mean=np.array(stats.qaoa_agreement_mean()),
        max_qubits=np.array(stats.max_qubits()),
        seed=np.array(args.seed),
        elapsed_seconds=np.array(time.time() - start_time),
        max_binary_violation=np.array(final.max_binary_violation),
        max_mixed_violation=np.array(final.max_mixed_violation),
    )
    print(f"[DONE] saved={args.output}")
    print(f"[DONE] objective={final.obj:.12f}")
    print(f"[DONE] feasible={final.feasible}")
    print(f"[DONE] optimal_value={opt:.12f}")
    print(f"[DONE] optimality_gap={gap:.12f}")
    print(f"[DONE] qaoa_calls={stats.qaoa_calls}, max_qubits={stats.max_qubits()}")
    print(f"[DONE] lp_evals={stats.lp_evals}, cache_hits={stats.cache_hits}")
    return best_x, y_out, final.obj, final.feasible


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="solution_v4.npz")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--time-limit-seconds", type=float, default=0.0)
    parser.add_argument("--q-max", type=int, default=18)
    parser.add_argument("--qaoa-qubits", type=int, default=18)
    parser.add_argument("--initial-sub-size", type=int, default=12)
    parser.add_argument("--min-sub-size", type=int, default=8)
    parser.add_argument("--eta-ema", type=float, default=0.3)
    parser.add_argument("--eta-resc", type=float, default=0.5)
    parser.add_argument("--qaoa-opt-steps", type=int, default=20)
    parser.add_argument("--qaoa-opt-steps-large", type=int, default=0)
    parser.add_argument("--qaoa-multistart", type=int, default=2)
    parser.add_argument("--qaoa-depth-small", type=int, default=2)
    parser.add_argument("--qaoa-depth-medium", type=int, default=2)
    parser.add_argument("--qaoa-depth-large", type=int, default=1)
    parser.add_argument("--qaoa-rhobeg", type=float, default=0.35)
    parser.add_argument("--shots-small", type=int, default=1024)
    parser.add_argument("--shots-large", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-size", type=int, default=5000)
    parser.add_argument("--repair-flip-limit", type=int, default=0)
    parser.add_argument("--temperature-scale", type=float, default=0.05)
    parser.add_argument("--cooling-rate", type=float, default=0.95)
    parser.add_argument("--temperature-min", type=float, default=1e-12)
    parser.add_argument("--elite-size", type=int, default=20)
    parser.add_argument("--no-improve-restart-threshold", type=int, default=10)
    parser.add_argument("--no-improve-expand-threshold", type=int, default=5)
    parser.add_argument("--q-growth-step", type=int, default=3)
    parser.add_argument("--q-shrink-step", type=int, default=2)
    parser.add_argument("--late-break-fraction", type=float, default=0.0)
    parser.add_argument("--exact-init-limit", type=int, default=15)
    parser.add_argument("--init-random-small", type=int, default=15)
    parser.add_argument("--init-random-large", type=int, default=8)
    args = parser.parse_args()
    if args.q_max > 18:
        raise ValueError("v4 q_max must be <= 18.")
    if args.qaoa_qubits > 18:
        raise ValueError("v4 qaoa_qubits must be <= 18.")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.repair_flip_limit <= 0:
        data_preview = load_instance(parsed.input)
        parsed.repair_flip_limit = max(1, data_preview["n"] // 4)
    run_hybrid_v4(parsed)
