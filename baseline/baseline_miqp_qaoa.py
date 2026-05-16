import argparse
import numpy as np
from scipy.optimize import linprog

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def load_instance(path: str):
    raw = np.load(path)
    data = {k: raw[k] for k in raw.files}

    # 标量转 int
    for key in ["n", "p", "m1", "m2"]:
        if key in data:
            data[key] = int(np.array(data[key]).item())

    return data


def objective_value(x, y, data):
    Q = data["Q"]
    c = data["c"]
    h = data["h"]

    return float(x @ Q @ x + c @ x + h @ y)


def binary_constraint_violation(x, data):
    if "B" not in data or "b_prime" not in data:
        return 0.0

    B = data["B"]
    b_prime = data["b_prime"]

    if B.size == 0:
        return 0.0

    v = B @ x - b_prime
    return float(np.sum(np.maximum(v, 0.0)))


def solve_continuous_subproblem(x, data):
    """
    固定二元变量 x 后，求连续变量 y。

    原问题大致是：
        max x^T Q x + c^T x + h^T y
        s.t. A x + G y <= b
             B x <= b_prime
             y >= 0

    固定 x 后，连续部分变成 LP：
        max h^T y
        s.t. G y <= b - A x
             y >= 0
    """
    A = data["A"]
    G = data["G"]
    b = data["b"]
    h = data["h"]
    p = data["p"]

    # 先检查纯二元约束
    if binary_constraint_violation(x, data) > 1e-8:
        return False, None, -np.inf

    rhs = b - A @ x

    res = linprog(
        c=-h,                    # linprog 是最小化，所以 max h@y 写成 min -h@y
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
    给 QUBO 加一项：
        penalty * (const + sum_i linear_i z_i)^2

    其中 z_i 是 0/1 变量。
    """
    k = len(linear)

    for i in range(k):
        l[i] += penalty * (2.0 * const * linear[i] + linear[i] ** 2)

    for i in range(k):
        for j in range(i + 1, k):
            pair[(i, j)] = pair.get((i, j), 0.0) + penalty * 2.0 * linear[i] * linear[j]


def build_subqubo(x, subset, data, penalty=10.0):
    """
    构造一个局部 subQUBO。

    我们最小化：
        - 原目标函数的二元部分
        + 约束罚项

    注意：这是 baseline，不是严格最优建模。
    """
    Q = data["Q"]
    c = data["c"]
    A = data["A"]
    b = data["b"]
    B = data["B"]
    b_prime = data["b_prime"]

    subset = np.array(subset, dtype=int)
    rest = np.array([i for i in range(len(x)) if i not in set(subset)], dtype=int)

    k = len(subset)
    l = np.zeros(k, dtype=float)
    pair = {}

    Qss = Q[np.ix_(subset, subset)]

    if len(rest) > 0:
        Qsr = Q[np.ix_(subset, rest)]
        xr = x[rest]
        fixed_linear = 2.0 * Qsr @ xr
    else:
        fixed_linear = np.zeros(k)

    # 原问题是最大化，这里转成最小化，所以取负号
    # x^T Q x 的子块：
    for i in range(k):
        l[i] += -Qss[i, i]

    for i in range(k):
        for j in range(i + 1, k):
            pair[(i, j)] = pair.get((i, j), 0.0) - 2.0 * Qss[i, j]

    # c^T x 和固定变量带来的线性项
    l += -(c[subset] + fixed_linear)

    # 纯二元约束：B x <= b_prime
    # baseline 简化为平方罚项，实际可以引入 slack 做得更严谨
    if B.size > 0:
        for row, rhs in zip(B, b_prime):
            linear = row[subset]
            const = float(row @ x - linear @ x[subset] - rhs)
            add_squared_linear_penalty(linear, const, penalty, l, pair)

    # 混合约束中的 A x 部分也加一个轻量罚项
    # 连续变量 y 后面由 LP 修正，这里只是防止 x 太离谱
    if A.size > 0:
        for row, rhs in zip(A, b):
            linear = row[subset]
            const = float(row @ x - linear @ x[subset] - rhs)
            add_squared_linear_penalty(linear, const, penalty * 0.2, l, pair)

    return l, pair


def qubo_energy(bits, l, pair):
    e = float(l @ bits)
    for (i, j), v in pair.items():
        e += float(v * bits[i] * bits[j])
    return e


def apply_cost_layer(qc, gamma, l, pair):
    """
    对 QUBO 成本函数做一个简单相位编码：
        E(z) = sum_i l_i z_i + sum_ij q_ij z_i z_j

    z_i = (1 - Z_i) / 2
    """
    k = len(l)

    # 线性项
    for i in range(k):
        qc.rz(-gamma * l[i], i)

    # 二次项
    for (i, j), q in pair.items():
        # z_i z_j 会产生 ZiZj 项和两个单 Z 项
        qc.rz(-gamma * q / 2.0, i)
        qc.rz(-gamma * q / 2.0, j)
        qc.rzz(gamma * q / 2.0, i, j)


def apply_mixer_layer(qc, beta, k):
    for i in range(k):
        qc.rx(2.0 * beta, i)


def qaoa_like_solve_subqubo(l, pair, shots=512, layers=1, device="CPU"):
    """
    一个很小的 QAOA-like baseline。

    不引入 qiskit_algorithms，避免额外依赖。
    用粗粒度 grid search 找一组 gamma/beta。
    """
    k = len(l)

    if k == 0:
        return np.array([], dtype=int)

    gammas = [0.2, 0.8, 1.4, 2.0]
    betas = [0.2, 0.6, 1.0]

    best_bits = None
    best_energy = np.inf

    sim = AerSimulator(method="statevector", device=device)

    for gamma in gammas:
        for beta in betas:
            qc = QuantumCircuit(k, k)

            for i in range(k):
                qc.h(i)

            for _ in range(layers):
                apply_cost_layer(qc, gamma, l, pair)
                apply_mixer_layer(qc, beta, k)

            qc.measure(range(k), range(k))

            result = sim.run(qc, shots=shots).result()
            counts = result.get_counts()

            # 取出现频率最高的若干候选，再用 QUBO energy 精排
            candidates = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]

            for bitstr, _count in candidates:
                # Qiskit 输出 bitstr 的顺序和 qubit index 相反
                bits = np.array([int(v) for v in bitstr[::-1]], dtype=int)
                e = qubo_energy(bits, l, pair)

                if e < best_energy:
                    best_energy = e
                    best_bits = bits.copy()

    return best_bits


def initialize_solution(data, random_trials=100, seed=42):
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
        # 如果完全找不到可行解，先用全 0 兜底
        best_x = np.zeros(n, dtype=int)
        feasible, y, obj = solve_continuous_subproblem(best_x, data)
        if y is None:
            y = np.zeros(data["p"])
            obj = -np.inf

        best_y = y
        best_obj = obj
        best_feasible = feasible

    return best_x, best_y, best_obj, best_feasible


def choose_subset(x, data, sub_size, rng):
    """
    baseline 子问题选择策略：
    优先选 Q 中耦合强的变量，再混一点随机性。
    """
    Q = data["Q"]
    n = data["n"]

    strength = np.sum(np.abs(Q), axis=1)
    top = np.argsort(-strength)[: max(sub_size * 2, sub_size)]

    if len(top) <= sub_size:
        return top

    return rng.choice(top, size=sub_size, replace=False)


def run_baseline(
    path,
    output_path="solution_baseline.npz",
    iterations=20,
    sub_size=12,
    shots=512,
    layers=1,
    penalty=10.0,
    device="CPU",
    seed=42,
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
    print(f"[INFO] sub_size={sub_size}, iterations={iterations}, shots={shots}, device={device}")

    x, y, best_obj, feasible = initialize_solution(data, seed=seed)

    print(f"[INIT] feasible={feasible}, objective={best_obj:.6f}")

    for it in range(1, iterations + 1):
        subset = choose_subset(x, data, sub_size, rng)

        l, pair = build_subqubo(x, subset, data, penalty=penalty)
        sub_bits = qaoa_like_solve_subqubo(
            l,
            pair,
            shots=shots,
            layers=layers,
            device=device,
        )

        new_x = x.copy()
        new_x[subset] = sub_bits

        new_feasible, new_y, new_obj = solve_continuous_subproblem(new_x, data)

        accepted = False

        if new_feasible and new_obj > best_obj:
            x = new_x
            y = new_y
            best_obj = new_obj
            feasible = True
            accepted = True

        print(
            f"[ITER {it:03d}] "
            f"candidate_feasible={new_feasible} "
            f"candidate_obj={new_obj:.6f} "
            f"best_obj={best_obj:.6f} "
            f"accepted={accepted}"
        )

    np.savez(
        output_path,
        x=x.astype(int),
        y=y.astype(float),
        objective=np.array(best_obj),
        feasible=np.array(feasible),
    )

    print(f"[DONE] saved to {output_path}")
    print(f"[DONE] best objective = {best_obj:.6f}")
    print(f"[DONE] feasible = {feasible}")

    return x, y, best_obj, feasible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to miqp_xxx.npz")
    parser.add_argument("--output", default="solution_baseline.npz")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--sub-size", type=int, default=12)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--penalty", type=float, default=10.0)
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_baseline(
        path=args.input,
        output_path=args.output,
        iterations=args.iterations,
        sub_size=args.sub_size,
        shots=args.shots,
        layers=args.layers,
        penalty=args.penalty,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
