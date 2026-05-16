"""
量子求解器模块 - QAOA求解subQUBO
支持Qiskit和PennyLane两个后端
"""

import numpy as np
from typing import Tuple, Optional
import config


# ============================================================
# Qiskit后端（推荐用于黑客松）
# ============================================================
class QiskitQAOASolver:
    """
    基于Qiskit的QAOA求解器
    优势：
    - 文档完善，API稳定
    - QAOA有内置的优化循环
    - 与Aer模拟器集成好
    - 社区资源丰富，调试方便
    """

    def __init__(self, reps: int = 2, shots: int = 1024, maxiter: int = 100):
        self.reps = reps
        self.shots = shots
        self.maxiter = maxiter
        self._backend = None
        self._history = []

    def _get_backend(self):
        """延迟初始化backend，节省导入时间"""
        if self._backend is None:
            from qiskit_aer import AerSimulator
            self._backend = AerSimulator()
        return self._backend

    def _build_qubo_operator(self, Q: np.ndarray, c: np.ndarray):
        """
        构建Ising哈密顿量
        QUBO: x^T Q x + c^T x
        映射到Ising: z_i = 1 - 2x_i -> x_i = (1 - z_i)/2
        """
        n = Q.shape[0]

        # 常数项
        constant = np.sum(Q) / 4 + np.sum(c) / 2

        # 线性系数（Z项）
        linear = np.zeros(n)
        for i in range(n):
            linear[i] = -0.5 * c[i]
            for j in range(n):
                linear[i] -= 0.5 * (Q[i, j] + Q[j, i]) / 2

        # 二次系数（ZZ项）
        quadratic = Q / 4

        # 构建Pauli算子
        from qiskit.quantum_info import SparsePauliOp

        pauli_list = []
        coeffs = []

        # 常数项 (I^\otimes n)
        pauli_list.append("I" * n)
        coeffs.append(constant)

        # 线性项 (Z_i)
        for i in range(n):
            z_str = ["I"] * n
            z_str[i] = "Z"
            pauli_list.append("".join(z_str))
            coeffs.append(linear[i])

        # 二次项 (Z_i Z_j, i < j)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(quadratic[i, j]) > 1e-10:
                    z_str = ["I"] * n
                    z_str[i] = "Z"
                    z_str[j] = "Z"
                    pauli_list.append("".join(z_str))
                    coeffs.append(quadratic[i, j] + quadratic[j, i])

        hamiltonian = SparsePauliOp(pauli_list, coeffs)
        return hamiltonian

    def solve(self, Q: np.ndarray, c: np.ndarray,
              initial_point: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        求解QUBO，返回最优解和目标值

        Args:
            Q: QUBO二次矩阵 (k, k)
            c: QUBO一次项 (k,)
            initial_point: 热启动初始参数

        Returns:
            x_best: 最优二元解 (k,)
            obj_best: 最优目标值
        """
        from qiskit_algorithms import QAOA
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit.primitives import BackendSampler
        from qiskit.quantum_info import SparsePauliOp

        n = Q.shape[0]
        hamiltonian = self._build_qubo_operator(Q, c)

        # 构建QAOA
        sampler = BackendSampler(backend=self._get_backend(), options={"shots": self.shots})
        optimizer = COBYLA(maxiter=self.maxiter, rhobeg=0.1)

        qaoa = QAOA(
            sampler=sampler,
            optimizer=optimizer,
            reps=self.reps,
            initial_point=initial_point
        )

        # 运行
        result = qaoa.compute_minimum_eigenvalue(hamiltonian)

        # 从最优bitstring解码
        best_bitstring = result.best_measurement["bitstring"]
        # bitstring格式: "0101..." (Qiskit默认从右到左)，需要反转
        x_best = np.array([int(best_bitstring[i]) for i in range(n)], dtype=float)

        # 计算目标值
        obj_best = x_best @ Q @ x_best + c @ x_best

        self._history.append({
            "energies": result.eigenvalue,
            "cost_function_evals": result.cost_function_evals,
            "best": obj_best
        })

        return x_best, obj_best


# ============================================================
# PennyLane后端（备选）
# ============================================================
class PennyLaneQAOASolver:
    """
    基于PennyLane的QAOA求解器
    优势：
    - 自动微分，梯度计算更快
    - 支持更多优化器（Adam, AdamW等）
    - 代码更简洁

    劣势：
    - QAOA需要自己构建整个电路
    - 测量后处理较复杂
    """

    def __init__(self, reps: int = 2, shots: int = 1024, maxiter: int = 100):
        self.reps = reps
        self.shots = shots
        self.maxiter = maxiter
        self._dev = None

    def _get_device(self, n_qubits: int):
        if self._dev is None or self._dev.num_wires != n_qubits:
            import pennylane as qml
            self._dev = qml.device("default.qubit", wires=n_qubits, shots=self.shots)
        return self._dev

    def solve(self, Q: np.ndarray, c: np.ndarray,
              initial_point: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        import pennylane as qml
        from pennylane import numpy as pnp

        n = Q.shape[0]
        dev = self._get_device(n)

        # 构建cost Hamiltonian
        coeffs, ops = self._build_hamiltonian(Q, c)
        cost_h = qml.Hamiltonian(coeffs, ops)

        # 构建mixer Hamiltonian
        mixer_ops = []
        for i in range(n):
            mixer_ops.append(qml.PauliX(i))
        mixer_h = qml.Hamiltonian([1.0] * n, mixer_ops)

        # QAOA电路
        @qml.qnode(dev)
        def qaoa_circuit(params):
            # 初始叠加态
            for i in range(n):
                qml.Hadamard(i)

            # QAOA层
            for p in range(self.reps):
                gamma = params[2 * p]
                beta = params[2 * p + 1]

                # Cost层 (ZZ相互作用)
                for i in range(n):
                    for j in range(i + 1, n):
                        if abs(Q[i, j]) > 1e-10:
                            qml.IsingZZ(2 * gamma * (Q[i, j] + Q[j, i]) / 4, wires=[i, j])

                # 线性项 (RZ旋转)
                for i in range(n):
                    angle = gamma * (-0.5 * c[i] - 0.5 * np.sum(Q[i, :] + Q[:, i]))
                    qml.RZ(angle, wires=i)

                # Mixer层
                for i in range(n):
                    qml.RX(2 * beta, wires=i)

            return qml.expval(cost_h)

        # 测量电路（用于获取bitstring）
        @qml.qnode(dev)
        def measure_circuit(params):
            for i in range(n):
                qml.Hadamard(i)
            for p in range(self.reps):
                gamma = params[2 * p]
                beta = params[2 * p + 1]
                for i in range(n):
                    for j in range(i + 1, n):
                        if abs(Q[i, j]) > 1e-10:
                            qml.IsingZZ(2 * gamma * (Q[i, j] + Q[j, i]) / 4, wires=[i, j])
                for i in range(n):
                    angle = gamma * (-0.5 * c[i] - 0.5 * np.sum(Q[i, :] + Q[:, i]))
                    qml.RZ(angle, wires=i)
                for i in range(n):
                    qml.RX(2 * beta, wires=i)
            return qml.sample(wires=range(n))

        # 优化
        if initial_point is not None:
            params = pnp.array(initial_point, requires_grad=True)
        else:
            # 随机初始参数
            params = pnp.random.uniform(0, np.pi, 2 * self.reps, requires_grad=True)

        opt = qml.AdamOptimizer(stepsize=0.05)

        for _ in range(self.maxiter):
            params, cost = opt.step_and_cost(lambda p: qaoa_circuit(p), params)

        # 获取最优解
        samples = measure_circuit(params)
        # 统计最频繁的结果
        from collections import Counter
        bitstrings = [tuple(s.tolist()) for s in samples]
        most_common = Counter(bitstrings).most_common(1)[0][0]
        x_best = np.array(most_common, dtype=float)

        obj_best = x_best @ Q @ x_best + c @ x_best
        return x_best, obj_best

    def _build_hamiltonian(self, Q: np.ndarray, c: np.ndarray):
        """构建PennyLane哈密顿量"""
        import pennylane as qml
        n = Q.shape[0]
        coeffs = []
        ops = []

        # 线性项
        for i in range(n):
            linear_coeff = -0.5 * c[i]
            for j in range(n):
                linear_coeff -= 0.5 * (Q[i, j] + Q[j, i]) / 2
            if abs(linear_coeff) > 1e-10:
                coeffs.append(linear_coeff)
                ops.append(qml.PauliZ(i))

        # 二次项
        for i in range(n):
            for j in range(i + 1, n):
                q_coeff = (Q[i, j] + Q[j, i]) / 4
                if abs(q_coeff) > 1e-10:
                    coeffs.append(q_coeff)
                    ops.append(qml.PauliZ(i) @ qml.PauliZ(j))

        return coeffs, ops


# ============================================================
# 统一的求解器工厂
# ============================================================
def create_quantum_solver(backend: str = None, reps: int = None, shots: int = None, maxiter: int = None):
    """
    工厂函数：创建量子求解器

    Usage:
        solver = create_quantum_solver()  # 使用config中的默认配置
        x, obj = solver.solve(Q_sub, c_sub)
    """
    backend = backend or config.QUANTUM_BACKEND
    reps = reps or config.QAOA_REPS
    shots = shots or config.SHOTS
    maxiter = maxiter or config.QAOA_MAXITER

    if backend == "qiskit":
        return QiskitQAOASolver(reps=reps, shots=shots, maxiter=maxiter)
    elif backend == "pennylane":
        return PennyLaneQAOASolver(reps=reps, shots=shots, maxiter=maxiter)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ============================================================
# 备用：经典穷举求解器（用于n<=15验证）
# ============================================================
def solve_qubo_exact(Q: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    经典穷举求解（仅用于验证小规模问题）
    n<=20时使用，确保量子求解结果正确
    """
    n = Q.shape[0]
    if n > 20:
        raise ValueError(f"穷举求解仅限n<=20，当前n={n}")

    best_obj = float("-inf")
    best_x = None

    for i in range(2 ** n):
        x = np.array([(i >> j) & 1 for j in range(n)], dtype=float)
        obj = x @ Q @ x + c @ x
        if obj > best_obj:
            best_obj = obj
            best_x = x

    return best_x, best_obj
