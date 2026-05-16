"""QAOA 求解器：基于 Qiskit Aer 的轻量 QAOA-like 量子电路优化。

实现特点:
    - 使用 AerSimulator (statevector 方法) 进行 CPU/GPU 模拟
    - 手动构建 CostLayer (RZ/RZZ) + MixerLayer (RX)
    - scipy.optimize.minimize (COBYLA) 做参数优化
    - 支持 multi-start 避免局部极小
    - 支持 warm-start (Ry 旋转替代 H 门)
    - Aer 不可用或失败时自动 fallback 到穷举/SA
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from numpy import ndarray

# Qiskit 导入
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    AER_AVAILABLE = True
except Exception:
    AER_AVAILABLE = False


@dataclass
class QAOAConfig:
    """QAOA 配置。"""
    p_layers: int = 1
    shots: int = 1024
    optimizer: str = 'COBYLA'
    max_opt_steps: int = 50
    warm_start_probs: Optional[ndarray] = None
    elite_correlations: Optional[List] = None
    use_structure_init: bool = False
    qaoa_multi_start: int = 2
    final_shots: int = 1024


@dataclass
class SolverResult:
    """求解器结果。"""
    solutions: List[Tuple[ndarray, float, int]]
    optimal_params: ndarray
    convergence_history: List[float]
    total_time: float


class QAOASolver:
    """QAOA 求解器（基于 AerSimulator 的量子电路模拟）。"""

    def __init__(self, max_qubits: int = 20, device: str = 'CPU'):
        self.max_qubits = max_qubits
        self.device = device.upper()
        self.simulator = None
        if AER_AVAILABLE:
            try:
                self.simulator = AerSimulator(method='statevector', device=self.device)
            except Exception:
                # GPU 不支持时回退到 CPU
                if self.device != 'CPU':
                    try:
                        self.simulator = AerSimulator(method='statevector', device='CPU')
                        self.device = 'CPU'
                    except Exception:
                        self.simulator = None
                else:
                    self.simulator = None

    def solve(self, qubo, config: QAOAConfig) -> SolverResult:
        """使用 QAOA-like 电路求解 QUBO 问题。"""
        import time
        t0 = time.perf_counter()

        q = qubo.matrix.shape[0]
        if q > self.max_qubits:
            raise ValueError(f"QUBO size {q} exceeds max_qubits {self.max_qubits}")

        # 将 QUBO 矩阵转为 (l, pair) 格式
        l_vec, pair_dict = self._qubo_to_lp(qubo.matrix)

        # 如果 Aer 不可用，fallback 到穷举
        if self.simulator is None or not AER_AVAILABLE:
            solutions = self._brute_force_fallback(qubo.matrix, q)
            return SolverResult(
                solutions=solutions,
                optimal_params=np.array([]),
                convergence_history=[],
                total_time=time.perf_counter() - t0,
            )

        # 参数优化
        best_params, best_energy, history = self._optimize_params(
            l_vec, pair_dict, config, q
        )

        # 最终采样：用最优参数跑更多 shots
        solutions = self._final_sample(
            l_vec, pair_dict, best_params, config.final_shots, q,
            config.warm_start_probs
        )

        total_time = time.perf_counter() - t0

        return SolverResult(
            solutions=solutions,
            optimal_params=np.array(best_params),
            convergence_history=history,
            total_time=total_time,
        )

    def _qubo_to_lp(self, qubo_matrix: ndarray) -> Tuple[ndarray, dict]:
        """将上三角 QUBO 矩阵转为 (l, pair) 格式用于电路构建。"""
        q = qubo_matrix.shape[0]
        l_vec = np.zeros(q)
        pair_dict = {}
        for i in range(q):
            l_vec[i] = qubo_matrix[i, i]
            for j in range(i + 1, q):
                if abs(qubo_matrix[i, j]) > 1e-12:
                    pair_dict[(i, j)] = qubo_matrix[i, j]
        return l_vec, pair_dict

    def _build_circuit(self, params: ndarray, l_vec: ndarray,
                       pair_dict: dict, q: int,
                       warm_start_probs: Optional[ndarray] = None) -> QuantumCircuit:
        """构建 QAOA-like 电路。"""
        p = len(params) // 2
        gammas = params[:p]
        betas = params[p:]

        qc = QuantumCircuit(q, q)

        # 初始层
        if warm_start_probs is not None and len(warm_start_probs) == q:
            for i in range(q):
                theta = 2.0 * np.arcsin(np.sqrt(np.clip(warm_start_probs[i], 0.0, 1.0)))
                qc.ry(theta, i)
        else:
            for i in range(q):
                qc.h(i)

        # QAOA 层
        for layer in range(p):
            # Cost layer: E(z) = sum_i l_i z_i + sum_{i<j} q_{ij} z_i z_j
            # z_i = (1 - Z_i)/2, 所以 Z_i 的系数是 -l_i/2, Z_iZ_j 是 q_{ij}/4
            # 为简化，直接用 RZ/RZZ 编码：
            # 线性项: exp(-i * gamma * l_i * z_i) → RZ(-gamma * l_i, i)
            # 二次项: z_i z_j 展开后有 Z_iZ_j/4 + Z_i/4 + Z_j/4 项
            # baseline 简化做法: RZ(-gamma*q/2, i), RZ(-gamma*q/2, j), RZZ(gamma*q/2, i, j)
            for i in range(q):
                if abs(l_vec[i]) > 1e-12:
                    qc.rz(-gammas[layer] * l_vec[i], i)
            for (i, j), qval in pair_dict.items():
                qc.rz(-gammas[layer] * qval / 2.0, i)
                qc.rz(-gammas[layer] * qval / 2.0, j)
                qc.rzz(gammas[layer] * qval / 2.0, i, j)

            # Mixer layer
            for i in range(q):
                qc.rx(2.0 * betas[layer], i)

        qc.measure(range(q), range(q))
        return qc

    def _qubo_energy(self, bits: ndarray, l_vec: ndarray, pair_dict: dict) -> float:
        """计算 QUBO 能量。"""
        e = float(l_vec @ bits)
        for (i, j), qval in pair_dict.items():
            e += float(qval * bits[i] * bits[j])
        return e

    def _sample_best_energy(self, params: ndarray, l_vec: ndarray,
                            pair_dict: dict, q: int, shots: int,
                            warm_start_probs: Optional[ndarray] = None) -> float:
        """运行电路并返回采样中最低的 QUBO 能量。"""
        qc = self._build_circuit(params, l_vec, pair_dict, q, warm_start_probs)
        try:
            result = self.simulator.run(qc, shots=shots).result()
            counts = result.get_counts()
        except Exception:
            return np.inf

        best_e = np.inf
        for bitstr, _count in counts.items():
            bits = np.array([int(v) for v in bitstr[::-1]], dtype=float)
            e = self._qubo_energy(bits, l_vec, pair_dict)
            if e < best_e:
                best_e = e
        return best_e

    def _optimize_params(self, l_vec: ndarray, pair_dict: dict,
                         config: QAOAConfig, q: int) -> Tuple[ndarray, float, List[float]]:
        """使用 COBYLA 优化 QAOA 参数，支持 multi-start。"""
        from scipy.optimize import minimize

        p = config.p_layers
        shots = config.shots
        warm_start = config.warm_start_probs

        all_history = []
        global_best_params = None
        global_best_energy = np.inf

        for start in range(config.qaoa_multi_start):
            # 随机初始点
            gamma0 = np.random.uniform(0.0, np.pi, size=p)
            beta0 = np.random.uniform(0.0, np.pi, size=p)
            x0 = np.concatenate([gamma0, beta0])

            history = []

            def objective(x):
                energy = self._sample_best_energy(x, l_vec, pair_dict, q, shots, warm_start)
                history.append(float(energy))
                return float(energy)

            try:
                res = minimize(
                    objective,
                    x0,
                    method='COBYLA',
                    options={'maxiter': config.max_opt_steps, 'rhobeg': 0.1}
                )
            except Exception:
                continue

            if res.fun < global_best_energy:
                global_best_energy = res.fun
                global_best_params = res.x.copy()
                all_history = history.copy()

        if global_best_params is None:
            # 优化全部失败，回退到默认参数
            global_best_params = np.concatenate([
                np.full(p, 0.5),
                np.full(p, 0.3)
            ])
            global_best_energy = self._sample_best_energy(
                global_best_params, l_vec, pair_dict, q, shots, warm_start
            )

        return global_best_params, global_best_energy, all_history

    def _final_sample(self, l_vec: ndarray, pair_dict: dict,
                      params: ndarray, shots: int, q: int,
                      warm_start_probs: Optional[ndarray] = None,
                      top_k: int = 20) -> List[Tuple[ndarray, float, int]]:
        """用最优参数进行最终采样，返回 top-k 解。"""
        qc = self._build_circuit(params, l_vec, pair_dict, q, warm_start_probs)
        try:
            result = self.simulator.run(qc, shots=shots).result()
            counts = result.get_counts()
        except Exception:
            return []

        candidates = []
        for bitstr, count in counts.items():
            bits = np.array([int(v) for v in bitstr[::-1]], dtype=float)
            e = self._qubo_energy(bits, l_vec, pair_dict)
            candidates.append((bits, e, count))

        candidates.sort(key=lambda x: x[1])
        return [(bits.astype(float), float(e), int(c)) for bits, e, c in candidates[:top_k]]

    def _brute_force_fallback(self, qubo_matrix: ndarray, q: int,
                              top_k: int = 20) -> List[Tuple[ndarray, float, int]]:
        """Aer 不可用时的穷举 fallback。"""
        N = 2 ** q
        bits = ((np.arange(N)[:, None] >> np.arange(q)) & 1).astype(float)
        energy = np.zeros(N)
        for i in range(q):
            energy += qubo_matrix[i, i] * bits[:, i]
            for j in range(i + 1, q):
                energy += qubo_matrix[i, j] * bits[:, i] * bits[:, j]
        best_indices = np.argsort(energy)[:top_k]
        return [(bits[idx].copy(), float(energy[idx]), 1) for idx in best_indices]

    @staticmethod
    def qubo_to_ising(qubo_matrix: ndarray) -> Tuple[ndarray, ndarray, float]:
        """QUBO → Ising 转换（保留接口兼容性）。"""
        q = qubo_matrix.shape[0]
        a = np.diag(qubo_matrix)
        b = qubo_matrix.copy()
        np.fill_diagonal(b, 0)
        h = np.zeros(q)
        J = np.zeros((q, q))
        for i in range(q):
            row_sum = np.sum(b[i, :])
            h[i] = -0.5 * (a[i] + 0.5 * row_sum)
        for i in range(q):
            for j in range(i + 1, q):
                J[i, j] = b[i, j] / 4.0
        const = 0.5 * np.sum(a) + 0.25 * np.sum(b)
        return h, J, const
