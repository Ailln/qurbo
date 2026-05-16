"""模拟退火求解器：经典备选方案，纯 CPU 计算。"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
from numpy import ndarray


@dataclass
class SAConfig:
    """模拟退火配置。"""
    num_reads: int = 100
    num_sweeps: int = 1000
    T_init: float = 10.0
    T_final: float = 0.01
    schedule: str = 'geometric'
    seed: Optional[int] = None


@dataclass
class SolverResult:
    """求解器结果。"""
    solutions: List[Tuple[ndarray, float, int]]  # [(bitstring, energy, count)]
    optimal_params: ndarray
    convergence_history: List[float]
    total_time: float


class SimulatedAnnealingSolver:
    """模拟退火求解器。

    适用场景:
        - subQUBO 规模 > max_qubits（QAOA 无法处理时）
        - QAOA 超时时的 fallback
        - 纯 CPU 环境（无 Qiskit 时的首选）
        - baseline 对比
    """

    def solve(self, qubo, config: SAConfig) -> SolverResult:
        """使用模拟退火求解 QUBO。

        单次退火流程:
            1. 随机初始化 z
            2. 对每步温度:
               a. 随机选择一个变量 i
               b. 计算翻转能量差 ΔE（O(q) 增量计算）
               c. 若 ΔE < 0 或 random() < exp(-ΔE/T): 接受翻转
            3. 降温
        """
        import time
        t0 = time.perf_counter()

        qubo_matrix = qubo.matrix
        q = qubo_matrix.shape[0]

        # 自适应参数
        if qubo_matrix.size == 0:
            max_coupling = 1e-12
        else:
            max_coupling = max(np.max(np.abs(qubo_matrix)), 1e-12)
        T_init = 3.0 * max_coupling
        T_final = 0.01 * max_coupling
        num_sweeps = max(500, 50 * q)

        if config.seed is not None:
            np.random.seed(config.seed)

        all_solutions = []

        for read in range(config.num_reads):
            # 随机初始化
            z = np.random.randint(0, 2, q).astype(float)
            current_energy = self._compute_energy(qubo_matrix, z)

            # 退火
            temperatures = self._temperature_schedule(T_init, T_final, num_sweeps, config.schedule)

            for T in temperatures:
                i = np.random.randint(q)
                delta_E = self._compute_delta_E(qubo_matrix, z, i)

                if delta_E < 0 or np.random.rand() < np.exp(-delta_E / max(T, 1e-12)):
                    z[i] = 1 - z[i]
                    current_energy += delta_E

            all_solutions.append((z.copy(), current_energy))

        # 统计并返回 top-k 唯一解
        unique_solutions = self._aggregate_solutions(all_solutions)

        total_time = time.perf_counter() - t0

        return SolverResult(
            solutions=unique_solutions,
            optimal_params=np.array([]),
            convergence_history=[],
            total_time=total_time,
        )

    def _compute_energy(self, qubo: ndarray, z: ndarray) -> float:
        """计算 QUBO 能量（O(q^2)）。仅用于初始化。"""
        energy = 0.0
        q = len(z)
        for i in range(q):
            energy += qubo[i, i] * z[i]
            for j in range(i + 1, q):
                energy += qubo[i, j] * z[i] * z[j]
        return energy

    def _compute_delta_E(self, qubo: ndarray, z: ndarray, i: int) -> float:
        """增量计算翻转变量 i 的能量差（O(q)）。

        ΔE_flip(i) = (1 - 2*z_i) * (qubo[i,i] + sum_{j≠i} qubo[min(i,j), max(i,j)] * z_j)
        """
        q = len(z)
        zi = z[i]
        sum_term = qubo[i, i]
        for j in range(q):
            if j == i:
                continue
            mi, ma = min(i, j), max(i, j)
            sum_term += qubo[mi, ma] * z[j]
        return (1 - 2 * zi) * sum_term

    def _temperature_schedule(self, T_init: float, T_final: float,
                              n_steps: int, schedule: str) -> ndarray:
        """生成温度调度序列。"""
        if schedule == 'geometric':
            ratio = (T_final / T_init) ** (1.0 / max(n_steps - 1, 1))
            return T_init * (ratio ** np.arange(n_steps))
        elif schedule == 'linear':
            return np.linspace(T_init, T_final, n_steps)
        else:
            return T_init * ((T_final / T_init) ** (np.arange(n_steps) / max(n_steps - 1, 1)))

    def _aggregate_solutions(self, solutions: List[Tuple[ndarray, float]],
                             top_k: int = 20) -> List[Tuple[ndarray, float, int]]:
        """聚合多次退火结果，返回唯一 top-k 解。"""
        # 按能量排序
        solutions.sort(key=lambda x: x[1])

        unique = []
        seen = set()
        for z, energy in solutions:
            key = tuple(z.astype(int))
            if key not in seen:
                seen.add(key)
                unique.append((z, energy, 1))
            if len(unique) >= top_k:
                break

        return unique
