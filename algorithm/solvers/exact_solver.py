"""精确 QUBO 求解器：对于小规模子问题（q <= 20），用向量化穷举找到全局最优。

在模拟器环境下，对于 q <= 18 的问题，穷举法比 QAOA 快 100-1000 倍，
且解质量更高（精确最优 vs 近似最优）。
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from numpy import ndarray


@dataclass
class ExactSolverResult:
    """精确求解器结果。"""
    solutions: List[Tuple[ndarray, float, int]]  # (bits, energy, count)
    total_time: float


class ExactQUBOSolver:
    """向量化穷举 QUBO 求解器。

    使用 numpy 向量化操作枚举所有 2^q 个状态，计算能量并排序。
    对于 q <= 18，单次求解 < 200ms；q <= 16，单次求解 < 25ms。
    """

    def __init__(self, max_qubits: int = 18):
        self.max_qubits = max_qubits

    def solve(self, qubo, top_k: int = 20) -> ExactSolverResult:
        """精确求解 QUBO，返回 top-k 最优解。"""
        import time
        t0 = time.perf_counter()

        Q = qubo.matrix
        q = Q.shape[0]
        if q > self.max_qubits:
            raise ValueError(f"QUBO size {q} exceeds exact solver limit {self.max_qubits}")

        solutions = self._exact_solve(Q, q, top_k)
        total_time = time.perf_counter() - t0
        return ExactSolverResult(solutions=solutions, total_time=total_time)

    def _exact_solve(self, Q: ndarray, q: int, top_k: int) -> List[Tuple[ndarray, float, int]]:
        """向量化穷举核心。"""
        N = 2 ** q

        # 生成所有位串: (N, q)
        bits = ((np.arange(N)[:, None] >> np.arange(q)[None, :]) & 1).astype(np.float64)

        # 向量化能量计算
        # E = sum_i Q_ii z_i + sum_{i<j} Q_ij z_i z_j
        # 其中 z_i z_j 对 i=j 也成立（因为 z_i^2 = z_i）
        # 所以 E = sum_i Q_ii z_i + sum_{i<j} Q_ij z_i z_j
        # 等价于: 0.5 * z^T (Q + Q^T) z 的适当形式
        # 但更简单的方式是：
        # diag_part = sum_i Q_ii z_i
        # offdiag_part = sum_{i<j} Q_ij z_i z_j

        diag = np.diag(Q)
        offdiag = Q.copy()
        np.fill_diagonal(offdiag, 0)

        # bits @ diag: (N,) 每个状态的对角项之和
        diag_part = bits @ diag  # (N,)

        # bits @ offdiag: (N, q) 每行是 z^T * offdiag
        # sum((bits @ offdiag) * bits, axis=1): sum_{i,j} z_i * offdiag_{ij} * z_j
        # 但这样计算了所有 i,j 对，包括 i=j（已设为0）和 i>j（offdiag 上三角，下三角也为0）
        # 所以实际上 sum_{i<j} Q_ij z_i z_j * 2 ? 不对，offdiag 只保留了上三角
        # 等等，我们的 Q 是上三角矩阵！所以只需要计算上三角部分
        # bits @ offdiag 给出的是 sum_j Q_{ij} z_j（对 i 求和）
        # 然后乘 bits[:, i] 再求和就是 sum_i z_i * sum_j Q_{ij} z_j = sum_{i,j} Q_{ij} z_i z_j
        # 由于 Q 上三角，Q_{ij} = 0 for i > j，所以只计算了 i <= j 的部分
        # 但 Q_{ii} 在 offdiag 中被设为 0，所以只计算了 i < j 的部分
        # 而我们需要的是 sum_{i<j} Q_{ij} z_i z_j
        # 所以 offdiag_part = sum((bits @ offdiag) * bits, axis=1) 就是对的！

        offdiag_part = np.sum((bits @ offdiag) * bits, axis=1)  # (N,)

        energy = diag_part + offdiag_part

        # 取 top-k 最优（最小能量）
        best_indices = np.argsort(energy)[:top_k]
        solutions = []
        for idx in best_indices:
            solutions.append((bits[idx].copy(), float(energy[idx]), 1))

        return solutions
