"""MIQP 问题实例定义与数据加载。"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional
import numpy as np
from numpy import ndarray


@dataclass
class DiagReport:
    """实例诊断报告。"""
    Q_density: float
    Q_eigval_range: Tuple[float, float]
    Q_condition_number: float
    G_rank: int
    G_is_full_rank: bool
    constraint_slack_stats: Dict
    estimated_feasible_density: float
    recommended_sub_qubo_size: int


class MIQPInstance:
    """MIQP 问题实例。

    属性:
        n: 二元变量数
        p: 连续变量数
        m1: 混合约束数
        m2: 纯二元约束数
        Q: 对称二次矩阵 (n, n)
        c: 二元线性系数 (n,)
        h: 连续线性系数 (p,)
        A: 混合约束二元系数 (m1, n)
        G: 混合约束连续系数 (m1, p)
        b: 混合约束 RHS (m1,)
        B: 纯二元约束系数 (m2, n)
        b_prime: 纯二元约束 RHS (m2,)
    """

    def __init__(self):
        self.n: int = 0
        self.p: int = 0
        self.m1: int = 0
        self.m2: int = 0
        self.Q: ndarray = np.array([])
        self.c: ndarray = np.array([])
        self.h: ndarray = np.array([])
        self.A: ndarray = np.array([])
        self.G: ndarray = np.array([])
        self.b: ndarray = np.array([])
        self.B: ndarray = np.array([])
        self.b_prime: ndarray = np.array([])

    def load(self, filepath: str) -> None:
        """从 .npz 文件加载实例数据。"""
        data = np.load(filepath, allow_pickle=True)
        self.n = int(data['n'])
        self.p = int(data['p'])
        self.m1 = int(data['m1'])
        self.m2 = int(data['m2'])
        self.Q = data['Q']
        self.c = data['c']
        self.h = data['h']
        self.A = data['A']
        self.G = data['G']
        self.b = data['b']
        self.B = data['B']
        self.b_prime = data['b_prime']
        # 对称化 Q
        self.Q = (self.Q + self.Q.T) / 2.0

    def validate(self) -> bool:
        """验证数据一致性与合法性。

        检查项:
            - Q 对称（在 load 中已处理）
            - 各矩阵维度匹配
            - 无 NaN / Inf
        """
        assert self.Q.shape == (self.n, self.n), f"Q shape {self.Q.shape} != ({self.n}, {self.n})"
        assert self.c.shape == (self.n,), f"c shape {self.c.shape} != ({self.n},)"
        assert self.h.shape == (self.p,), f"h shape {self.h.shape} != ({self.p},)"
        assert self.A.shape == (self.m1, self.n), f"A shape {self.A.shape} != ({self.m1}, {self.n})"
        assert self.G.shape == (self.m1, self.p), f"G shape {self.G.shape} != ({self.m1}, {self.p})"
        assert self.b.shape == (self.m1,), f"b shape {self.b.shape} != ({self.m1},)"
        assert self.B.shape == (self.m2, self.n), f"B shape {self.B.shape} != ({self.m2}, {self.n})"
        assert self.b_prime.shape == (self.m2,), f"b_prime shape {self.b_prime.shape} != ({self.m2},)"
        assert not np.any(np.isnan(self.Q)), "Q contains NaN"
        assert not np.any(np.isinf(self.Q)), "Q contains Inf"
        return True

    def diagnose(self) -> DiagReport:
        """生成实例诊断报告（< 2s 内完成）。

        使用 eigvalsh 而非 eig 以加速特征值计算。
        """
        n = self.n
        # Q 稠密度
        q_nonzero = np.count_nonzero(self.Q) - np.count_nonzero(np.diag(self.Q))
        q_density = q_nonzero / max(n * (n - 1), 1)

        # Q 特征值（仅计算特征值，使用更快速的 eigvalsh）
        eigvals = np.linalg.eigvalsh(self.Q)
        q_eigval_range = (float(eigvals[0]), float(eigvals[-1]))
        q_condition = float(eigvals[-1] / max(abs(eigvals[0]), 1e-12))

        # G 秩
        g_rank = np.linalg.matrix_rank(self.G)
        g_full_rank = g_rank == min(self.m1, self.p)

        # 约束松弛统计（在 x=0 时）
        slack_binary = self.b_prime.copy()  # b' - B*0 = b'
        slack_mixed = self.b.copy()         # b - A*0 - G*0 = b
        slack_stats = {
            'binary': {'mean': float(slack_binary.mean()),
                      'min': float(slack_binary.min()),
                      'max': float(slack_binary.max())},
            'mixed': {'mean': float(slack_mixed.mean()),
                     'min': float(slack_mixed.min()),
                     'max': float(slack_mixed.max())},
        }

        # 可行解比例估计（粗糙估计：假设变量独立）
        est_feasible = max(0.01, min(1.0, np.prod(1.0 / (1.0 + np.exp(-slack_binary)))))

        # 推荐 subQUBO 大小
        if n <= 15:
            rec_size = n
        elif n <= 40:
            rec_size = min(18, n)
        elif n <= 80:
            rec_size = 15
        elif n <= 120:
            rec_size = 15
        else:
            rec_size = 12

        return DiagReport(
            Q_density=float(q_density),
            Q_eigval_range=q_eigval_range,
            Q_condition_number=float(q_condition),
            G_rank=int(g_rank),
            G_is_full_rank=bool(g_full_rank),
            constraint_slack_stats=slack_stats,
            estimated_feasible_density=float(est_feasible),
            recommended_sub_qubo_size=int(rec_size),
        )
