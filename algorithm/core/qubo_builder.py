"""subQUBO 构建器：从全局 MIQP 提取局部 QUBO 子问题。"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from numpy import ndarray


@dataclass
class QUBOProblem:
    """QUBO 子问题。"""
    matrix: ndarray         # (|S| x |S|) 上三角 QUBO 矩阵
    offset: float           # 常数偏移
    variable_map: List[int] # 局部索引到全局索引的映射 S
    local_constraints: Dict # B_S, r_S
    is_minimization: bool   # True (QUBO 标准形式是最小化)


class SubQUBOBuilder:
    """subQUBO 构建器。

    核心推导:
        原始最大化目标在变量子集 S 上的局部形式:
        F_S(z) = z^T Q_SS z + (c_S + l_cont,S + 2 Q_S,Sbar x_Sbar)^T z + const

        转为最小化 QUBO:
        E_S(z) = -F_S(z) = z^T (-Q_SS) z + (-effective_linear)^T z
    """

    def __init__(self, instance):
        self.inst = instance

    def build(self, S: List[int], x_current: ndarray,
              l_cont: ndarray, lambda_B: Optional[float] = None) -> QUBOProblem:
        """构建标准 subQUBO。

        Args:
            S: 选择的变量索引列表
            x_current: 当前完整解 (n,)
            l_cont: Benders 对偶引导线性项 (n,)
            lambda_B: 约束惩罚权重（默认自动计算）

        Returns:
            QUBOProblem: 上三角 QUBO 矩阵
        """
        inst = self.inst
        q = len(S)
        S_set = set(S)
        Sbar = [i for i in range(inst.n) if i not in S_set]

        # 提取子矩阵
        Q_SS = inst.Q[np.ix_(S, S)]
        c_S = inst.c[S]
        l_cont_S = l_cont[S]

        # 计算有效线性项: c_S + l_cont,S + 2 * Q_{S,Sbar} * x_{Sbar}
        cross_term = np.zeros(q)
        if len(Sbar) > 0:
            Q_SSbar = inst.Q[np.ix_(S, Sbar)]
            cross_term = 2 * Q_SSbar @ x_current[Sbar]

        effective_linear = c_S + l_cont_S + cross_term

        # 构建最小化 QUBO（上三角形式）
        qubo = np.zeros((q, q))
        for i in range(q):
            # 对角项: -Q_SS[i,i] - effective_linear[i]
            qubo[i, i] = -Q_SS[i, i] - effective_linear[i]
            for j in range(i + 1, q):
                # 上三角项: -2 * Q_SS[i,j]
                qubo[i, j] = -2 * Q_SS[i, j]

        # 常数偏移（最大化问题中被忽略的常数项）
        offset = -np.sum(effective_linear * 0)  # 简化

        # 提取局部约束
        local_constraints = {}
        if inst.m2 > 0:
            B_S = inst.B[:, S]
            local_constraints = {'B_S': B_S, 'r_S': inst.b_prime}

        # 添加约束惩罚
        if local_constraints and inst.m2 > 0 and q > 0:
            if lambda_B is None:
                diag_abs = np.abs(np.diag(qubo))
                lambda_B = 2 * max(np.max(diag_abs) if diag_abs.size > 0 else 1.0, 1.0)
            qubo = self._add_constraint_penalty(
                qubo, local_constraints['B_S'],
                local_constraints['r_S'], lambda_B
            )

        return QUBOProblem(
            matrix=qubo,
            offset=offset,
            variable_map=S,
            local_constraints=local_constraints,
            is_minimization=True,
        )

    def build_with_dual_rescaling(self, S: List[int], x_current: ndarray,
                                   l_cont: ndarray, dual: ndarray,
                                   eta: float) -> QUBOProblem:
        """使用对偶重标度构建 subQUBO（创新 C2）。

        sensitivity_i = |A_:,i^T u*|
        factor_i = 1 + eta * sensitivity_i / max(sensitivity)
        Q_SS_rescaled[i,j] = Q_SS[i,j] * sqrt(factor_i * factor_j)
        """
        inst = self.inst
        q = len(S)

        # 基础构建
        qubo_prob = self.build(S, x_current, l_cont)

        # 计算敏感度
        sensitivity = np.abs(inst.A[:, S].T @ dual)
        if sensitivity.size == 0:
            # 退化：S 为空或无敏感度
            return self.build(S, x_current, l_cont)
        max_sens = np.max(sensitivity) if np.max(sensitivity) > 1e-12 else 1.0
        factors = 1 + eta * sensitivity / max_sens

        # 重标度 Q 矩阵
        Q_SS = inst.Q[np.ix_(S, S)].copy()
        scale_matrix = np.sqrt(np.outer(factors, factors))
        Q_rescaled = Q_SS * scale_matrix

        # 重新构建 QUBO
        c_S = inst.c[S]
        l_cont_S = l_cont[S]
        S_set = set(S)
        Sbar = [i for i in range(inst.n) if i not in S_set]
        cross_term = np.zeros(q)
        if len(Sbar) > 0:
            Q_SSbar = inst.Q[np.ix_(S, Sbar)]
            cross_term = 2 * Q_SSbar @ x_current[Sbar]

        effective_linear = c_S + l_cont_S + cross_term

        qubo = np.zeros((q, q))
        for i in range(q):
            qubo[i, i] = -Q_rescaled[i, i] - effective_linear[i]
            for j in range(i + 1, q):
                qubo[i, j] = -2 * Q_rescaled[i, j]

        qubo_prob.matrix = qubo
        return qubo_prob

    def _add_constraint_penalty(self, qubo: ndarray, B_S: ndarray,
                                 r_S: ndarray, penalty_weight: float) -> ndarray:
        """添加二次约束惩罚项。

        对约束 B_S z <= r_S，使用二次惩罚:
        penalty = lambda * sum_k max(0, B_S[k,:] z - r_S[k])^2
        """
        q = qubo.shape[0]
        for k in range(B_S.shape[0]):
            b_row = B_S[k, :]
            r_k = r_S[k]
            # 展开 max(0, b^T z - r)^2 在二元情况下
            # = sum_i sum_j b_i b_j z_i z_j - 2r sum_i b_i z_i + r^2
            for i in range(q):
                # 对角项: b_i^2 - 2r * b_i (注意 z_i^2 = z_i)
                qubo[i, i] += penalty_weight * (b_row[i] ** 2 - 2 * r_k * b_row[i])
                for j in range(i + 1, q):
                    # 交叉项: 2 * b_i * b_j (上下三角各一半)
                    qubo[i, j] += penalty_weight * 2 * b_row[i] * b_row[j]
            # 常数项 r^2 被忽略（不影响优化）
        return qubo
