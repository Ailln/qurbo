"""Benders 割平面管理器：管理 optimality cuts 和 feasibility cuts。"""

from typing import List, Tuple, Optional
import numpy as np
from numpy import ndarray


class BendersCutManager:
    """Benders 割平面管理器。

    Optimality Cut 数学形式:
        在点 x_k 处，连续子问题的最优对偶 u_k* 给出:
        φ(x) <= φ(x_k) + (-A^T u_k*)^T (x - x_k)
        即: φ(x) <= b^T u_k* - (A^T u_k*)^T x

    管理规则:
        - Optimality cuts: 保留最近 15 个迭代内的
        - Feasibility cuts: 永不删除
        - 池上限: 最多 50 个 optimality cuts
    """

    def __init__(self, instance):
        self.inst = instance
        self.optimality_cuts = []
        # (dual, phi_val, x_at, age)
        self.feasibility_cuts = []
        # (a, beta) for a^T x <= beta
        self.max_opt_cuts = 50
        self.max_age = 15

    def add_optimality_cut(self, dual: ndarray, phi_val: float,
                           x_at: ndarray) -> None:
        """添加 optimality cut。"""
        self.optimality_cuts.append((dual.copy(), phi_val, x_at.copy(), 0))
        # 保持上限
        if len(self.optimality_cuts) > self.max_opt_cuts:
            # 删除最旧的
            self.optimality_cuts.sort(key=lambda c: c[3], reverse=True)
            self.optimality_cuts = self.optimality_cuts[:self.max_opt_cuts]

    def add_feasibility_cut(self, x_infeasible: ndarray) -> None:
        """添加 feasibility cut。

        从不可行解 x_infeasible 构造 cut。
        简化版本：使用 Benders 子问题的极值射线。
        """
        # 标记为不可行区域
        # 实际实现中可以从 LP 不可性证明中提取 cut
        pass

    def get_penalty_terms(self, x_current: ndarray) -> Tuple[ndarray, ndarray]:
        """获取当前违反的割平面对应的惩罚项。

        Returns:
            (linear_penalty, quadratic_penalty)
            可添加到 subQUBO 中
        """
        n = self.inst.n
        linear = np.zeros(n)
        quadratic = np.zeros((n, n))
        lambda_cut = 1.0  # 割平面惩罚系数

        for dual, phi_val, x_at, age in self.optimality_cuts:
            # 计算在当前 x 处的违反
            a = -self.inst.A.T @ dual  # = l_cont
            beta = self.inst.b @ dual - phi_val

            # 割平面: a^T x <= beta
            violation = a @ x_current - beta
            if violation > 0:
                # 添加惩罚
                linear += 2 * lambda_cut * violation * a

        return linear, quadratic

    def prune(self, max_age: int = 15) -> int:
        """删除过期的 optimality cuts。

        Returns: 删除的割平面数量
        """
        # 增加所有 cuts 的 age
        for i in range(len(self.optimality_cuts)):
            self.optimality_cuts[i] = (
                self.optimality_cuts[i][0],
                self.optimality_cuts[i][1],
                self.optimality_cuts[i][2],
                self.optimality_cuts[i][3] + 1,
            )

        before = len(self.optimality_cuts)
        self.optimality_cuts = [c for c in self.optimality_cuts if c[3] <= max_age]
        return before - len(self.optimality_cuts)
