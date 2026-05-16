"""解修复器：将 subQUBO 的局部解修复为全局可行解。"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from numpy import ndarray


@dataclass
class RepairResult:
    """修复结果。"""
    x_repaired: ndarray
    is_feasible: bool
    num_flips: int
    repair_time: float
    objective: Optional[float]


class SolutionRepairer:
    """解修复器。

    修复流程（严格顺序）:
        Step 1: EMBED - 将局部解嵌入全局解
        Step 2: REPAIR BINARY CONSTRAINTS - 修复纯二元约束 Bx <= b'
        Step 3: REPAIR MIXED CONSTRAINT FEASIBILITY - 修复混合约束可行性
        Step 4: VALIDATE - 调用 evaluator 验证
    """

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.inst = evaluator.inst

    def repair(self, z_local: ndarray, S: List[int],
               x_current: ndarray) -> RepairResult:
        """修复单个解。

        最多允许翻转 floor(n/4) 个变量，超过则放弃。
        """
        import time
        t0 = time.perf_counter()

        # Step 1: EMBED
        x_new = x_current.copy()
        x_new[S] = z_local.copy()

        num_flips = 0
        max_flips = max(1, self.inst.n // 4)

        # Step 2: REPAIR BINARY CONSTRAINTS
        if self.inst.m2 > 0:
            violation = self.inst.B @ x_new - self.inst.b_prime
            while np.any(violation > 1e-10) and num_flips < max_flips:
                # 找到违反最严重的约束
                row = int(np.argmax(violation))
                # 候选：当前为 1 且在该约束中有正系数的变量
                candidates = [i for i in range(self.inst.n)
                             if x_new[i] > 0.5 and self.inst.B[row, i] > 1e-10]
                if not candidates:
                    break
                # 选择翻转目标损失最小的变量
                losses = self._compute_flip_losses(x_new, candidates)
                i_star = candidates[int(np.argmin(losses))]
                x_new[i_star] = 0
                num_flips += 1
                violation = self.inst.B @ x_new - self.inst.b_prime

        # Step 3: REPAIR MIXED CONSTRAINT FEASIBILITY
        if num_flips < max_flips:
            while num_flips < max_flips:
                lp_result = self.evaluator.solve_lp(x_new)
                if lp_result is not None:
                    break  # LP 可行

                # 释放约束空间：翻转对约束影响最大的活跃变量
                active_vars = [i for i in range(self.inst.n) if x_new[i] > 0.5]
                if not active_vars:
                    break

                impacts = []
                for i in active_vars:
                    # 计算该变量对混合约束的贡献（取正部分）
                    impact = np.sum(np.maximum(self.inst.A[:, i], 0))
                    impacts.append(impact)

                i_star = active_vars[int(np.argmax(impacts))]
                x_new[i_star] = 0
                num_flips += 1

        # Step 4: VALIDATE
        eval_result = self.evaluator.evaluate(x_new)
        repair_time = time.perf_counter() - t0

        return RepairResult(
            x_repaired=x_new,
            is_feasible=eval_result.is_feasible,
            num_flips=num_flips,
            repair_time=repair_time,
            objective=eval_result.objective if eval_result.is_feasible else None,
        )

    def repair_batch(self, solutions, S: List[int], x_current: ndarray,
                     top_k: int = 5) -> List[RepairResult]:
        """批量修复 top-k 解。"""
        results = []
        for z, energy, count in solutions[:top_k]:
            result = self.repair(z, S, x_current)
            if result.is_feasible:
                results.append(result)
        return results

    def _compute_flip_losses(self, x: ndarray, candidates: List[int]) -> ndarray:
        """计算翻转变量的目标损失。

        loss(i) = Q_ii + c_i + l_cont,i + 2 * sum_{j≠i} Q_ij * x_j
        """
        # 简化为贪心：优先翻转对目标影响最小的
        losses = np.zeros(len(candidates))
        for idx, i in enumerate(candidates):
            loss = self.inst.Q[i, i] + self.inst.c[i]
            for j in range(self.inst.n):
                if j != i:
                    loss += 2 * self.inst.Q[i, j] * x[j]
            losses[idx] = abs(loss)
        return losses
