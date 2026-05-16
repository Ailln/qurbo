"""变量选择器：基于多指标评分选择 subQUBO 变量子集。"""

from typing import List, Tuple, Optional
import numpy as np
from numpy import ndarray


class VariableSelector:
    """变量选择器。

    策略:
        'gradient': 翻转增益 + 对偶敏感度 + 不确定性
        'clustering': 谱聚类 + 集群价值评估
        'adaptive': 根据 Q 稠密度自动选择
    """

    def __init__(self, instance, config):
        self.inst = instance
        self.config = config
        self.alpha1 = config.alpha_flip_gain
        self.alpha2 = config.alpha_uncertainty
        self.alpha3 = config.alpha_coupling

    def select(self, x_current: ndarray, l_cont: ndarray,
               elite_pool, max_size: int) -> List[int]:
        """选择变量子集 S。"""
        n = self.inst.n
        scores = self._compute_scores(x_current, l_cont, elite_pool)

        # 选择 core 变量 (top half)
        core_size = max_size // 2
        core_indices = np.argsort(-scores)[:core_size].tolist()

        # 补全：按与 core 的耦合强度选择剩余变量
        remaining = [i for i in range(n) if i not in core_indices]
        if len(core_indices) < max_size and remaining:
            coupling_scores = np.zeros(len(remaining))
            for idx, i in enumerate(remaining):
                coupling_scores[idx] = np.sum(np.abs(self.inst.Q[i, core_indices]))
            n_fill = min(max_size - len(core_indices), len(remaining))
            fill_indices = [remaining[i] for i in np.argsort(-coupling_scores)[:n_fill]]
            S = sorted(core_indices + fill_indices)
        else:
            S = sorted(core_indices)

        return S

    def generate_neighborhoods(self, x_current: ndarray, l_cont: ndarray,
                               elite_pool, max_size: int,
                               num_nbrs: int) -> List[List[int]]:
        """生成多个互补邻域。"""
        neighborhoods = []

        # N1: 最佳策略选择（全局最优方向）
        s1 = self.select(x_current, l_cont, elite_pool, max_size)
        neighborhoods.append(s1)

        if num_nbrs >= 2:
            # N2: 与 N1 重叠率 < 30%，侧重不确定性高的变量
            scores = self._compute_scores(x_current, l_cont, elite_pool)
            uncertainty = self._uncertainty_scores(elite_pool)
            n = self.inst.n

            # 提高不确定性权重
            combined = 0.3 * scores + 0.7 * uncertainty
            # 排除 N1 中已选变量
            mask = np.ones(n)
            mask[s1] = 0
            combined = combined * mask

            s2_size = min(max_size, n - len(s1))
            if s2_size > 0:
                s2_candidates = np.argsort(-combined)[:s2_size].tolist()
                # 确保与 s1 重叠 < 30%
                overlap = len(set(s1) & set(s2_candidates))
                if overlap / max_size < 0.3:
                    neighborhoods.append(sorted(s2_candidates))
                else:
                    # 强制低重叠选择
                    available = [i for i in range(n) if i not in s1]
                    n_take = min(max_size, len(available))
                    s2 = np.random.choice(available, size=n_take, replace=False).tolist()
                    neighborhoods.append(sorted(s2))

        if num_nbrs >= 3:
            # N3: 纯随机多样化
            available = list(range(self.inst.n))
            s3 = np.random.choice(available, size=min(max_size, self.inst.n),
                                 replace=False).tolist()
            neighborhoods.append(sorted(s3))

        return neighborhoods

    def _compute_scores(self, x_current: ndarray, l_cont: ndarray,
                        elite_pool) -> ndarray:
        """计算综合评分。"""
        flip_gain = self._flip_gain_scores(x_current, l_cont)
        uncertainty = self._uncertainty_scores(elite_pool)
        # Coupling 在 select 中单独处理
        return (self.alpha1 * flip_gain +
                self.alpha2 * uncertainty)

    def _flip_gain_scores(self, x_current: ndarray, l_cont: ndarray) -> ndarray:
        """翻转增益评分。

        FlipGain(i) = |Q_ii + c_i + l_cont,i + 2 * sum_j Q_ij * x_j|
        """
        inst = self.inst
        linear_part = np.diag(inst.Q) + inst.c + l_cont
        quadratic_part = 2 * (inst.Q @ x_current - np.diag(inst.Q) * x_current)
        flip_gain = np.abs(linear_part + quadratic_part)
        # 归一化到 [0, 1]
        if flip_gain.max() > 1e-12:
            flip_gain = flip_gain / flip_gain.max()
        return flip_gain

    def _uncertainty_scores(self, elite_pool) -> ndarray:
        """不确定性评分。

        Uncertainty(i) = 4 * freq_i * (1 - freq_i)
        freq 来自 elite pool 中变量取 1 的频率
        """
        if elite_pool is None or elite_pool.size() == 0:
            return np.ones(self.inst.n) * 0.5
        freqs = elite_pool.get_frequencies()
        uncertainty = 4 * freqs * (1 - freqs)
        if uncertainty.max() > 1e-12:
            uncertainty = uncertainty / uncertainty.max()
        return uncertainty
