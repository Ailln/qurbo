"""精英池：管理高质量可行解集合，提供统计信息用于 warm-start。"""

from typing import List, Tuple, Optional
import numpy as np
from numpy import ndarray


class ElitePool:
    """精英池。

    属性:
        max_size: 最大容量
        pool: 按 objective 降序排列的 EvalResult 列表

    多样性保证: Hamming 距离比例 >= DIVERSITY_THRESHOLD (默认 0.1)
    """

    DIVERSITY_THRESHOLD = 0.1

    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self._pool = []

    def size(self) -> int:
        """当前池大小。"""
        return len(self._pool)

    def add(self, result) -> bool:
        """添加解到精英池。

        返回 True 表示成功添加/替换。
        若新解与池中某解过于相似但目标值更优，则替换该旧解。
        """
        if not result.is_feasible:
            return False

        # 先检查是否已有相似解；若新解更优则替换
        for idx, existing in enumerate(self._pool):
            hamming_ratio = np.sum(result.x != existing.x) / len(result.x)
            if hamming_ratio < self.DIVERSITY_THRESHOLD:
                if result.objective > existing.objective:
                    self._pool[idx] = result
                    self._pool.sort(key=lambda r: r.objective, reverse=True)
                    return True
                return False

        if len(self._pool) < self.max_size:
            self._pool.append(result)
            self._pool.sort(key=lambda r: r.objective, reverse=True)
            return True

        # 已满：若优于最差解则替换
        if result.objective > self._pool[-1].objective:
            self._pool[-1] = result
            self._pool.sort(key=lambda r: r.objective, reverse=True)
            return True

        return False

    def get_best(self):
        """获取当前最优解。"""
        return self._pool[0] if self._pool else None

    def get_frequencies(self) -> ndarray:
        """变量取 1 的频率，用于 QAOA warm-start。

        Returns: (n,) 数组，每个元素的取值频率
        """
        if not self._pool:
            return np.array([])
        X = np.array([r.x for r in self._pool])
        return X.mean(axis=0)

    def get_correlations(self, top_k: int = 20) -> List[Tuple[int, int, float]]:
        """变量对的统计相关性，用于结构感知初始态（C3）。

        deviation[i,j] = |P(x_i=x_j) - P(x_i=1)P(x_j=1) - P(x_i=0)P(x_j=0)|

        Returns: 按相关性降序排列的 (i, j, corr) 列表
        """
        if len(self._pool) < 2:
            return []

        X = np.array([r.x for r in self._pool])
        n = X.shape[1]
        freqs = X.mean(axis=0)

        correlations = []
        for i in range(n):
            for j in range(i + 1, n):
                p_eq = np.mean(X[:, i] == X[:, j])
                p_both_1 = freqs[i] * freqs[j]
                p_both_0 = (1 - freqs[i]) * (1 - freqs[j])
                corr = abs(p_eq - p_both_1 - p_both_0)
                if corr > 0.1:  # 仅保留显著相关
                    correlations.append((i, j, corr))

        correlations.sort(key=lambda x: -x[2])
        return correlations[:top_k]

    def get_diversity_metric(self) -> float:
        """计算池内平均 Hamming 距离比例。"""
        if len(self._pool) < 2:
            return 0.0
        total_dist = 0.0
        count = 0
        for i in range(len(self._pool)):
            for j in range(i + 1, len(self._pool)):
                dist = np.sum(self._pool[i].x != self._pool[j].x) / len(self._pool[i].x)
                total_dist += dist
                count += 1
        return total_dist / max(count, 1)

    def get_all(self):
        """获取所有解（按目标值降序）。"""
        return self._pool.copy()

    def _is_duplicate(self, x_new: ndarray) -> bool:
        """检查是否已有足够相似的解。"""
        for result in self._pool:
            hamming_ratio = np.sum(x_new != result.x) / len(x_new)
            if hamming_ratio < self.DIVERSITY_THRESHOLD:
                return True
        return False
