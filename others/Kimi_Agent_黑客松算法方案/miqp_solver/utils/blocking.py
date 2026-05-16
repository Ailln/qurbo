"""
subQUBO分块模块 - 三种分块策略的实现
"""

import numpy as np
from typing import List, Tuple
import itertools


def random_blocks(n: int, block_size: int, seed: int = 42) -> List[List[int]]:
    """
    随机分块策略 - 最快，用于sample_A验证
    """
    rng = np.random.RandomState(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    blocks = []
    for i in range(0, n, block_size):
        block = indices[i:i + block_size]
        if len(block) > 0:
            blocks.append(block)
    return blocks


def coupling_strength_blocks(Q: np.ndarray, block_size: int) -> List[List[int]]:
    """
    耦合强度分块策略 - 推荐用于正式比赛
    将强耦合的变量放在同一个块内，减少块间耦合

    算法：
    1. 计算每个变量对的总耦合强度 |Q[i,j]| + |Q[j,i]|
    2. 贪心地将强耦合变量放入同一block
    """
    n = Q.shape[0]
    assigned = set()
    blocks = []

    # 计算耦合强度矩阵（对称化）
    coupling = np.abs(Q) + np.abs(Q.T)
    np.fill_diagonal(coupling, 0)

    while len(assigned) < n:
        # 从未分配变量中选一个作为种子
        unassigned = [i for i in range(n) if i not in assigned]
        seed = unassigned[0]

        block = [seed]
        assigned.add(seed)

        # 贪心添加与当前block耦合最强的变量
        while len(block) < block_size and len(assigned) < n:
            # 计算每个未分配变量与当前block的耦合强度
            best_var = None
            best_strength = -1

            for v in unassigned:
                if v in assigned:
                    continue
                strength = sum(coupling[v, b] for b in block)
                if strength > best_strength:
                    best_strength = strength
                    best_var = v

            if best_var is not None and best_strength > 0:
                block.append(best_var)
                assigned.add(best_var)
                unassigned.remove(best_var)
            else:
                break

        blocks.append(block)

    return blocks


def spectral_blocks(Q: np.ndarray, block_size: int, n_blocks: int = None) -> List[List[int]]:
    """
    谱分块策略 - 基于图的谱聚类，适合大规模问题
    """
    from sklearn.cluster import SpectralClustering

    n = Q.shape[0]
    coupling = np.abs(Q) + np.abs(Q.T)
    np.fill_diagonal(coupling, 0)

    # 归一化
    row_sums = coupling.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = coupling / row_sums

    if n_blocks is None:
        n_blocks = max(1, n // block_size)

    sc = SpectralClustering(n_clusters=n_blocks, affinity='precomputed',
                            assign_labels='kmeans', random_state=42)
    labels = sc.fit_predict(W)

    blocks = [[] for _ in range(n_blocks)]
    for i, lbl in enumerate(labels):
        blocks[lbl].append(i)

    # 如果某个block太大，进一步拆分
    result = []
    for block in blocks:
        if len(block) > block_size:
            for i in range(0, len(block), block_size):
                result.append(block[i:i + block_size])
        elif len(block) > 0:
            result.append(block)

    return result


def extract_subqubo(Q: np.ndarray, c: np.ndarray, fixed_vars: np.ndarray,
                    block_vars: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从完整QUBO中提取subQUBO

    Args:
        Q: 完整QUBO矩阵 (n, n)
        c: 完整一次项 (n,)
        fixed_vars: 当前固定的变量值 (n,)，-1表示未固定
        block_vars: 当前块的变量索引列表

    Returns:
        Q_sub: subQUBO矩阵 (k, k)
        c_sub: subQUBO一次项 (k,)
        var_map: 变量映射 [block_idx -> global_idx]
        constant: 由固定变量产生的常数项
    """
    k = len(block_vars)
    var_map = np.array(block_vars, dtype=int)

    # 提取子矩阵
    Q_sub = Q[np.ix_(block_vars, block_vars)].copy()
    c_sub = c[block_vars].copy()

    # 处理固定变量的贡献
    constant = 0.0
    fixed_mask = fixed_vars != -1
    fixed_indices = np.where(fixed_mask)[0]

    for i, bi in enumerate(block_vars):
        # 线性贡献：Q[bi, fj] * x_fj 和 Q[fj, bi] * x_fj
        for fj in fixed_indices:
            c_sub[i] += Q[bi, fj] * fixed_vars[fj]
            c_sub[i] += Q[fj, bi] * fixed_vars[fj]

    # 固定变量之间的二次项（常数项）
    for fi in fixed_indices:
        for fj in fixed_indices:
            constant += Q[fi, fj] * fixed_vars[fi] * fixed_vars[fj]

    # 固定变量的一次项
    for fi in fixed_indices:
        constant += c[fi] * fixed_vars[fi]

    return Q_sub, c_sub, var_map, constant


def merge_solution(x_global: np.ndarray, x_sub: np.ndarray,
                   var_map: np.ndarray) -> np.ndarray:
    """
    将subQUBO的解合并到全局解中
    """
    x_new = x_global.copy()
    for i, global_idx in enumerate(var_map):
        x_new[global_idx] = x_sub[i]
    return x_new


def evaluate_partial_objective(Q, c, x):
    """评估QUBO目标函数值"""
    return x.T @ Q @ x + c @ x
