"""
数据加载模块 - 统一的数据加载和预处理
"""

import numpy as np
import os
import json


def load_data(data_dir: str, prefix: str):
    """
    加载MIQP问题数据

    期望的数据文件：
        {prefix}_Q.npy  - 二次项矩阵 (n, n)
        {prefix}_c.npy  - 一次项向量 (n,)
        {prefix}_h.npy  - 连续变量系数 (p,)
        {prefix}_A.npy  - 混合约束矩阵 (m, n)
        {prefix}_G.npy  - 连续变量约束矩阵 (m, p)
        {prefix}_b.npy  - 混合约束右端项 (m,)
        {prefix}_B.npy  - 二元变量约束矩阵 (m2, n)
        {prefix}_bp.npy - 二元变量约束右端项 (m2,)

    Returns:
        dict: {Q, c, h, A, G, b, B, bp, n, p, m, m2}
    """
    def _load(fname):
        path = os.path.join(data_dir, f"{prefix}_{fname}.npy")
        if os.path.exists(path):
            return np.load(path)
        return None

    Q = _load("Q")
    c = _load("c")
    h = _load("h")
    A = _load("A")
    G = _load("G")
    b = _load("b")
    B = _load("B")
    bp = _load("bp")

    n = Q.shape[0] if Q is not None else 0
    p = h.shape[0] if h is not None else 0
    m = b.shape[0] if b is not None else 0
    m2 = bp.shape[0] if bp is not None else 0

    return {
        "Q": Q, "c": c, "h": h,
        "A": A, "G": G, "b": b,
        "B": B, "bp": bp,
        "n": n, "p": p, "m": m, "m2": m2,
        "prefix": prefix
    }


def load_data_from_csv(data_dir: str, prefix: str):
    """
    备用：从CSV加载（如果主办方提供CSV格式）
    """
    def _load_csv(fname):
        path = os.path.join(data_dir, f"{prefix}_{fname}.csv")
        if os.path.exists(path):
            return np.loadtxt(path, delimiter=",")
        return None

    Q = _load_csv("Q")
    c = _load_csv("c")
    h = _load_csv("h")
    A = _load_csv("A")
    G = _load_csv("G")
    b = _load_csv("b")
    B = _load_csv("B")
    bp = _load_csv("bp")

    # 处理1维数组的形状
    if c is not None and c.ndim == 0:
        c = np.array([c])
    if h is not None and h.ndim == 0:
        h = np.array([h])
    if b is not None and b.ndim == 0:
        b = np.array([b])
    if bp is not None and bp.ndim == 0:
        bp = np.array([bp])

    n = Q.shape[0] if Q is not None else 0
    p = h.shape[0] if h is not None else 0
    m = b.shape[0] if b is not None else 0
    m2 = bp.shape[0] if bp is not None else 0

    return {
        "Q": Q, "c": c, "h": h,
        "A": A, "G": G, "b": b,
        "B": B, "bp": bp,
        "n": n, "p": p, "m": m, "m2": m2,
        "prefix": prefix
    }


def save_solution(output_dir: str, prefix: str, x: np.ndarray, y: np.ndarray,
                  obj_val: float, info: dict = None):
    """
    保存解到标准格式
    """
    sol = {
        "prefix": prefix,
        "x": x.tolist(),
        "y": y.tolist(),
        "objective": float(obj_val),
        "info": info or {}
    }
    path = os.path.join(output_dir, f"{prefix}_solution.json")
    with open(path, "w") as f:
        json.dump(sol, f, indent=2)

    # 也保存为npy方便后续处理
    np.save(os.path.join(output_dir, f"{prefix}_x.npy"), x)
    np.save(os.path.join(output_dir, f"{prefix}_y.npy"), y)

    print(f"[OK] Solution saved: {path}, obj={obj_val:.6f}")
    return path
