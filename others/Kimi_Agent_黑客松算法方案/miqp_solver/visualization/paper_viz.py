"""
Paper图表快速生成模块
为算法说明文档生成必要的可视化图表
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无头环境
import matplotlib.pyplot as plt
from typing import Dict, List
import os


def plot_convergence(history: dict, save_path: str = "convergence.png"):
    """
    绘制Benders迭代收敛曲线
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    iters = history.get("iterations", [])
    objs = history.get("objectives", [])

    if len(iters) > 0 and len(objs) > 0:
        ax.plot(iters, objs, "b-o", linewidth=2, markersize=4, label="Objective Value")
        best = history.get("best_obj", max(objs) if objs else 0)
        ax.axhline(y=best, color="r", linestyle="--", label=f"Best: {best:.4f}")

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Objective Value", fontsize=12)
    ax.set_title("Benders Decomposition Convergence", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[VIZ] Convergence plot saved: {save_path}")
    return save_path


def plot_block_structure(Q: np.ndarray, blocks: List[List[int]],
                         save_path: str = "blocks.png"):
    """
    绘制分块结构热力图
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # 重排矩阵以显示分块结构
    block_order = []
    for block in blocks:
        block_order.extend(block)

    Q_perm = Q[np.ix_(block_order, block_order)]

    # 绘制热力图
    im = ax.imshow(np.abs(Q_perm), cmap="YlOrRd", aspect="auto")

    # 画分块边界
    pos = 0
    for block in blocks[:-1]:
        pos += len(block)
        ax.axhline(y=pos - 0.5, color="blue", linewidth=1)
        ax.axvline(x=pos - 0.5, color="blue", linewidth=1)

    ax.set_title("QUBO Matrix Block Structure", fontsize=14)
    ax.set_xlabel("Variable Index", fontsize=12)
    ax.set_ylabel("Variable Index", fontsize=12)
    plt.colorbar(im, ax=ax, label="|Q(i,j)|")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[VIZ] Block structure saved: {save_path}")
    return save_path


def plot_results_summary(results: dict, save_path: str = "summary.png"):
    """
    绘制所有test实例的结果汇总图
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    prefixes = sorted(results.keys())
    objs = [results[p]["objective"] for p in prefixes]
    times = [results[p].get("time", 0) for p in prefixes]

    # 目标值
    axes[0].bar(prefixes, objs, color="steelblue", edgecolor="black")
    axes[0].set_title("Objective Values", fontsize=14)
    axes[0].set_ylabel("Objective", fontsize=12)
    axes[0].tick_params(axis="x", rotation=45)

    # 求解时间
    axes[1].bar(prefixes, times, color="coral", edgecolor="black")
    axes[1].set_title("Solve Time (seconds)", fontsize=14)
    axes[1].set_ylabel("Time (s)", fontsize=12)
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[VIZ] Summary plot saved: {save_path}")
    return save_path


def generate_circuit_diagram(n_qubits: int, reps: int = 2,
                             save_path: str = "circuit.png"):
    """
    生成QAOA电路示意图（简化版）
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制简化的电路图
    y_positions = list(range(n_qubits - 1, -1, -1))
    x_start = 0.5

    # 初始Hadamard
    for y in y_positions:
        ax.plot([x_start, x_start + 0.3], [y, y], "k-", linewidth=1.5)
        ax.text(x_start + 0.15, y + 0.2, "H", fontsize=8, ha="center")

    x = x_start + 0.3

    # QAOA层
    for p in range(reps):
        # Cost层标注
        ax.text(x + 0.5, n_qubits + 0.5, f"Cost (p={p+1})",
                fontsize=10, ha="center", color="blue")

        # ZZ门（简化表示）
        for i in range(n_qubits):
            for j in range(i + 1, min(i + 2, n_qubits)):
                y1, y2 = y_positions[i], y_positions[j]
                ax.plot([x, x + 1], [y1, y1], "k-", linewidth=1)
                ax.plot([x, x + 1], [y2, y2], "k-", linewidth=1)
                ax.plot([x + 0.5, x + 0.5], [y1, y2], "k-", linewidth=1)
                ax.plot(x + 0.5, y1, "ko", markersize=6)
                ax.plot(x + 0.5, y2, "ko", markersize=6)
                ax.text(x + 0.7, (y1 + y2) / 2, "ZZ", fontsize=6, color="red")

        x += 1.5

        # Mixer层
        ax.text(x + 0.3, n_qubits + 0.5, f"Mixer (p={p+1})",
                fontsize=10, ha="center", color="green")
        for y in y_positions:
            ax.plot([x, x + 0.6], [y, y], "k-", linewidth=1.5)
            ax.text(x + 0.3, y + 0.2, "RX", fontsize=8, ha="center", color="green")

        x += 0.8

    # 测量
    for y in y_positions:
        ax.plot([x, x + 0.5], [y, y], "k-", linewidth=1.5)
    ax.text(x + 0.25, -1, "Measure", fontsize=10, ha="center")

    ax.set_xlim(0, x + 1)
    ax.set_ylim(-2, n_qubits + 2)
    ax.set_title(f"QAOA Circuit (n={n_qubits}, reps={reps})", fontsize=14)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[VIZ] Circuit diagram saved: {save_path}")
    return save_path


def generate_all_paper_figures(data: dict, blocks: list, history: dict,
                                results: dict, output_dir: str):
    """
    一键生成Paper所需的所有图表
    """
    os.makedirs(output_dir, exist_ok=True)

    figures = {}

    # 收敛曲线
    if history.get("iterations"):
        figures["convergence"] = plot_convergence(
            history,
            os.path.join(output_dir, "fig_convergence.png")
        )

    # 分块结构
    if blocks and data.get("Q") is not None:
        figures["blocks"] = plot_block_structure(
            data["Q"], blocks,
            os.path.join(output_dir, "fig_blocks.png")
        )

    # 电路图
    figures["circuit"] = generate_circuit_diagram(
        n_qubits=min(data.get("n", 5), 5), reps=2,
        save_path=os.path.join(output_dir, "fig_circuit.png")
    )

    # 结果汇总
    if results:
        figures["summary"] = plot_results_summary(
            results,
            os.path.join(output_dir, "fig_summary.png")
        )

    print(f"[VIZ] All figures generated in {output_dir}")
    return figures
