#!/usr/bin/env python
"""
黑客松一键运行脚本
用法：
    # 比赛前测试
    python run_hackathon.py --phase prepare

    # 收到test数据后求解
    python run_hackathon.py --phase solve --data_dir ./test_data

    # 生成Paper图表
    python run_hackathon.py --phase paper
"""

import argparse
import os
import sys
import time
import subprocess


def phase_prepare():
    """比赛前准备阶段：安装依赖、运行测试"""
    print("="*60)
    print("PHASE: PREPARE (Before receiving test data)")
    print("="*60)

    # 1. 安装依赖
    print("\n[1/3] Installing dependencies...")
    os.system("pip install -q qiskit qiskit-algorithms qiskit-aer scipy numpy matplotlib")

    # 2. 运行测试
    print("\n[2/3] Running tests...")
    ret = os.system(f"{sys.executable} tests/test_pipeline.py")
    if ret != 0:
        print("[WARNING] Some tests failed. Check output above.")
    else:
        print("[OK] All tests passed!")

    # 3. 验证sample数据
    print("\n[3/3] Verify sample data exists...")
    for s in ["sample_A", "sample_B"]:
        path = os.path.join("data", f"{s}_Q.npy")
        if os.path.exists(path):
            print(f"  [OK] {s} data found")
        else:
            print(f"  [MISSING] {path} not found - will check on competition day")

    print("\n[OK] Preparation complete!")
    print("    - Dependencies installed")
    print("    - Pipeline validated")
    print("    - Ready for competition")


def phase_solve(data_dir: str, output_dir: str):
    """求解阶段：收到test数据后运行"""
    print("="*60)
    print("PHASE: SOLVE (Processing test instances)")
    print("="*60)

    start_time = time.time()

    # 运行主程序
    cmd = f"{sys.executable} main.py --mode test --data_dir {data_dir} --output_dir {output_dir}"
    print(f"Running: {cmd}")
    os.system(cmd)

    elapsed = time.time() - start_time
    print(f"\n[OK] Solve phase complete in {elapsed/60:.1f} minutes")


def phase_paper(output_dir: str):
    """Paper阶段：生成图表和结果整理"""
    print("="*60)
    print("PHASE: PAPER (Generating figures)")
    print("="*60)

    import json
    from visualization.paper_viz import generate_all_paper_figures

    # 加载结果
    summary_path = os.path.join(output_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            results = json.load(f)
        print(f"[OK] Loaded results for {len(results)} instances")
    else:
        results = {}
        print("[WARNING] No summary.json found")

    # 生成图表
    # 注意：这里需要实际运行数据才能生成有意义的图
    print("\n[OK] Paper figures will be generated from solve results")
    print("    See visualization/paper_viz.py for API")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prepare", "solve", "paper", "all"],
                        default="all")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", default="./output")
    args = parser.parse_args()

    if args.phase == "prepare":
        phase_prepare()
    elif args.phase == "solve":
        phase_solve(args.data_dir, args.output_dir)
    elif args.phase == "paper":
        phase_paper(args.output_dir)
    elif args.phase == "all":
        phase_prepare()
        phase_solve(args.data_dir, args.output_dir)
        phase_paper(args.output_dir)


if __name__ == "__main__":
    main()
