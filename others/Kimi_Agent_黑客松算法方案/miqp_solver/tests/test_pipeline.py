"""
测试模块 - 用随机数据验证整个pipeline
在比赛前运行此脚本确保所有组件工作正常
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_data
from utils.blocking import coupling_strength_blocks, random_blocks, extract_subqubo
from utils.validator import check_feasibility, compute_objective, quick_validation_report
from core.quantum_solver import solve_qubo_exact
from core.lp_solver import solve_lp_scipy
import config


def generate_test_data(n: int, p: int, m: int, m2: int, seed: int = 42):
    """
    生成随机测试数据
    """
    np.random.seed(seed)

    Q = np.random.randn(n, n)
    Q = (Q + Q.T) / 2  # 对称化
    c = np.random.randn(n)
    h = np.random.randn(p)
    A = np.random.randn(m, n)
    G = np.random.randn(m, p)
    # 确保可行域非空
    b = np.abs(np.random.randn(m)) * 10 + 5
    B = np.random.randn(m2, n)
    bp = np.abs(np.random.randn(m2)) * 5 + 2

    data_dir = config.DATA_DIR
    os.makedirs(data_dir, exist_ok=True)

    prefix = f"test_random_n{n}"
    np.save(os.path.join(data_dir, f"{prefix}_Q.npy"), Q)
    np.save(os.path.join(data_dir, f"{prefix}_c.npy"), c)
    np.save(os.path.join(data_dir, f"{prefix}_h.npy"), h)
    np.save(os.path.join(data_dir, f"{prefix}_A.npy"), A)
    np.save(os.path.join(data_dir, f"{prefix}_G.npy"), G)
    np.save(os.path.join(data_dir, f"{prefix}_b.npy"), b)
    np.save(os.path.join(data_dir, f"{prefix}_B.npy"), B)
    np.save(os.path.join(data_dir, f"{prefix}_bp.npy"), bp)

    print(f"[TEST] Generated test data: n={n}, p={p}, prefix={prefix}")
    return prefix


def test_data_loading():
    """测试数据加载"""
    print("\n" + "="*60)
    print("TEST 1: Data Loading")
    print("="*60)

    prefix = generate_test_data(n=10, p=3, m=2, m2=1)
    data = load_data(config.DATA_DIR, prefix)

    assert data["Q"] is not None, "Q matrix missing"
    assert data["c"] is not None, "c vector missing"
    assert data["n"] == 10, f"Expected n=10, got {data['n']}"

    print(f"[PASS] Data loaded: n={data['n']}, p={data['p']}, m={data['m']}")
    return True


def test_exact_solver():
    """测试精确求解器"""
    print("\n" + "="*60)
    print("TEST 2: Exact QUBO Solver")
    print("="*60)

    n = 8
    Q = np.random.randn(n, n)
    Q = (Q + Q.T) / 2
    c = np.random.randn(n)

    x_opt, obj_opt = solve_qubo_exact(Q, c)

    assert x_opt is not None, "Solver returned None"
    assert all(xi in [0, 1] for xi in x_opt), "x not binary"

    print(f"[PASS] Exact solver: obj={obj_opt:.4f}, x={x_opt}")
    return True


def test_lp_solver():
    """测试LP求解器"""
    print("\n" + "="*60)
    print("TEST 3: LP Solver")
    print("="*60)

    h = np.array([1.0, 2.0, 0.5])
    G = np.array([[1.0, 0.5, 0.0],
                  [0.0, 1.0, 0.5]])
    rhs = np.array([5.0, 3.0])

    y, success, obj = solve_lp_scipy(h, G, rhs)

    print(f"[PASS] LP solver: y={y}, success={success}, obj={obj:.4f}")
    return success


def test_blocking():
    """测试分块策略"""
    print("\n" + "="*60)
    print("TEST 4: Blocking Strategies")
    print("="*60)

    n = 30
    Q = np.random.randn(n, n)
    Q = (Q + Q.T) / 2

    blocks = coupling_strength_blocks(Q, block_size=10)

    # 检查覆盖性
    all_vars = set()
    for b in blocks:
        all_vars.update(b)

    assert len(all_vars) == n, f"Blocks cover {len(all_vars)}/{n} variables"
    assert len(blocks) > 0, "No blocks generated"

    print(f"[PASS] Coupling blocks: {len(blocks)} blocks, all {n} vars covered")
    return True


def test_subqubo_extraction():
    """测试subQUBO提取"""
    print("\n" + "="*60)
    print("TEST 5: subQUBO Extraction")
    print("="*60)

    n = 10
    Q = np.random.randn(n, n)
    Q = (Q + Q.T) / 2
    c = np.random.randn(n)

    # 固定前5个变量为1
    fixed = np.concatenate([np.ones(5), -np.ones(5)])
    block_vars = list(range(5, 10))

    Q_sub, c_sub, var_map, constant = extract_subqubo(Q, c, fixed, block_vars)

    assert Q_sub.shape == (5, 5), f"Expected (5,5), got {Q_sub.shape}"
    assert len(c_sub) == 5, f"Expected len 5, got {len(c_sub)}"

    print(f"[PASS] subQUBO extracted: shape={Q_sub.shape}, constant={constant:.4f}")
    return True


def test_full_pipeline():
    """测试完整pipeline"""
    print("\n" + "="*60)
    print("TEST 6: Full Pipeline (n=15)")
    print("="*60)

    from main import solve_single_instance

    prefix = generate_test_data(n=15, p=5, m=3, m2=2)
    data = load_data(config.DATA_DIR, prefix)

    result = solve_single_instance(data, config.OUTPUT_DIR, time_limit=60)

    assert result is not None, "Pipeline returned None"
    assert "x" in result, "Missing x in result"
    assert "y" in result, "Missing y in result"

    print(f"[PASS] Full pipeline: obj={result['objective']:.4f}")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "#"*60)
    print("# Running All Pre-Competition Tests")
    print("#"*60)

    tests = [
        test_data_loading,
        test_exact_solver,
        test_lp_solver,
        test_blocking,
        test_subqubo_extraction,
        test_full_pipeline,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    print("\n" + "#"*60)
    print(f"# Test Results: {passed} passed, {failed} failed")
    print("#"*60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
