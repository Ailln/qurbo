import numpy as np
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'alpha-test')

from algorithm.data.instance import MIQPInstance
from algorithm.core.evaluator import ObjectiveEvaluator
from algorithm.config import auto_config
from algorithm.solver import HybridMIQPSolver


class TestIntegration:
    def test_end_to_end_sample_A(self):
        """端到端验证：sample_A 上完整流程。"""
        path = os.path.join(DATA_DIR, 'miqp_sample_A.npz')
        instance = MIQPInstance()
        instance.load(path)
        instance.validate()

        config = auto_config(instance)
        config.time_limit = 30
        # 测试时缩短 QAOA 开销
        config.qaoa_max_opt_steps = 8
        config.qaoa_multi_start = 1
        config.qaoa_shots = 256

        solver = HybridMIQPSolver(instance, config)
        result = solver.solve()

        assert result.best_objective > -np.inf
        assert result.total_time < 45

        # 验证可行性
        eval_check = ObjectiveEvaluator(instance).evaluate(result.best_x)
        assert eval_check.is_feasible == True

        # 应优于全零解
        zero_result = ObjectiveEvaluator(instance).evaluate(np.zeros(instance.n))
        assert result.best_objective >= zero_result.objective - 1e-6

    def test_end_to_end_sample_B(self):
        """端到端验证：sample_B 上完整流程。"""
        path = os.path.join(DATA_DIR, 'miqp_sample_B.npz')
        instance = MIQPInstance()
        instance.load(path)
        instance.validate()

        config = auto_config(instance)
        config.time_limit = 60
        config.qaoa_max_opt_steps = 8
        config.qaoa_multi_start = 1
        config.qaoa_shots = 256

        solver = HybridMIQPSolver(instance, config)
        result = solver.solve()

        assert result.best_objective > -np.inf
        assert result.total_time < 90

        eval_check = ObjectiveEvaluator(instance).evaluate(result.best_x)
        assert eval_check.is_feasible == True

        zero_result = ObjectiveEvaluator(instance).evaluate(np.zeros(instance.n))
        assert result.best_objective >= zero_result.objective - 1e-6

    def test_sa_only_mode(self):
        """纯 SA 模式运行。"""
        path = os.path.join(DATA_DIR, 'miqp_sample_A.npz')
        instance = MIQPInstance()
        instance.load(path)
        instance.validate()

        config = auto_config(instance)
        config.time_limit = 15
        config.max_qubits = 0  # 禁用 QAOA

        solver = HybridMIQPSolver(instance, config)
        result = solver.solve()

        assert result.best_objective > -np.inf
        eval_check = ObjectiveEvaluator(instance).evaluate(result.best_x)
        assert eval_check.is_feasible == True
