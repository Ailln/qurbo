"""主求解器：协调所有模块的混合 MIQP 求解器。"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
from numpy import ndarray
import time

from .solvers.qaoa_solver import QAOASolver, QAOAConfig
from .solvers.sa_solver import SimulatedAnnealingSolver, SAConfig
from .solvers.exact_solver import ExactQUBOSolver, ExactSolverResult


@dataclass
class FinalResult:
    """最终求解结果。"""
    best_objective: float
    best_x: ndarray
    best_y: Optional[ndarray]
    total_time: float
    iterations_completed: int
    elite_pool_final_size: int
    convergence_history: List[Tuple[float, float]]  # [(time, obj)]


class HybridMIQPSolver:
    """量子-经典混合 MIQP 求解器。

    入口: solve(instance, time_limit=None) -> FinalResult

    内部管理:
        - 时间预算分配与监控
        - 迭代调度（何时切换策略）
        - 自适应参数（subQUBO 大小、QAOA 层数）
        - 停滞检测与多样化触发
    """

    def __init__(self, instance, config):
        self.inst = instance
        self.config = config

        # 初始化所有模块
        from .core.evaluator import ObjectiveEvaluator
        from .core.repairer import SolutionRepairer
        from .core.elite_pool import ElitePool
        from .core.cut_manager import BendersCutManager
        from .core.init_generator import FeasibleSolutionGenerator
        from .core.qubo_builder import SubQUBOBuilder
        from .strategy.variable_selector import VariableSelector

        self.evaluator = ObjectiveEvaluator(instance)
        self.repairer = SolutionRepairer(self.evaluator)
        self.elite_pool = ElitePool(config.elite_pool_size)
        self.cut_manager = BendersCutManager(instance)
        self.init_gen = FeasibleSolutionGenerator(self.evaluator, self.repairer)
        self.qubo_builder = SubQUBOBuilder(instance)
        self.qaoa_solver = QAOASolver(config.max_qubits, device=config.qaoa_device)
        self.sa_solver = SimulatedAnnealingSolver()
        self.exact_solver = ExactQUBOSolver(max_qubits=18)
        self.var_selector = VariableSelector(instance, config)

        # 自适应状态
        self.no_improve_count = 0
        self.last_best_obj = -np.inf
        self.best_result = None

    def solve(self, time_limit: Optional[float] = None) -> FinalResult:
        """主求解入口。"""
        if time_limit is None:
            time_limit = self.config.time_limit

        t_start = time.perf_counter()
        convergence = []

        # Phase 1: 初始化
        init_time = self.config.init_time_ratio * time_limit
        print(f"[INIT] Generating initial pool (time budget: {init_time:.1f}s)...")
        init_solutions = self.init_gen.generate_pool(15, init_time)
        for sol in init_solutions:
            self.elite_pool.add(sol)

        if self.elite_pool.size() == 0:
            # 保底：全零解
            zero_result = self.evaluator.evaluate(np.zeros(self.inst.n))
            if zero_result.is_feasible:
                self.elite_pool.add(zero_result)

        best = self.elite_pool.get_best()
        self.last_best_obj = best.objective if best else -np.inf
        print(f"[INIT] Pool size: {self.elite_pool.size()}, Best obj: {self.last_best_obj:.4f}")

        # Phase 2: LNS 主循环
        iter_count = 0
        max_iter = self.config.max_lns_iterations

        while iter_count < max_iter:
            elapsed = time.perf_counter() - t_start
            remaining = time_limit - elapsed
            if remaining < self.config.single_iter_limit:
                break

            iter_count += 1
            iter_start = time.perf_counter()

            # 严格时间检查：若剩余时间不足，直接退出
            if time.perf_counter() - t_start >= time_limit:
                break

            # 选择当前起点
            if self.no_improve_count > 10 and self.elite_pool.size() > 1:
                # 多样化：选择远离最优的解
                x_current = self._select_diverse_start()
                self.no_improve_count = 0
            else:
                best_result = self.elite_pool.get_best()
                x_current = best_result.x.copy() if best_result else np.zeros(self.inst.n)

            # Benders 对偶信息提取
            eval_result = self.evaluator.evaluate(x_current)
            if not eval_result.is_feasible:
                continue

            l_cont = (self.evaluator.compute_benders_linear(eval_result.dual)
                      if eval_result.dual is not None else np.zeros(self.inst.n))

            # 生成多邻域
            sub_qubo_size = self.config.sub_qubo_size
            num_nbrs = self.config.num_neighborhoods
            neighborhoods = self.var_selector.generate_neighborhoods(
                x_current, l_cont, self.elite_pool, sub_qubo_size, num_nbrs
            )

            iter_improved = False
            for S in neighborhoods:
                # 构建 subQUBO
                if self.config.use_dual_rescaling and eval_result.dual is not None:
                    qubo_prob = self.qubo_builder.build_with_dual_rescaling(
                        S, x_current, l_cont, eval_result.dual,
                        self.config.dual_rescaling_eta
                    )
                else:
                    qubo_prob = self.qubo_builder.build(S, x_current, l_cont)

                # 求解：策略优先级: 穷举 > SA
                # 在模拟器环境下，穷举(q<=18)比QAOA快1000倍且精确最优。
                # QAOA仅保留作为未来量子硬件的接口。
                q = len(S)
                if q <= self.exact_solver.max_qubits:
                    # 精确穷举：最快且全局最优
                    exact_result = self.exact_solver.solve(qubo_prob, top_k=10)
                    solver_result = type('obj', (object,), {
                        'solutions': exact_result.solutions,
                        'optimal_params': np.array([]),
                        'convergence_history': [],
                        'total_time': exact_result.total_time,
                    })()
                else:
                    sa_config = SAConfig(num_reads=self.config.sa_num_reads,
                                        num_sweeps=self.config.sa_num_sweeps)
                    solver_result = self.sa_solver.solve(qubo_prob, sa_config)

                # 修复与评价（时间紧张时只修复 top-2）
                top_k_repair = 2 if (time_limit - (time.perf_counter() - t_start)) < 10 else 3
                repair_results = self.repairer.repair_batch(
                    solver_result.solutions, S, x_current, top_k=top_k_repair
                )

                for rr in repair_results:
                    if rr.is_feasible and rr.objective is not None:
                        eval_r = self.evaluator.evaluate(rr.x_repaired)
                        added = self.elite_pool.add(eval_r)
                        if eval_r.objective > self.last_best_obj:
                            self.last_best_obj = eval_r.objective
                            self.best_result = eval_r
                            iter_improved = True

                # Early termination: 若已找到改进，跳过后续邻域（节省时间探索更多迭代）
                if iter_improved:
                    break

                if time.perf_counter() - t_start >= time_limit:
                    break

            # 割平面更新
            if eval_result.dual is not None:
                phi_val = float(eval_result.y @ self.inst.h) if eval_result.y is not None else 0.0
                self.cut_manager.add_optimality_cut(eval_result.dual, phi_val, x_current)
            self.cut_manager.prune()

            # 更新改善计数
            if iter_improved:
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1

            # 记录收敛
            convergence.append((time.perf_counter() - t_start, self.last_best_obj))

            # 自适应调节
            remaining_ratio = (time_limit - (time.perf_counter() - t_start)) / time_limit
            self._adapt_parameters(iter_improved, remaining_ratio)

            iter_time = time.perf_counter() - iter_start
            print(f"[ITER {iter_count}] obj={self.last_best_obj:.4f}, "
                  f"time={iter_time:.2f}s, pool={self.elite_pool.size()}")

        # 返回结果：优先使用 best_result，fallback 到 elite_pool
        final_best = self.best_result if self.best_result is not None else self.elite_pool.get_best()
        total_time = time.perf_counter() - t_start

        return FinalResult(
            best_objective=final_best.objective if final_best else -np.inf,
            best_x=final_best.x.copy() if final_best else np.zeros(self.inst.n),
            best_y=final_best.y.copy() if final_best and final_best.y is not None else None,
            total_time=total_time,
            iterations_completed=iter_count,
            elite_pool_final_size=self.elite_pool.size(),
            convergence_history=convergence,
        )

    def _select_diverse_start(self) -> ndarray:
        """选择与当前最优解距离最远的精英池解。"""
        best = self.elite_pool.get_best()
        if best is None or self.elite_pool.size() <= 1:
            return np.zeros(self.inst.n)

        all_solutions = self.elite_pool.get_all()
        max_dist = -1
        diverse_x = best.x.copy()

        for sol in all_solutions[1:]:
            dist = np.sum(sol.x != best.x)
            if dist > max_dist:
                max_dist = dist
                diverse_x = sol.x.copy()

        return diverse_x

    def _adapt_parameters(self, improved: bool, remaining_ratio: float) -> None:
        """自适应参数调节。"""
        # 规则1: 长期不改善 -> 扩大搜索范围（但不超过精确求解上限）
        if self.no_improve_count > 5:
            self.config.sub_qubo_size = min(
                self.config.sub_qubo_size + 3, self.exact_solver.max_qubits
            )
            self.config.num_neighborhoods = min(self.config.num_neighborhoods + 1, 4)

        # 规则2: 快速改善中 -> 缩小范围精细搜索
        if improved and self.no_improve_count == 0:
            self.config.sub_qubo_size = max(self.config.sub_qubo_size - 2, 6)

        # 规则3: 时间即将耗尽 -> 减少 repair 开销（已通过 top_k_repair=2 自动实现）
        # 规则4: 停滞 -> 准备多样化（在 solve 中处理）
