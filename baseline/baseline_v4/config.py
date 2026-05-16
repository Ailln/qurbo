"""求解器配置。"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SolverConfig:
    """主配置，所有模块从此读取参数。"""

    # === 问题相关 ===
    instance_path: str = ""

    # === 时间控制 ===
    time_limit: float = 300.0
    init_time_ratio: float = 0.12
    single_iter_limit: float = 8.0

    # === subQUBO 尺寸 ===
    max_qubits: int = 20
    sub_qubo_size: int = 15
    min_sub_qubo_size: int = 6

    # === QAOA 配置 ===
    qaoa_layers: int = 1
    qaoa_shots: int = 512
    qaoa_optimizer: str = 'COBYLA'
    qaoa_max_opt_steps: int = 20
    qaoa_multi_start: int = 2
    qaoa_device: str = 'CPU'  # 'CPU' or 'GPU'

    # === SA 配置 ===
    sa_num_reads: int = 100
    sa_num_sweeps: int = 1000

    # === 策略配置 ===
    elite_pool_size: int = 20
    diversity_threshold: float = 0.1
    max_lns_iterations: int = 100
    num_neighborhoods: int = 2
    no_improve_restart: int = 10

    # === 创新组件开关 ===
    use_dual_rescaling: bool = True
    dual_rescaling_eta: float = 0.5
    use_structure_init: bool = True
    use_cut_penalty: bool = True

    # === 变量选择权重 ===
    alpha_flip_gain: float = 0.5
    alpha_uncertainty: float = 0.3
    alpha_coupling: float = 0.2


def auto_config(instance) -> SolverConfig:
    """根据问题规模自动生成合理配置。"""
    n, p, m1, m2 = instance.n, instance.p, instance.m1, instance.m2

    config = SolverConfig()

    # 时间限制
    if n <= 15:
        config.time_limit = 60
        config.sub_qubo_size = n
        config.qaoa_layers = 2
        config.qaoa_max_opt_steps = 80
        config.single_iter_limit = 10.0
    elif n <= 40:
        config.time_limit = 120
        config.sub_qubo_size = min(18, n)
        config.qaoa_layers = 1
        config.qaoa_max_opt_steps = 50
        config.single_iter_limit = 8.0
    elif n <= 80:
        config.time_limit = 300
        config.sub_qubo_size = 15
        config.qaoa_layers = 1
        config.qaoa_max_opt_steps = 30
        config.single_iter_limit = 6.0
    elif n <= 120:
        config.time_limit = 420
        config.sub_qubo_size = 15
        config.qaoa_max_opt_steps = 25
        config.single_iter_limit = 5.0
    else:
        config.time_limit = 540
        config.sub_qubo_size = 12
        config.qaoa_layers = 1
        config.qaoa_max_opt_steps = 20
        config.single_iter_limit = 4.0

    init_ratio_map = {
        60: 0.10,
        120: 0.12,
        300: 0.10,
        420: 0.08,
        540: 0.07,
    }
    config.init_time_ratio = init_ratio_map.get(config.time_limit, 0.10)

    # Q 矩阵稠密度影响
    Q_density = np.count_nonzero(instance.Q) / max(n * n, 1)
    if Q_density > 0.7:
        config.sub_qubo_size = min(config.sub_qubo_size + 2, config.max_qubits, n, 15)
    elif Q_density < 0.2:
        config.sub_qubo_size = max(config.sub_qubo_size - 3, config.min_sub_qubo_size)
    else:
        config.sub_qubo_size = min(config.sub_qubo_size, n, 15)

    # 连续变量比例影响
    if p > 2 * n:
        config.use_dual_rescaling = True
        config.dual_rescaling_eta = 0.8
    elif p == 0:
        config.use_dual_rescaling = False
        config.use_cut_penalty = False

    return config
