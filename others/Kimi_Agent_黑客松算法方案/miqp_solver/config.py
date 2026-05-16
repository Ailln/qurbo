"""
全局配置 - 3小时黑客松的核心参数
所有可调参数集中在此，方便比赛时快速调整
"""

import os

# ============ 量子计算参数 ============
QUANTUM_BACKEND = "qiskit"           # "qiskit" 或 "pennylane"
MAX_QUBITS = 20                       # 安全上限，硬件允许30但建议≤20
QAOA_REPS = 2                         # QAOA层数，黑客松用2层平衡质量与速度
QAOA_OPTIMIZER = "COBYLA"           # 经典优化器，COBYLA在无梯度时最快
QAOA_MAXITER = 100                    # 每次subQUBO的经典优化迭代次数
SHOTS = 1024                          # 量子测量次数

# ============ 分块策略参数 ============
BLOCK_SIZE = 15                       # 每个subQUBO的变量数，≤MAX_QUBITS
BLOCK_STRATEGY = "coupling"          # "random" | "coupling" | "spectral"
MAX_BLOCKS = 20                       # 最多分块数，控制总求解时间

# ============ Benders分解参数 ============
BENDERS_MAX_ITER = 50                 # 最大迭代次数
BENDERS_TOLERANCE = 1e-3             # 收敛容差
WARM_START = True                     # 是否用上一轮解热启动

# ============ 求解器选择 ============
LP_SOLVER = "scipy"                  # "scipy" | "cvxpy" | "pulp" | "numpy"

# ============ 时间控制（秒）===========
MAX_TIME_PER_TEST = 1800             # 每个test最多30分钟
TIMEOUT_SUBQUBO = 30                  # 单个subQUBO超时时间

# ============ 路径 ============
DATA_DIR = "./data"
OUTPUT_DIR = "./output"
SUBMISSION_DIR = "./submission"

# 自动创建目录
for d in [DATA_DIR, OUTPUT_DIR, SUBMISSION_DIR]:
    os.makedirs(d, exist_ok=True)
