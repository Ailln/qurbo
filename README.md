# QURBO = Quantum Turbo Optimization

QURBO 是一个面向混合整数二次规划（MIQP, Mixed-Integer Quadratic Programming）的量子优化实验仓库。当前版本提供了数据读取示例、alpha 测试样例数据，以及一个基于 Qiskit Aer 的 QAOA-like baseline，用于快速验证问题格式、跑通求解流程并生成可提交/可分析的解文件。

本仓库目前更偏向研究和竞赛 baseline：实现尽量轻量、依赖少、便于修改，不追求工业级求解器的完整性或最优性保证。

## 问题形式

代码中的 baseline 处理如下形式的 MIQP：

```text
maximize    x^T Q x + c^T x + h^T y
subject to  A x + G y <= b
            B x <= b_prime
            x in {0, 1}^n
            y >= 0
```

其中：

- `x` 是二元决策变量，长度为 `n`。
- `y` 是连续非负变量，长度为 `p`。
- `Q, c, h` 定义目标函数。
- `A, G, b` 定义混合约束。
- `B, b_prime` 定义纯二元约束。

## 仓库结构

```text
.
├── baseline/
│   ├── baseline_miqp_qaoa.py   # QAOA-like MIQP baseline
│   ├── baseline_miqp_qaoa_v2.py # 改进候选池与约束激活策略
│   ├── baseline_miqp_qaoa_v3.py # LP 代理价值 + 聚类 subQUBO + 修复重启
│   ├── bruteforce_check.py     # 小规模实例暴力枚举校验器
│   ├── run_base.sh             # GPU baseline 运行示例
│   ├── run_base_cpu.sh         # CPU baseline 运行示例，适合 macOS 或无 CUDA 环境
│   ├── run_base_v3.sh          # GPU v3 运行示例
│   └── run_base_v3_cpu.sh      # CPU v3 运行示例
├── data/
│   └── alpha-test/
│       ├── miqp_sample_A.npz   # 样例实例 A
│       ├── miqp_sample_B.npz   # 样例实例 B
│       └── 读取数据示例.ipynb   # NPZ 数据读取示例
├── doc/
│   └── 2026量子计算大赛·混合整数优化问题赛道.docx
├── LICENSE
└── README.md
```

## 环境安装

推荐使用 conda 创建独立环境。Qiskit 相关依赖需要使用指定版本，避免 Aer 后端和 Qiskit 主包版本不匹配。

```bash
conda create -n qurbo python=3.11 -y
conda activate qurbo

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

默认依赖文件 `requirements.txt` 使用 GPU 版 Aer，适合有 NVIDIA CUDA 环境的机器。macOS 或没有 CUDA 的机器可以直接使用 CPU 版依赖：

```bash
python -m pip install -r requirements-cpu.txt
```

安装完成后可以检查核心依赖：

```bash
python - <<'PY'
import numpy
import scipy
import qiskit
import qiskit_aer

print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("qiskit", qiskit.__version__)
print("qiskit-aer", qiskit_aer.__version__)
PY
```

`qiskit-aer-gpu` 需要可用的 NVIDIA CUDA 环境，并且 CUDA/驱动版本需要与 wheel 兼容。在 macOS 或没有 CUDA 的机器上，GPU 后端通常不可用；此时请安装 `requirements-cpu.txt` 并使用 `--device CPU`。

## 快速开始

下面的运行命令默认从 `baseline/` 目录执行，因此样例数据路径使用 `../data/alpha-test/`。

GPU 版本：

```bash
conda activate qurbo
cd baseline

python baseline_miqp_qaoa.py \
  --input ../data/alpha-test/miqp_sample_A.npz \
  --output solution_A_gpu.npz \
  --iterations 20 \
  --sub-size 12 \
  --shots 512 \
  --device GPU
```

macOS 或无 CUDA 环境可以直接跑 CPU 版本：

```bash
conda activate qurbo
cd baseline

python baseline_miqp_qaoa.py \
  --input ../data/alpha-test/miqp_sample_A.npz \
  --output solution_A_cpu.npz \
  --iterations 20 \
  --sub-size 12 \
  --shots 512 \
  --device CPU
```

v3 版本适合后续测试集和较大样例，仍然保证单次量子调用不超过 30 qubit：

```bash
conda activate qurbo
cd baseline

python baseline_miqp_qaoa_v3.py \
  --input ../data/alpha-test/miqp_sample_B.npz \
  --output solution_B_v3_cpu.npz \
  --iterations 120 \
  --sub-size 20 \
  --shots 512 \
  --top-k 30 \
  --candidate-pool 30 \
  --device CPU
```

运行过程中会输出初始解、每轮迭代的候选解可行性和目标函数值，结束后生成一个 `.npz` 解文件。

## 命令行参数

`baseline/baseline_miqp_qaoa.py` 支持以下参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--input` | 必填 | 输入 MIQP `.npz` 文件路径。 |
| `--output` | `solution_baseline.npz` | 输出解文件路径。 |
| `--iterations` | `20` | 局部搜索迭代轮数。 |
| `--sub-size` | `12` | 每轮送入量子子问题的二元变量数量；代码限制不超过 30。 |
| `--shots` | `512` | Qiskit Aer 每组参数的采样次数。 |
| `--layers` | `1` | QAOA-like circuit 的层数。 |
| `--penalty` | `10.0` | 约束罚项权重。 |
| `--device` | `CPU` | AerSimulator 设备，可选 `CPU` 或 `GPU`。 |
| `--seed` | `42` | 随机种子。 |

`baseline/baseline_miqp_qaoa_v3.py` 额外支持 `--top-k`、`--candidate-pool`、`--random-candidates`、`--init-trials`、`--mixed-penalty-scale`、`--repair-steps`、`--stagnation-limit` 和 `--temperature`。v3 会将 LP 对偶/历史回归得到的连续子问题代理价值反馈到 subQUBO，并使用强耦合聚类变量块、不可行候选修复和精英重启来增强大规模实例的搜索稳定性。

## 输入数据格式

输入文件使用 NumPy `.npz` 格式。每个实例应包含以下字段：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `n` | 标量 | 二元变量 `x` 的维度。 |
| `p` | 标量 | 连续变量 `y` 的维度。 |
| `m1` | 标量 | 混合约束数量。 |
| `m2` | 标量 | 纯二元约束数量。 |
| `Q` | `(n, n)` | 二次目标项矩阵。 |
| `c` | `(n,)` | 二元变量线性目标项。 |
| `h` | `(p,)` | 连续变量线性目标项。 |
| `A` | `(m1, n)` | 混合约束中的二元变量系数。 |
| `G` | `(m1, p)` | 混合约束中的连续变量系数。 |
| `b` | `(m1,)` | 混合约束右端项。 |
| `B` | `(m2, n)` | 纯二元约束系数。 |
| `b_prime` | `(m2,)` | 纯二元约束右端项。 |

可以用下面的命令查看样例实例的字段：

```bash
python - <<'PY'
import numpy as np

path = "../data/alpha-test/miqp_sample_A.npz"
data = np.load(path)

for key in data.files:
    value = data[key]
    print(f"{key:14s} shape={value.shape}, dtype={value.dtype}")
PY
```

也可以打开 `data/alpha-test/读取数据示例.ipynb` 交互式查看数据；从 `baseline/` 目录看，该文件路径是 `../data/alpha-test/读取数据示例.ipynb`。

## 输出解文件

baseline 会将结果保存为 `.npz` 文件，包含：

| 字段 | 含义 |
| --- | --- |
| `x` | 求得的二元变量解，整数数组。 |
| `y` | 固定 `x` 后由线性规划求得的连续变量解。 |
| `objective` | 目标函数值。 |
| `feasible` | 当前解是否满足可行性检查。 |

v2/v3 还会额外保存 `accepted_count`、`evaluated_count` 等统计信息；v3 进一步保存 `restart_count`、`lp_eval_count`、`repaired_candidate_count` 和 `best_trace`，便于复现实验和撰写算法说明。

读取输出示例：

```bash
python - <<'PY'
import numpy as np

sol = np.load("solution_A_gpu.npz")
print("feasible =", bool(sol["feasible"]))
print("objective =", float(sol["objective"]))
print("x =", sol["x"])
print("y =", sol["y"])
PY
```

## Baseline 方法概览

`baseline_miqp_qaoa.py` 的基础流程如下：

1. 读取 MIQP 实例。
2. 随机生成若干初始二元解，并通过 `scipy.optimize.linprog` 求连续变量 `y`。
3. 每轮选择一批二元变量子集，固定其他二元变量。
4. 将局部子问题构造成 subQUBO，并加入简单平方罚项处理约束。
5. 使用 Qiskit Aer 构造并采样一个轻量 QAOA-like circuit。
6. 将采样结果按 QUBO energy 精排，得到候选子解。
7. 回代完整 `x`，再次求解连续变量 `y`，若目标函数更好且可行则接受。
8. 保存当前最好解。

需要注意：这是 baseline 实现，主要用于提供一个可运行的参考流程。约束罚项、变量子集选择、QAOA 参数搜索和接受策略都可以继续改进。

`baseline_miqp_qaoa_v3.py` 在此基础上做了三项主要升级：

1. 固定 `x` 后仍用 `scipy.optimize.linprog` 精确求连续变量 `y`，但把 LP 对偶信息或历史样本回归得到的 recourse value 代理反馈到二元 subQUBO。
2. 用 `Q` 的二次耦合、`A/B` 的共约束关系构造变量图，优先抽取强耦合聚类块作为量子子问题。
3. 对量子采样候选先做有限步可行性修复，再进行 LP 回代评估；搜索停滞后从精英池扰动重启。

## 暴力枚举校验

对于很小的实例，可以使用暴力枚举脚本校验 baseline 结果。脚本会枚举所有二元变量 `x`，并对每个 `x` 求解连续 LP 子问题。

```bash
cd baseline

python bruteforce_check.py \
  --input ../data/alpha-test/miqp_sample_A.npz
```

脚本内置保护：当 `n > 25` 时会拒绝运行，避免指数级枚举耗时过长。

## 开发建议

- 优先从 `baseline/baseline_miqp_qaoa.py` 修改策略，例如子集选择、罚项构造、QAOA 参数搜索或接受准则。
- 若要比较算法效果，建议固定 `--seed` 并记录 `iterations/sub-size/shots/layers/penalty`。
- 若要加入新的样例数据，请保持 `.npz` 字段命名和形状与本 README 一致。
- 生成的解文件、日志文件和本地实验结果建议不要提交到仓库。

## License

本项目使用 MIT License，详见 `LICENSE`。
