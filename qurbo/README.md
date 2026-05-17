# QURBO MIQP Hybrid v4

本目录是可独立交付的 QURBO 混合整数二次规划（MIQP）求解子项目。最终提交入口为 `miqp_hybrid_v4.py`，输入为 `.npz` 格式的 MIQP 实例，输出为包含二元解、连续解、目标值和运行统计的 `.npz` 文件。

## 1. 环境配置

推荐环境：

- 操作系统：Linux / macOS；GPU 模式需要 NVIDIA CUDA 环境。
- Python：`3.11`。
- 核心依赖：`numpy`、`scipy`、`qiskit==1.4.3`、`qiskit-aer==0.15.1` 或 `qiskit-aer-gpu==0.15.1`。
- CPU 环境配置文件：`requirements-cpu.txt`。
- GPU 环境配置文件：`requirements.txt`。

CPU 标准复现环境：

```bash
conda create -n qurbo python=3.11 -y
conda activate qurbo
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt
```

如评测机器具备可用的 NVIDIA CUDA 与兼容驱动，可以安装 GPU 依赖：

```bash
conda create -n qurbo python=3.11 -y
conda activate qurbo
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

安装后可检查依赖版本：

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

## 2. 子项目目录结构

```text
.
├── miqp_hybrid_v4.py           # 最终提交求解器入口
├── README.md                   # 本运行说明
├── requirements.txt            # GPU 依赖配置，使用 qiskit-aer-gpu
├── requirements-cpu.txt        # CPU 依赖配置，使用 qiskit-aer
└── data/
    └── final-validation/
        ├── miqp_test_1.npz     # 最终验证实例 1
        ├── miqp_test_2.npz     # 最终验证实例 2
        ├── miqp_test_3.npz     # 最终验证实例 3
        ├── miqp_test_4.npz     # 最终验证实例 4
        ├── miqp_test_5.npz     # 最终验证实例 5
        └── 读取数据示例.ipynb   # NPZ 数据读取示例
```

## 3. 一键运行命令

请从本目录执行命令。

CPU 快速验证：

```bash
python miqp_hybrid_v4.py \
  --input data/final-validation/miqp_test_1.npz \
  --output solution_test_1_v4_cpu.npz \
  --iterations 20 \
  --time-limit-seconds 60 \
  --device CPU \
  --seed 42
```

CPU 标准运行：

```bash
python miqp_hybrid_v4.py \
  --input data/final-validation/miqp_test_1.npz \
  --output solution_test_1_v4_cpu.npz \
  --iterations 80 \
  --time-limit-seconds 300 \
  --q-max 18 \
  --qaoa-qubits 18 \
  --device CPU \
  --seed 42
```

批量运行 5 个最终验证实例：

```bash
for i in 1 2 3 4 5; do
  python miqp_hybrid_v4.py \
    --input data/final-validation/miqp_test_${i}.npz \
    --output solution_test_${i}_v4_cpu.npz \
    --iterations 80 \
    --time-limit-seconds 300 \
    --q-max 18 \
    --qaoa-qubits 18 \
    --device CPU \
    --seed 42
done
```

GPU 运行方式：

```bash
python miqp_hybrid_v4.py \
  --input data/final-validation/miqp_test_1.npz \
  --output solution_test_1_v4_gpu.npz \
  --iterations 80 \
  --time-limit-seconds 300 \
  --q-max 18 \
  --qaoa-qubits 18 \
  --device GPU \
  --seed 42
```

运行过程中会打印每轮迭代日志，例如当前目标值、历史最好目标值、是否接受候选解、QAOA 调用次数等。运行结束后会打印 `[DONE]` 汇总信息并保存输出文件。

## 4. 参数说明

常用参数如下：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--input` | 必填 | 输入 MIQP `.npz` 文件路径。 |
| `--output` | `solution_v4.npz` | 输出解文件路径。 |
| `--iterations` | `80` | 最大搜索迭代轮数。 |
| `--time-limit-seconds` | `0.0` | 时间上限，单位秒；`0` 表示不按时间提前停止。 |
| `--q-max` | `18` | 每个局部子问题最多选择的二元变量数，代码限制不超过 `18`。 |
| `--qaoa-qubits` | `18` | QAOA 子问题最大量子比特数，代码限制不超过 `18`。 |
| `--initial-sub-size` | `12` | 初始局部子问题变量数。 |
| `--min-sub-size` | `8` | 局部子问题变量数下限。 |
| `--qaoa-opt-steps` | `20` | QAOA 参数优化步数。 |
| `--qaoa-multistart` | `2` | QAOA 参数多起点次数。 |
| `--shots-small` | `1024` | 小规模 QAOA 子问题采样次数。 |
| `--shots-large` | `512` | 较大 QAOA 子问题采样次数。 |
| `--top-k` | `20` | 每个子问题保留并回代评估的候选数量。 |
| `--device` | `CPU` | AerSimulator 设备，可选 `CPU` 或 `GPU`。 |
| `--seed` | `42` | 随机种子，用于复现实验。 |
| `--cache-size` | `5000` | LP 回代评估缓存大小。 |
| `--elite-size` | `20` | 精英候选池大小。 |

如需查看全部参数：

```bash
python miqp_hybrid_v4.py --help
```

## 5. 输入文件格式

输入文件为 NumPy `.npz` 格式，描述如下 MIQP：

```text
maximize    x^T Q x + c^T x + h^T y
subject to  A x + G y <= b
            B x <= b_prime
            x in {0, 1}^n
            y >= 0
```

必须字段：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `n` | 标量 | 二元变量 `x` 的数量。 |
| `p` | 标量 | 连续非负变量 `y` 的数量。 |
| `m1` | 标量 | 混合约束 `A x + G y <= b` 的数量。 |
| `m2` | 标量 | 纯二元约束 `B x <= b_prime` 的数量。 |
| `Q` | `(n, n)` | 二元变量二次目标项矩阵。 |
| `c` | `(n,)` | 二元变量线性目标项。 |
| `h` | `(p,)` | 连续变量线性目标项。 |
| `A` | `(m1, n)` | 混合约束中二元变量系数。 |
| `G` | `(m1, p)` | 混合约束中连续变量系数。 |
| `b` | `(m1,)` | 混合约束右端项。 |
| `B` | `(m2, n)` | 纯二元约束系数。 |
| `b_prime` | `(m2,)` | 纯二元约束右端项。 |

样例数据还可能包含 `optimal_value`、`x_opt`、`y_opt`、`is_optimal`，用于结果对比；求解器不依赖这些字段。

查看输入字段：

```bash
python - <<'PY'
import numpy as np

data = np.load("data/final-validation/miqp_test_1.npz")
for key in data.files:
    value = data[key]
    print(f"{key:14s} shape={value.shape}, dtype={value.dtype}")
PY
```

## 6. 输出文件格式

求解器输出 `.npz` 文件，主要字段如下：

| 字段 | 含义 |
| --- | --- |
| `x` | 求得的二元变量解。 |
| `y` | 固定 `x` 后通过 `scipy.optimize.linprog` 求得的连续变量解。 |
| `objective` | 当前可行解的目标函数值。 |
| `feasible` | 输出解是否通过可行性检查。 |
| `optimal_value` | 输入文件提供的参考最优值；若输入无该字段则为 `nan`。 |
| `optimality_gap` | 与参考最优值的相对差距；若无法计算则为 `nan`。 |
| `iterations_done` | 实际完成的迭代轮数。 |
| `best_trace` | 每轮历史最好目标值轨迹。 |
| `current_trace` | 每轮当前解目标值轨迹。 |
| `qaoa_calls` | QAOA 子问题调用次数。 |
| `lp_eval_count` | LP 回代评估次数。 |
| `accepted_count` | 被接受的候选解数量。 |
| `restart_count` | 精英池重启次数。 |
| `elapsed_seconds` | 总运行时间。 |
| `max_binary_violation` | 纯二元约束最大违反量。 |
| `max_mixed_violation` | 混合约束最大违反量。 |

读取并展示运行结果：

```bash
python - <<'PY'
import numpy as np

sol = np.load("solution_test_1_v4_cpu.npz")
print("feasible =", bool(sol["feasible"]))
print("objective =", float(sol["objective"]))
print("optimal_value =", float(sol["optimal_value"]))
print("optimality_gap =", float(sol["optimality_gap"]))
print("iterations_done =", int(sol["iterations_done"]))
print("qaoa_calls =", int(sol["qaoa_calls"]))
print("lp_eval_count =", int(sol["lp_eval_count"]))
print("elapsed_seconds =", float(sol["elapsed_seconds"]))
print("x =", sol["x"])
print("y =", sol["y"])
PY
```

## 7. 预期运行时间与硬件要求

最终验证实例规模较大，建议使用 `--time-limit-seconds 300` 到 `1800` 进行评测；同一随机种子下结果可复现，但运行时间会随 CPU、BLAS、Qiskit Aer 后端和 CUDA 环境变化。

硬件建议：

| 场景 | 建议配置 |
| --- | --- |
| 快速功能验证 | 4 核 CPU、8 GB 内存、`--device CPU`。 |
| 标准 CPU 评测 | 8 核或以上 CPU、16 GB 内存、`--time-limit-seconds 300` 以上。 |
| GPU 评测 | NVIDIA GPU、可用 CUDA 驱动、安装 `requirements.txt` 并使用 `--device GPU`。 |

为保证可复现性，建议评审运行时固定 `--seed 42`，并在提交结果中同时保留命令行日志和输出 `.npz` 文件。
