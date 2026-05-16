# Algorithm

这里整合 `qurbo-fresh` 中新的混合 MIQP 求解器计算代码，不包含重复数据、实验解文件或报告产物。

## 目录

- `data/instance.py`：读取和校验 `.npz` MIQP 实例。
- `core/`：目标评估、初始解生成、subQUBO 构建、修复器、精英池和 Benders cut 管理。
- `solvers/`：QAOA-like、模拟退火和小规模 QUBO 精确穷举求解器。
- `strategy/`：变量子集选择策略。
- `solver.py`：混合 MIQP 主求解器。
- `run.py`：命令行运行入口。
- `evaluate.py`：统一评估 `.npz` 解文件。

## 运行

从仓库根目录执行：

```bash
python -m algorithm.run \
  --instance data/alpha-test/miqp_sample_A.npz \
  --output solution_A.npz \
  --time-limit 60 \
  --device CPU

python -m algorithm.run \
  --instance data/alpha-test/miqp_sample_B.npz \
  --output solution_B.npz \
  --time-limit 60 \
  --device CPU
```

评估解文件：

```bash
python -m algorithm.evaluate \
  --instance data/alpha-test/miqp_sample_A.npz \
  --sol solution_A.npz
```
