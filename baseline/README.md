# Baseline

本目录下的命令默认从 `baseline/` 目录执行，样例数据位于 `../data/alpha-test/`。

GPU 版本：

```bash
bash run_base.sh
```

macOS 或无 CUDA 环境可以直接跑 CPU 版本：

```bash
bash run_base_cpu.sh
```

v4 版本位于 `baseline_v4/` 包内，默认从仓库根目录按模块运行：

```bash
python -m baseline.baseline_v4.run \
  --instance data/alpha-test/miqp_sample_B.npz \
  --output solution_B_v4_cpu.npz \
  --time-limit 60 \
  --device CPU
```

如果已经在 `baseline/` 目录下，也可以直接使用脚本：

```bash
bash run_base_v4_cpu.sh
```
