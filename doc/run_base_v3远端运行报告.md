# run_base_v3.sh 远端运行报告

## 1. 运行任务

目标：在远端沐曦 GPU 服务器的 `qiskit` Docker 容器中运行项目仓库 `qurbo` 的：

```bash
baseline/run_base_v3.sh
```

并计算样例 B 的 Optimality Gap。

本机归档的远端输出文件：

```text
/Users/hhy/AI/黑客松/项目资料/黑客松/量子黑客松/solution_B_v3_gpu_remote.npz
```

## 2. 远端环境

登录链路：

```bash
ssh hanhongying@jump.zs.shaipower.online
docker exec -it qiskit bash
```

实际登录后进入：

```text
infra@qiskit-metax-gpu-20
```

Docker 容器：

```text
qiskit
```

容器内 Python 环境：

```text
Python 3.12.3
python path: /opt/qiskit_env/bin/python
numpy 2.4.4
scipy 1.17.1
qiskit 2.4.0
qiskit_aer 0.17.2
```

说明：容器内没有 `git`，且直接从 GitHub 下载 zip 时连接卡住。因此采用本机 clone 后打包，通过 `scp` 传到远端宿主机，再 `docker cp` 进入容器。

远端容器代码路径：

```text
/root/qurbo
```

## 3. 运行命令

容器内执行：

```bash
cd /root/qurbo/baseline
bash run_base_v3.sh
```

脚本实际运行命令：

```bash
python baseline_miqp_qaoa_v3.py \
  --input ../data/alpha-test/miqp_sample_B.npz \
  --output solution_B_v3_gpu.npz \
  --iterations 120 \
  --sub-size 20 \
  --shots 512 \
  --top-k 30 \
  --candidate-pool 30 \
  --device GPU
```

实例规模：

```text
n=80, p=20, m1=20, m2=4
```

运行配置：

```text
sub_size=20
iterations=120
shots=512
layers=1
device=GPU
top_k=30
candidate_pool=30
random_candidates=10
init_trials=300
```

## 4. 运行过程摘要

初始解：

```text
[INIT] feasible=True, objective=532.052446
```

关键提升节点：

| 迭代 | best objective |
|---:|---:|
| INIT | 532.052446 |
| 1 | 542.610804 |
| 6 | 544.056471 |
| 8 | 546.208748 |
| 14 | 552.518827 |
| 15 | 554.662761 |
| 20 | 563.212347 |
| 30 | 563.858432 |
| 36 | 575.885118 |
| 40 | 579.967062 |
| 50 | 581.788292 |
| 63 | 584.593993 |
| 70 | 604.187024 |
| 120 | 604.187024 |

第 70 轮达到本次运行最佳值，之后未继续刷新。

结束日志：

```text
[DONE] saved to solution_B_v3_gpu.npz
[DONE] best objective = 604.187024
[DONE] feasible = True
[DONE] accepted_count = 19
[DONE] restart_count = 2
[DONE] evaluated_count = 5136
[DONE] feasible_candidate_count = 948
[DONE] repaired_candidate_count = 4188
[DONE] lp_eval_count = 3054
```

## 5. 独立复算结果

使用输出文件和原始 `miqp_sample_B.npz` 在远端和本机均复算，结果一致。

| 指标 | 数值 |
|---|---:|
| 输出文件 | `solution_B_v3_gpu.npz` |
| 本机归档 | `solution_B_v3_gpu_remote.npz` |
| stored objective | 604.1870238736836 |
| recomputed objective | 604.1870238736836 |
| sample_B optimal_value | 610.2666386047233 |
| feasible flag | True |
| max binary violation | -0.8901909789010318 |
| max mixed violation | 1.8740564655672642e-13 |
| min y | 0.0 |
| number of selected x variables | 42 |
| accepted_count | 19 |
| restart_count | 2 |
| evaluated_count | 5136 |
| feasible_candidate_count | 948 |
| repaired_candidate_count | 4188 |
| lp_eval_count | 3054 |

约束可行性：

- `Bx <= b_prime` 满足，最大 violation 为负数，说明有安全余量；
- `Ax + Gy <= b` 满足，最大 violation 约 `1.87e-13`，属于数值误差；
- `y >= 0` 满足。

## 6. Optimality Gap

样例 B 是最大化问题，使用：

$$
\text{Gap}=\frac{F^*-F_{\text{alg}}}{|F^*|}.
$$

其中：

$$
F^*=610.2666386047233,
$$

$$
F_{\text{alg}}=604.1870238736836.
$$

因此：

$$
\text{Gap}=0.0099622269.
$$

百分比：

$$
\text{Gap}=0.9962226913\%.
$$

结论：`run_base_v3.sh` 在 `miqp_sample_B.npz` 上得到 **约 1.00% Optimality Gap**，显著优于赛题说明中的官方 baseline 平均 gap 约 15%。

## 7. 运行结果解读

### 7.1 正面结果

1. **可行性可靠**  
   最终解严格满足二元约束和混合约束。

2. **目标值质量较好**  
   对 80 个二元变量的样例 B，gap 约 1%，说明 v3 baseline 在样例分布上已经具备较强搜索能力。

3. **修复机制发挥作用**  
   总候选评价 5136 次，其中 repaired candidates 为 4188，说明大量 QAOA/classical 候选需要修复；repair 是保证可行性的关键模块。

4. **重启机制有效避免完全停滞**  
   运行中发生 2 次 restart。第 70 轮后达到最佳，后续重启未刷新，但机制本身正常工作。

5. **GPU Qiskit Aer 可运行**  
   `device=GPU` 没有报错，说明远端容器中的 Aer GPU 后端可用于该脚本。

### 7.2 暴露的问题

1. **最佳候选来源多为 classical two_flip**  
   日志中的 `source=two_flip` 频繁出现，说明最终改善主要由 QAOA 候选池叠加的经典 two-flip 候选贡献。Paper 中不能夸大“纯 QAOA 求得最优”的说法，应强调量子-经典候选池混合。

2. **第 70 轮后停滞明显**  
   best objective 在第 70 轮到达 604.187024，后 50 轮未刷新。说明 120 轮配置中后段边际收益低，最终测试时应加入 time budget 和 early stopping。

3. **sub_size=20 与 v4 安全方案不完全一致**  
   赛题限制是 30 qubit，因此 20 合规；但 v4 建议 18 以留安全余量。若追求严格执行 v4，可将脚本参数改为 `--sub-size 18`。

4. **当前代码仍使用 active squared penalty**  
   v4 理想方案是 hard filter + repair + optional Lagrange price。当前代码对 active inequalities 加平方罚项，属于工程近似。可以保留，但报告中应说明它只作为局部 QUBO guide，不是 slack-free hard constraint encoding。

5. **缺少 wall-clock 控制**  
   当前 `run_base_v3.sh` 固定 120 轮。如果最终测试时间有限，应增加 `--time-limit-seconds` 或外部 timeout。

## 8. 与 v4 方案的符合程度

| v4 要求 | 当前 run_base_v3.sh | 符合度 |
|---|---|---|
| 原问题最大化 | 是 | 高 |
| 连续变量由 LP 求解 | 是 | 高 |
| subQUBO 量子局部搜索 | 是 | 高 |
| 不使用 slack qubits | 是 | 高 |
| hard filter + repair | 有 repair，但仍有 active penalty | 中 |
| dual-price surrogate | 有 dual + regression recourse model | 中高 |
| sub_size <= 18 | 当前为 20 | 中 |
| 自适应 QAOA depth | 固定 layers=1 | 中 |
| per-test time budget | 无 | 低 |
| brute-force sample_A sanity check | 脚本未集成 | 低 |

## 9. 建议后续改动

优先级从高到低：

1. **增加验证脚本**

   自动输出：

   - objective；
   - max binary violation；
   - max mixed violation；
   - min y；
   - sample optimality gap；
   - selected x count。

2. **增加 wall-clock time guard**

   最终测试集应按时间停止，而不是固定迭代轮数。

3. **将 `run_base_v3.sh` 改为批量脚本**

   自动遍历：

   ```text
   miqp_test_1.npz ... miqp_test_5.npz
   ```

   并生成 CSV 汇总。

4. **增加 `--sub-size 18` 的 v4 脚本**

   保留当前 20 qubit 版本，同时新建：

   ```text
   run_base_v4_safe.sh
   ```

5. **增加 sample_A brute-force sanity check**

   在正式运行前自动跑：

   ```bash
   python bruteforce_check.py --input ../data/alpha-test/miqp_sample_A.npz
   ```

6. **Paper 中准确表述量子贡献**

   建议表述为：

   > QAOA-like sampler generates a diversified candidate pool for local binary neighborhoods; classical repair and LP recourse evaluation certify feasibility and objective quality.

## 10. 结论

本次远端运行成功，`run_base_v3.sh` 在 `miqp_sample_B.npz` 上得到：

```text
objective = 604.1870238736836
feasible = True
optimality gap = 0.9962226913%
```

这说明项目 baseline 已经具备较好的样例表现，并且远端 GPU 环境可以运行 Qiskit Aer。当前版本适合作为最终比赛代码的基础，但还需要补充时间控制、验证脚本、批量测试脚本，以及与 v4 方案更一致的 safe 参数版本。
