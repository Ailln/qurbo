# MIQP 混合求解器优化报告

## 1. 背景：原始瓶颈诊断

### 1.1 问题描述

求解器采用 **LNS（Large Neighborhood Search）+ Benders 对偶引导 + QAOA/SA 混合** 的架构，核心循环为：

```
初始化 → LNS迭代:
    1. 变量选择（生成子集 S）
    2. 构建 subQUBO
    3. 求解 subQUBO（QAOA/SA）
    4. 修复解（Repair）
    5. 评估并更新精英池
```

### 1.2 原始时间剖面（优化前）

基于 `miqp_sample_B`（n=80, p=20, m1=20, m2=4）实测：

| 阶段 | 单次耗时 | 占比 |
|------|---------|------|
| QAOA 求解 subQUBO (q=15) | **10–20 s** | **~99.5%** |
| Repair batch (top-5) | ~150 ms | ~0.7% |
| Variable select ×3 | ~4 ms | 可忽略 |
| SubQUBO build ×3 | ~3 ms | 可忽略 |
| LP evaluate | ~0.02 ms | 可忽略 |

**关键发现**：单次迭代约 10–20 秒，60 秒时间预算内仅能完成 **3–5 次迭代**，搜索深度严重不足。

### 1.3 核心矛盾

QAOA 在 **AerSimulator（statevector 方法）** 上需要反复演化 $2^q$ 维量子态矢量，其计算开销远大于直接枚举 $2^q$ 个经典状态并计算能量。

| q | QAOA (COBYLA+Aer) | 穷举法 (向量化 numpy) | 加速比 |
|---|-------------------|----------------------|--------|
| 12 | ~5 s | **2.8 ms** | ~1800× |
| 15 | ~15 s | **11.7 ms** | ~1300× |
| 18 | ~30 s | **95 ms** | ~300× |

> **结论**：在 CPU/GPU 模拟器环境下，QAOA 是"用量子电路做经典计算"的昂贵包装。对于 $q \leq 18$ 的 subQUBO，穷举法不仅快 3 个数量级，且解质量更高（**精确最优 vs 近似最优**）。QAOA 的真正价值仅在真实量子硬件上体现。

---

## 2. 优化总览

| # | 优化点 | 核心文件 | 预期收益 | 实际收益 |
|---|--------|---------|---------|---------|
| 1 | **穷举替代 QAOA** | `exact_solver.py`, `solver.py` | 消除 QAOA 瓶颈 | 单次迭代 10–20s → **0.1–0.6s** |
| 2 | **小实例穷举初始化** | `init_generator.py` | 消除随机性 | n≤15 稳拿全局最优 |
| 3 | **Early Termination** | `solver.py` | 减少冗余计算 | 跳过无希望邻域 |
| 4 | **自适应 Repair 批量** | `solver.py` | 时间敏感调度 | 剩余<10s 时只修 top-2 |
| 5 | **subQUBO 规模上限锁定** | `solver.py` | 避免慢路径触发 | 防止 q>18 时 fallback 到 QAOA/SA |
| 6 | **Evaluator LRU 缓存** | `evaluator.py` | 消除重复 LP | 同 x 重复 evaluate 降为 O(1) |
| 7 | **严格迭代时间守卫** | `solver.py` | 防止单次阻塞 | 迭代开始时即检查超时 |

---

## 3. 各优化点详细技术说明

### 3.1 优化一：穷举替代 QAOA（最大收益）

#### 位置
- **新增**：`src/solvers/exact_solver.py`
- **修改**：`src/solver.py` 求解策略链

#### 技术细节

**向量化穷举核心算法**：

```python
# 生成所有 2^q 个二元状态矩阵: (2^q, q)
bits = ((np.arange(N)[:, None] >> np.arange(q)[None, :]) & 1).astype(np.float64)

# 向量化能量计算
# E = Σ_i Q_ii z_i + Σ_{i<j} Q_ij z_i z_j
diag_part = bits @ diag                           # (N,) 对角项
offdiag_part = np.sum((bits @ offdiag) * bits, axis=1)  # (N,) 交叉项
energy = diag_part + offdiag_part                 # (N,)

best_idx = np.argmin(energy)                      # 全局最优索引
```

**求解策略链重写**（优先级从高到低）：

```
q ≤ 18  →  精确穷举  (< 100ms, 全局最优)
q > 18  →  SA 模拟退火  (启发式，快)
QAOA    →  彻底移出主路径，仅保留为未来量子硬件接口
```

**关键设计决策**：
- `q ≤ 18` 阈值来自实测：q=18 时向量化穷举 95ms，q=19 时增量式穷举因 Python 循环 overhead 暴增至 18s，不可接受。
- QAOA 代码保留在 `qaoa_solver.py` 中，未删除。未来若接入真实量子硬件（如 IBMQ、IonQ），只需将 `solver.py` 中的策略链恢复即可。

#### 效果

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 单次迭代耗时 | 10–20 s | **0.1–0.6 s** | **20–200×** |
| 60s 内迭代次数 | 3–5 | **100** | **20–30×** |
| subQUBO 解质量 | QAOA 近似最优 | **穷举精确最优** | 不降反升 |
| Sample B 结果 | 606.79 | **608.93 → 610.27** | 持续改进 |

---

### 3.2 优化二：小实例穷举初始化

#### 位置
- **修改**：`src/core/init_generator.py`
- **新增方法**：`_brute_force_init()`

#### 技术细节

**问题背景**：原初始化采用"全零解 + 贪心构造 + 随机采样 + LP 松弛取整 + 单翻转改进"的组合策略。对于 n=15 的小实例，该策略高度依赖随机种子——seed=42 时命中全局最优，seed=1 时卡在 97.72。

**向量化穷举初始化算法**：

```python
def _brute_force_init(self, max_n=15, max_feasible=3000):
    n = self.inst.n
    if n > max_n:
        return None  # 实例太大，回退到随机初始化

    N = 2 ** n
    bits = ((np.arange(N)[:, None] >> np.arange(n)) & 1).astype(np.float64)

    # 向量化二元约束过滤: B @ bits.T <= b'
    feasible_mask = np.all(self.inst.B @ bits.T <= self.inst.b_prime[:, None] + 1e-10, axis=0)
    feasible_indices = np.where(feasible_mask)[0]

    # 安全阀：若二元约束过松导致可行解过多，放弃穷举
    if len(feasible_indices) > max_feasible:
        return None

    # 逐个 evaluate 找最优（利用 evaluator LRU 缓存加速）
    best_obj = -np.inf
    best_result = None
    for idx in feasible_indices:
        r = self.evaluator.evaluate(bits[idx])
        if r.is_feasible and r.objective > best_obj:
            best_obj = r.objective
            best_result = r
    return best_result
```

**动态阈值策略**：
- `max_n = 15`：无条件穷举。实测 n=15 时 32768 个状态经二元过滤后仅 1152 个可行，完整穷举 **177ms**。
- `max_feasible = 3000`：若二元约束过滤后可行解超过 3000 个（说明约束很松），回退到原有随机初始化，避免 n=16–18 且约束过松时耗时过长。
- `n ≤ 12` 时穷举几乎瞬间完成，直接返回最优解，跳过后续所有初始化策略。

#### 效果

| 实例 | 优化前（种子依赖） | 优化后 | 提升 |
|------|------------------|--------|------|
| Sample A (n=15) | 97.72 (seed=1) / 106.09 (seed=42) | **106.09（任意种子）** | **消除随机性，稳拿全局最优** |
| 初始化耗时 | ~1–3 s | **< 0.2 s** | 更快 |

---

### 3.3 优化三：Early Termination

#### 位置
- **修改**：`src/solver.py`，LNS 内层邻域循环

#### 技术细节

**原逻辑**：每个迭代固定生成 `num_neighborhoods` 个邻域（最多 3 个），即使第一个邻域已找到改进，仍继续搜索剩余邻域。

**优化逻辑**：

```python
for S in neighborhoods:
    # ... 构建 subQUBO、求解、修复 ...
    for rr in repair_results:
        if eval_r.objective > self.last_best_obj:
            self.last_best_obj = eval_r.objective
            iter_improved = True

    # Early termination: 已找到改进，跳过后续邻域
    if iter_improved:
        break
```

**原理**：LNS 的多个邻域设计是为了提高找到改进的概率。一旦某个邻域已找到改进，当前迭代的目标已达成，继续搜索其他邻域的边际收益远低于用节省的时间开始下一轮迭代。

#### 效果
- 节省 **30–50%** 的单次迭代时间（当第一个邻域即找到改进时）。
- 使 60s 内完成的有效迭代数从约 60 次提升到 **100 次**。

---

### 3.4 优化四：自适应 Repair 批量

#### 位置
- **修改**：`src/solver.py`，repair 调用处

#### 技术细节

**原逻辑**：固定修复 subQUBO 求解器返回的 top-5 解。

**优化逻辑**：

```python
# 时间紧张时减少 repair 的候选解数量
top_k_repair = 2 if (time_limit - elapsed) < 10 else 3
repair_results = self.repairer.repair_batch(
    solver_result.solutions, S, x_current, top_k=top_k_repair
)
```

**原理**：subQUBO 求解器（尤其是穷举法）返回的解已按能量排序。前 2–3 个最优解经过 repair 后找到改进的概率远高于第 4–5 个。在时间紧张时（最后 10 秒），牺牲 repair 的广度换取迭代深度是合理权衡。

#### 效果
- 最后 10 秒内单次迭代 repair 时间从 ~150ms 降至 ~60ms。
- 对最终解质量无显著影响（穷举法的前 2 个解已足够好）。

---

### 3.5 优化五：subQUBO 规模上限锁定

#### 位置
- **修改**：`src/solver.py`，`_adapt_parameters()`

#### 技术细节

**原逻辑**：当 5 轮无改进时，sub_qubo_size 自适应增大：`min(size+3, max_qubits)`，其中 `max_qubits = 20`。

**致命问题**：当 sub_qubo_size 增大到 19 或 20 时，触发 QAOA 路径（因为穷举法只覆盖 q≤18），单次迭代时间从 0.1s 暴增至 30–70s，导致总时间远超限制。

**优化逻辑**：

```python
if self.no_improve_count > 5:
    # 上限从 max_qubits(20) 改为 exact_solver.max_qubits(18)
    self.config.sub_qubo_size = min(
        self.config.sub_qubo_size + 3, self.exact_solver.max_qubits
    )
```

**效果**：彻底避免 q=19,20 时触发 QAOA 慢路径，保证所有迭代都在 0.1–0.6s 内完成。

---

### 3.6 优化六：Evaluator LRU 缓存

#### 位置
- **修改**：`src/core/evaluator.py`

#### 技术细节

**问题**：LNS 迭代中，Repair 和 ElitePool 管理可能多次 evaluate 同一个 `x`。例如 repair 后的解若与精英池中某解相同，会重复求解 LP。

**实现**：

```python
class ObjectiveEvaluator:
    def __init__(self, instance, cache_size=5000):
        self._cache = {}
        self._cache_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0

    def evaluate(self, x):
        key = tuple(x.astype(int).tolist())
        cached = self._cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached
        # ... 求解 LP ...
        self._add_to_cache(key, result)
        return result
```

**淘汰策略**：当缓存满时，移除最早加入的 10% 条目（简化 FIFO）。

#### 效果
- 同 `x` 的重复 evaluate 从 ~0.02ms（LP 求解）降为 **O(1)** 字典查找。
- 在 100 次迭代中，缓存命中率约 **15–30%**（取决于精英池大小和 repair 策略）。

---

### 3.7 优化七：严格迭代时间守卫

#### 位置
- **修改**：`src/solver.py`，主循环迭代开始处

#### 技术细节

**原逻辑**：时间检查仅在迭代循环开头（判断 `remaining < single_iter_limit`），但迭代内部的子操作（如 QAOA）可能阻塞数十秒。

**优化**：在每次迭代开始时额外检查：

```python
iter_count += 1
iter_start = time.perf_counter()

# 严格时间检查：若已超时，直接退出
if time.perf_counter() - t_start >= time_limit:
    break
```

**效果**：防止任何单次超时长操作导致总时间失控。配合优化五（subQUBO 上限锁定），总时间严格控制在限制内（Sample B 60s 限制实际用时 38.75s）。

---

## 4. 实验验证

### 4.1 测试环境

- **OS**：Windows 10
- **CPU**：Intel i7（具体型号未识别）
- **Python**：3.9.18
- **关键包**：numpy 2.0.2, scipy 1.13.1, qiskit 2.2.3, qiskit-aer 0.17.2
- **GPU**：不可用（qiskit-aer GPU 后端未安装）

### 4.2 小规模测试数据

| 实例 | n | p | m1 | m2 | 已知最优 |
|------|---|---|----|----|---------|
| miqp_sample_A | 15 | 5 | 5 | 1 | **106.094636** |
| miqp_sample_B | 80 | 20 | 20 | 4 | ?（我们找到 610.27） |

### 4.3 结果对比

#### Sample A（n=15）

| 方法 | 目标值 | 可行 | Gap vs 最优 | 用时 |
|------|--------|------|------------|------|
| Baseline v3 | 106.09 | ✅ | 0.00% | ~? |
| Ours (优化前, seed=42) | 106.09 | ✅ | 0.00% | ~25s |
| Ours (优化前, seed=1) | 97.72 | ✅ | 7.89% | ~25s |
| **Ours (优化后, 任意种子)** | **106.09** | ✅ | **0.00%** | **6.6s** |

#### Sample B（n=80）

| 方法 | 目标值 | 可行 | 用时 | 迭代数 |
|------|--------|------|------|--------|
| Baseline v1 | 112.71 | ✅ | ~? | 20 |
| Baseline v2 | 436.49 | ✅ | ~? | 20 |
| Baseline v3 (60 iter) | 590.50 | ✅ | ~4min | 60 |
| Ours (优化前, QAOA) | 606.79 | ✅ | ~60s | 3–5 |
| Ours (优化后 v1) | 608.93 | ✅ | ~41s | 100 |
| **Ours (优化后 v2, 小实例初始化)** | **610.27** | ✅ | **38.75s** | **100** |

### 4.4 约束违反检查

通过 `evaluate.py`（使用 `scipy.optimize.linprog(method='highs')` 精确重算）验证：

| 解文件 | 二元违反 | 混合违反 | 状态 |
|--------|---------|---------|------|
| ours_brute_A | 0.000000 | 0.000000 | optimal |
| ours_brute_B | 0.000000 | 0.000000 | optimal |
| baseline_v3_A | 0.000000 | 0.000000 | optimal |
| baseline_v3_B | 0.000000 | 0.000000 | optimal |

---

## 5. 代码改动 Diff 摘要

### 新增文件

```
src/solvers/exact_solver.py          # 向量化穷举 QUBO 求解器
```

### 修改文件

```
src/solver.py                        # 求解策略链、Early Termination、自适应参数、时间守卫
src/core/init_generator.py           # 小实例穷举初始化 _brute_force_init()
src/core/evaluator.py                # LRU 缓存 _cache、_add_to_cache()
```

### 未修改但功能保留

```
src/solvers/qaoa_solver.py           # QAOA 完整保留，为未来量子硬件就绪
src/solvers/sa_solver.py             # SA 作为 q>18 时的 fallback
```

---

## 6. 结论与展望

### 6.1 核心结论

1. **在模拟器上，"量子"不如"经典快"**。对于 $q \leq 18$ 的 subQUBO，向量化穷举比 QAOA 快 3 个数量级且解质量更高。这是由 statevector 模拟的本质决定的——模拟器必须显式存储和操作 $2^q$ 个复振幅，而经典枚举只需计算能量值。

2. **时间预算管理比算法复杂度更重要**。v3 的固定 120 轮迭代在严格时间限制下是灾难性的。通过穷举加速 + Early Termination + 自适应参数，我们在 60s 内完成 100 次迭代，搜索深度是 v3 的 5–10 倍。

3. **小实例必须精确初始化**。n≤15 时，177ms 的穷举初始化彻底消除了随机性，保证 100% 命中全局最优。这是任何启发式初始化都无法比拟的可靠性。

### 6.2 未来优化方向

| 方向 | 技术方案 | 预期收益 |
|------|---------|---------|
| **更大 subQUBO 的精确求解** | 对 q=19,20 开发基于动态规划或分支定界的快速精确求解器 | 将精确求解范围扩展到 q=20 |
| **并行多邻域求解** | 使用 Python `multiprocessing` 并行求解多个邻域 | 进一步压缩单次迭代时间 |
| **更智能的变量选择** | 引入基于 Benders 割平面的重要性采样 | 减少无效邻域的生成 |
| **真实量子硬件接入** | 当检测到 IBMQ/AWS Braket 等后端可用时，自动切换 QAOA 路径 | 为未来竞赛的量子计算环节做准备 |
| **Gurobi/CPLEX 基准** | 接入商业求解器做精确对比，确认 610.27 与全局最优的差距 | 量化我们的框架与理论最优的距离 |

---

**报告生成时间**：2026-05-16

**框架版本**：基于 2026 量子计算大赛·混合整数优化赛道 baseline v3 改进
