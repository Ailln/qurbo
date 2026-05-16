# Baseline v3 vs Our Framework：数学形式对比与能力边界分析

## 1. 问题统一形式化

两种方法求解同一类混合整数二次规划（MIQP）：

$$
\begin{aligned}
\max_{x,y} \quad & f(x,y) = x^T Q x + c^T x + h^T y \\
\text{s.t.} \quad & A x + G y \leq b \quad &(\text{mixed constraints, } m_1 \text{ rows}) \\
                 & B x \leq b' \quad &(\text{binary-only constraints, } m_2 \text{ rows}) \\
                 & x \in \{0,1\}^n, \; y \geq 0
\end{aligned}
$$

对于固定的二元决策 $x$，连续子问题为线性规划：

$$
\phi(x) = \max_{y \geq 0} \{h^T y \mid G y \leq b - A x\}
$$

其对偶问题给出**影子价格**（shadow prices）$u^* \in \mathbb{R}^{m_1}_+$，满足：

$$
\phi(x) = u^{*T}(b - A x), \quad G^T u^* \geq h, \; u^* \geq 0
$$

---

## 2. Baseline v3 的数学模型

### 2.1 RecourseModel：对偶线性项 + Ridge 回归

v3 的核心创新是将连续子问题的价值显式建模为关于 $x$ 的仿射函数。

**对偶线性估计**（Dual Linear）：

$$
\hat{d}_i^{(t)} = -(A^T u^*)_i \quad \text{(从 LP 对偶价格提取)}
$$

采用指数平滑更新：

$$
d_i^{(t)} = 0.8 \cdot d_i^{(t-1)} + 0.2 \cdot \hat{d}_i^{(t)}
$$

**回归线性估计**（Regression Linear）：

维护最近最多 300 个样本 $(x^{(s)}, \phi(x^{(s)}))$，求解岭回归：

$$
\beta^* = \arg\min_\beta \sum_s \left(\phi(x^{(s)}) - \beta^T x^{(s)}\right)^2 + \lambda_{\text{ridge}} \|\beta\|_2^2
$$

其中 $\lambda_{\text{ridge}}$ 自适应为：

$$
\lambda_{\text{ridge}} = \lambda_0 \cdot \left(\frac{\text{tr}(X^T X)}{n} + 1\right)
$$

**综合 Recourse 线性项**：

$$
r_i = \begin{cases}
0.7 \cdot d_i + 0.3 \cdot \beta_i^*, & \text{if } \|\beta^*\| > 10^{-12} \\
d_i, & \text{otherwise}
\end{cases}
$$

> **数学本质**：v3 将 Benders 分解中的对偶割平面近似为**参数化线性函数**，并通过 Ridge 回归从历史轨迹中学习连续变量的"机会成本"。

---

### 2.2 Surrogate SubQUBO 构建

对于当前解 $x$ 和选定的子集 $S \subseteq \{1,\dots,n\}$，令 $\bar{S}$ 为补集，定义有效线性系数：

$$
c^{\text{eff}} = c + r \quad \text{(原始系数 + recourse 修正)}
$$

固定 $\bar{S}$ 中的变量，子问题在 $S$ 上的目标为：

$$
F_S(z) = z^T Q_{SS} z + (c^{\text{eff}}_S + 2 Q_{S\bar{S}} x_{\bar{S}})^T z + \text{const}
$$

转化为最小化 QUBO（标准形式 $E(z) = z^T \tilde{Q} z + \tilde{c}^T z$）：

$$
\tilde{Q}_{ii} = -Q_{SS,ii}, \quad \tilde{Q}_{ij} = -2 Q_{SS,ij} \; (i < j)
$$

$$
\tilde{c}_i = -(c^{\text{eff}}_{S_i} + 2 \sum_{j \in \bar{S}} Q_{S_i, j} x_j)
$$

**活跃约束惩罚**（Active Inequality Penalties）：

仅对接近活跃的约束（松弛量 $\leq \text{active\_margin} = 1.0$）添加二次惩罚：

$$
\text{对于约束 } k: \quad \sum_{j \in S} B_{kj} z_j + \sum_{j \in \bar{S}} B_{kj} x_j \leq b'_k
$$

若当前松弛 $s_k = b'_k - (B x)_k \leq 1.0$，则展开平方惩罚：

$$
\lambda \cdot \max(0, \sum_{j \in S} B_{kj} z_j - (b'_k - \sum_{j \in \bar{S}} B_{kj} x_j))^2
$$

该二次项被展开为对角项和交叉项加入 QUBO。

> **关键设计**：v3 使用"活跃约束"而非全部约束加罚，显著减少了惩罚项对 QUBO 结构的扭曲。

---

### 2.3 交互图与变量选择

**交互图**（Interaction Graph）：

$$
W = 0.65 \cdot \frac{|Q|}{\max|Q|} + 0.20 \cdot \frac{|A^T A|}{\max|A^T A|} + 0.15 \cdot \frac{|B^T B|}{\max|B^T B|}
$$

**变量评分**：

$$
\text{score}_i = 0.42 \cdot g_i + 0.25 \cdot c_i + 0.23 \cdot p_i + 0.10 \cdot s_i
$$

其中：
- $g_i = |2(Qx)_i + c_i + r_i|$：梯度（翻转增益）
- $c_i = \sum_j |Q_{ij}|$：耦合度
- $p_i$：约束压力（活跃/违反约束中该变量系数的累积）
- $s_i$：历史成功翻转次数的衰减累积

**贪心聚类选择**：
1. 按 score 选 top-$4|S|$ 变量构成候选池
2. 以概率权重选中心点
3. 每次加入与已选集合交互强度最大的变量：
   $$
   j^* = \arg\max_{j \notin S} \left[0.65 \cdot \max_{i \in S} W_{ij} + 0.35 \cdot \text{score}_j\right]
   $$
4. 每 5 轮随机替换 $|S|/5$ 个尾部变量

---

### 2.4 QAOA-like 求解：固定参数网格

v3 的"QAOA"实际上是**固定参数网格采样**，而非真正的参数优化：

$$
\Gamma = \{0.15, 0.45, 0.9, 1.4, 2.0\}, \quad \mathcal{B} = \{0.15, 0.4, 0.75, 1.1\}
$$

对每一对 $(\gamma, \beta) \in \Gamma \times \mathcal{B}$，构建电路：

$$
|\psi(\gamma, \beta)\rangle = e^{-i\beta H_M} e^{-i\gamma H_C} |+\rangle^{\otimes k}
$$

其中：
- $H_C = \sum_i l_i Z_i + \sum_{i<j} q_{ij} Z_i Z_j$（通过 RZ + RZZ 门实现）
- $H_M = \sum_i X_i$（通过 RX 门实现）

**每个参数组合采样 512 shots**，总计 $5 \times 4 \times 512 = 10240$ 次测量。

> **数学本质**：v3 的 QAOA 层是**参数扫描式启发采样**，没有利用 QAOA 的参数可优化性。理论上这是一个 $|\Gamma| \times |\mathcal{B}|$ 的离散网格搜索。

---

### 2.5 解修复（Repair Candidate）

贪心修复策略，最多 30 步：

**Step 1**：若存在二元约束违反 $Bx > b'$，计算每个活跃变量（当前为 1）的：

$$
\text{release}_i = \sum_{k: \text{active}} \max(0, B_{ki}) + 0.5 \sum_{k: \text{tight}} \max(0, A_{ki})
$$

$$
\text{loss}_i = |2(Qx)_i + c_i + r_i|
$$

**翻转决策**：

$$
i^* = \arg\max_i \frac{\text{release}_i}{\text{loss}_i + 10^{-6}}
$$

若没有任何 release > 0，则选择 loss 最小的变量翻转。

> **数学本质**：这是一个带权重的贪心松弛算法，目标是在最小化目标损失的前提下消除约束违反。

---

### 2.6 接受准则与重启机制

**Metropolis-Hastings 式接受**：

设 $\Delta = f(x_{\text{cand}}) - f(x_{\text{current}})$：

$$
P(\text{accept}) = \begin{cases}
1, & \Delta > 0 \\
\exp\left(\frac{\Delta}{T \cdot \max(1, |f(x_{\text{current}})|)}\right), & \Delta < -10^{-9} \\
0.2, & |\Delta| \leq 10^{-9} \text{ 且 } x_{\text{cand}} \neq x_{\text{current}}
\end{cases}
$$

其中 $T = 0.01$ 为固定温度。

**精英池与重启**：
- 精英池容量：8 个解
- 停滞阈值：25 轮无改进
- 重启：75% 概率从精英池扰动（翻转 $n/20$ 位）后修复，25% 概率完全随机

---

## 3. Our Framework 的数学模型

### 3.1 Benders 对偶引导

我们采用**纯 Benders 对偶信息**，不引入回归近似：

$$
l^{\text{cont}} = -A^T u^*
$$

其中 $u^*$ 直接从 `scipy.optimize.linprog(method='highs')` 的 `ineqlin.marginals` 提取。

**与 v3 的区别**：
- v3 使用 $r = 0.7 d + 0.3 \beta^*$（平滑 + 回归混合）
- 我们使用 $l^{\text{cont}} = -A^T u^*$（精确对偶价格，无平滑滞后）

> **数学含义**：$l^{\text{cont}}_i > 0$ 意味着增加 $x_i$ 会放松混合约束，从而提升连续子问题的最优值。这直接对应 Benders 分解中的**对偶割平面斜率**。

---

### 3.2 SubQUBO 构建与 Dual Rescaling（创新 C2）

**基础 SubQUBO**（与 v3 等价）：

对于子集 $S$，固定 $x_{\bar{S}}$：

$$
E_S(z) = z^T (-Q_{SS}) z + (-c_S - l^{\text{cont}}_S - 2 Q_{S\bar{S}} x_{\bar{S}})^T z
$$

约束惩罚：对所有二元约束统一添加二次惩罚，惩罚权重：

$$
\lambda_B = 2 \cdot \max\left(\max_i |\tilde{Q}_{ii}|, \; 1.0\right)
$$

**Dual Rescaling（创新）**：

计算每个变量对对偶约束的敏感度：

$$
\text{sens}_i = |(A_{:,i})^T u^*|, \quad i \in S
$$

构建缩放因子：

$$
\omega_i = 1 + \eta \cdot \frac{\text{sens}_i}{\max_j \text{sens}_j}
$$

重标度二次项（保留线性项不变）：

$$
\hat{Q}_{ij} = Q_{ij} \cdot \sqrt{\omega_i \omega_j}, \quad \forall i,j \in S
$$

重构建 QUBO：

$$
\tilde{Q}_{ii}^{\text{rescaled}} = -\hat{Q}_{ii}, \quad \tilde{Q}_{ij}^{\text{rescaled}} = -2\hat{Q}_{ij}
$$

> **数学直觉**：对偶敏感度高的变量在连续子问题中"更重要"，通过放大其与其他变量的二次耦合，引导 QAOA/SA 探索这些变量的协同效应。

---

### 3.3 多邻域变量选择

**核心评分**：

$$
\text{score}_i = \alpha_1 \cdot \underbrace{|Q_{ii} + c_i + l^{\text{cont}}_i + 2 \sum_{j \neq i} Q_{ij} x_j|}_{\text{FlipGain}(i)} + \alpha_2 \cdot \underbrace{4 p_i (1 - p_i)}_{\text{Uncertainty}(i)}
$$

其中 $p_i$ 是精英池中变量 $i$ 取 1 的频率。

**多邻域生成策略**：

| 邻域 | 构造方式 | 目的 |
|------|---------|------|
| $N_1$ | Top $\lceil |S|/2 \rceil$ 按 score + 按耦合强度补全 | 全局最优方向 |
| $N_2$ | 提高 Uncertainty 权重至 70%，与 $N_1$ 重叠 $<30\%$ | 探索不确定区域 |
| $N_3$ | 完全随机子集 | 多样化保证 |

> **与 v3 的区别**：v3 每次只生成一个邻域（贪心聚类），我们同时生成 3 个互补邻域，覆盖" exploitation + exploration + randomization "三个维度。

---

### 3.4 自适应 QAOA：真实参数优化

我们的 QAOA 不是固定网格，而是**真正的变分优化**：

**电路 ansatz**（$p$ 层）：

$$
|\psi(\boldsymbol{\gamma}, \boldsymbol{\beta})\rangle = \prod_{l=1}^p e^{-i\beta_l H_M} e^{-i\gamma_l H_C} |\psi_0\rangle
$$

**初始态**（Warm-start）：

$$
|\psi_0\rangle = \bigotimes_{i=1}^k R_y(\theta_i) |0\rangle, \quad \theta_i = 2 \arcsin\sqrt{p_i}
$$

其中 $p_i$ 来自精英池频率。若未提供 warm-start，则退化为 $|+\rangle^{\otimes k}$（H 门）。

**参数优化**：

$$
(\boldsymbol{\gamma}^*, \boldsymbol{\beta}^*) = \arg\min_{\boldsymbol{\gamma}, \boldsymbol{\beta}} \; \mathbb{E}_{z \sim |\psi\rangle}[E(z)]
$$

通过 `scipy.optimize.minimize(method='COBYLA')` 迭代优化，支持 multi-start（默认 2 次）避免局部极小。

**最终采样**：用最优参数跑 $1024$ shots，提取 top-20 解。

> **数学本质**：这是标准的 Variational Quantum Eigensolver (VQE) 框架，将 QAOA 参数视为经典优化变量，通过测量期望值梯度（或零阶优化）寻找最优参数。v3 的固定网格是其退化形式（$|\Gamma| \times |\mathcal{B}|$ 个离散点）。

---

### 3.5 两阶段解修复

**Stage 1：修复二元约束**（$Bx \leq b'$）

贪心循环：
1. 找到违反最严重的约束行 $k^* = \arg\max_k (Bx - b')_k$
2. 候选变量：当前为 1 且 $B_{k^*, i} > 0$ 的变量
3. 选择目标损失最小的变量翻转至 0

**Stage 2：修复混合约束可行性**

若 LP 不可行（$b - Ax < 0$）：
1. 求解 LP 验证可行性
2. 若不可行，选择对混合约束"正贡献"最大的活跃变量翻转：
   $$
   i^* = \arg\max_{i: x_i = 1} \sum_k \max(0, A_{ki})
   $$

**限制**：最多翻转 $\lfloor n/4 \rfloor$ 个变量，超过则放弃该解。

> **与 v3 的区别**：v3 的修复是单阶段（二元 + 混合混合处理），我们是严格的两阶段（先二元、后混合），且有翻转上限保护。

---

### 3.6 精英池与自适应调度

**精英池管理**：
- 容量：20（vs v3 的 8）
- 多样性阈值：Hamming 距离比例 $\geq 10\%$
- **关键改进**：若新解与池中某解相似但目标值更优，**替换旧解**（v3 会直接丢弃）

**自适应参数调度**：

| 触发条件 | 动作 | 数学表达 |
|---------|------|---------|
| 5 轮无改进 | 扩大搜索范围 | $\text{sub\_qubo\_size} \leftarrow \min(\text{sub\_qubo\_size} + 3, \text{max\_qubits})$ |
| 刚改进 | 精细搜索 | $\text{sub\_qubo\_size} \leftarrow \max(\text{sub\_qubo\_size} - 2, 6)$ |
| 剩余时间 $<10\%$ | 禁用 QAOA | $\text{max\_qubits} \leftarrow 0$（纯 SA） |
| 10 轮无改进 | 多样化重启 | 从精英池选与当前最优最远解作为起点 |

---

## 4. 核心差异对比表

| 维度 | Baseline v3 | Our Framework |
|------|------------|---------------|
| **对偶信息** | $r = 0.7 d + 0.3 \beta^*$（平滑+回归） | $l^{\text{cont}} = -A^T u^*$（精确） |
| **对偶利用** | 仅线性修正 $c^{\text{eff}} = c + r$ | 线性修正 + **Dual Rescaling**（二次项重标度） |
| **QUBO 求解器** | 固定网格 QAOA ($5 \times 4$ 参数组合) | 变分优化 QAOA (COBYLA + multi-start) |
| **subQUBO 规模** | 默认 20（可至 30） | 自适应，默认上限 15（因 QAOA 模拟成本） |
| **变量选择** | 单邻域（交互图贪心聚类） | **多邻域**（$N_1$ exploitation + $N_2$ exploration + $N_3$ random） |
| **约束惩罚** | 仅活跃约束（slack $\leq$ 1.0） | 全部二元约束，统一权重 $\lambda_B$ |
| **修复机制** | 单阶段贪心（release/loss 比率） | **两阶段**（先二元、后混合）+ 翻转上限 |
| **劣解接受** | Metropolis 准则 ($T = 0.01$) | **严格改进**（无劣解接受） |
| **重启机制** | 停滞 25 轮后精英池/随机重启 | 10 轮无改进即多样化 + 自适应参数调整 |
| **精英池** | 容量 8，相似解直接丢弃 | 容量 20，相似解可替换升级 |
| **时间管控** | 固定 120 轮迭代 | **时间预算驱动**（按剩余时间自适应） |
| **Fallback 链** | 无 | QAOA → 穷举 → SA（三重保险） |

---

## 5. 能力边界分析

### 5.1 Baseline v3 的能力边界

**优势场景**：

1. **大规模实例 + 充足时间**：v3 的 sub_size 可达 20–30，单次迭代评估的候选解数量多（QAOA 30 个 + 经典 60 个 = 90 个候选），在 120 轮迭代 + 无严格时间限制时，搜索空间覆盖更广。

2. **对偶信息不稳定的问题**：v3 的指数平滑（0.8/0.2）和 Ridge 回归对噪声有鲁棒性，当 LP 对偶价格在不同 $x$ 间剧烈波动时，回归估计比纯对偶更稳定。

3. **需要探索劣解 basin 的问题**：Metropolis 接受准则允许偶尔接受劣解，对于"多峰 + 深谷"结构的问题，这种退火式策略有助于逃离局部最优。

**劣势场景**：

1. **严格时间限制（如竞赛 60s）**：v3 固定 120 轮，无时间预算概念。sub_size=20 时 QAOA 需要模拟 20 qubit 的电路（$2^{20} = 10^6$ 维态矢量），单次迭代约 10–30s，60s 内只能完成 2–4 轮，搜索不充分。

2. **QAOA 参数固定导致 subQUBO 求解质量受限**：固定网格 $5 \times 4 = 20$ 个参数组合，每个只采样 512 shots，没有针对当前 QUBO 做参数优化。理论上最多只能覆盖参数空间的稀疏采样点。

3. **精英池容量小 + 相似解丢弃**：容量仅 8，且相似解直接丢弃而非替换。当找到更优的相似解时，无法更新精英池，导致 warm-start 质量受限。

4. **修复无翻转上限**：repair_candidate 最多 30 步，对于 $n=80$ 最多翻转 30 位（37.5%），可能过度偏离原解，丢失 QAOA/SA 发现的局部结构。

### 5.2 Our Framework 的能力边界

**优势场景**：

1. **严格时间预算（竞赛场景）**：时间预算驱动的自适应调度，能在 60s 内完成 5–15 轮迭代（SA）或 3–5 轮（QAOA），且最后 10% 时间自动切换纯 SA 保证收敛。

2. **高质量 subQUBO 求解**：变分 QAOA 通过 COBYLA 优化参数，理论上能找到比固定网格更好的参数组合；multi-start + warm-start 进一步提升了解质量。对于 $q \leq 15$ 的 subQUBO，QAOA 通常能找到比 SA 更优的解。

3. **Dual Rescaling 有效的问题**：当变量间的对偶敏感度差异大时（即某些变量对连续子问题影响远大于其他变量），重标度能显著改善 QUBO 结构，引导求解器关注关键变量组合。

4. **多邻域覆盖**：$N_1 + N_2 + N_3$ 的组合保证了对 exploitation/exploration/randomization 的全覆盖，相比单邻域更不容易遗漏改进方向。

**劣势场景**：

1. **subQUBO 规模受限**：为保证 QAOA 能在时间预算内完成，sub_size 上限设为 15（v3 可达 30）。对于需要更大邻域才能逃逸的复杂地形，可能被限制在局部 basin。

2. **严格改进接受可能陷入局部最优**：不接受劣解的策略在单峰或浅多峰问题上是优势，但在"深谷 + 必须跨越劣解区域"的问题上，可能提前收敛到局部最优。

3. **QAOA 模拟成本**：即使 $q=15$，AerSimulator 的 statevector 模拟仍需操作 $2^{15} = 32768$ 维复向量。在 CPU 上每次参数评估约 0.5–2s，COBYLA 的 20 步优化约需 10–20s。若未来竞赛要求单次量子调用更短时间（如 $<1$s），QAOA 层可能需要进一步简化或完全退化为 SA。

4. **对偶信息高度噪声时**：我们没有 v3 的平滑和回归机制，若 LP 对偶价格在相邻迭代间剧烈抖动，$l^{\text{cont}}$ 的不稳定性可能影响变量选择和 dual rescaling 的质量。

---

## 6. 实验验证与边界确认

| 实例 | 规模 | 最优值 | v3 (60 iter) | Ours (60s) | 验证结论 |
|------|------|--------|-------------|-----------|---------|
| A | $n=15$ | 106.09 | **106.09** | **106.09** | 小实例：两者均达最优，v3 初始化更强 |
| B | $n=80$ | ? | 590.50 | **606.79** | 大实例：我们的多邻域 + 变分 QAOA 领先 2.8% |

**边界确认**：
- **v3 的时间劣势**：v3 的 60 轮迭代在 sample_B 上耗时约 4 分钟，同等 60s 时间下其迭代数将锐减至 5–8 轮，性能可能降至 560 以下。
- **我们的规模劣势**：若竞赛允许 sub_size = 25 且时间充裕（如 10 分钟），v3 的大邻域覆盖可能反超。

---

## 7. 总结

从数学上看，两个方法代表了**同一技术路线**（对偶引导 + LNS + QAOA + 修复）的两种工程实现：

- **v3 偏向"广度优先"**：更大的 sub_size、更多的候选解、更多的迭代轮数、Metropolis 式探索。适合**时间充裕、需要深探索**的场景。
- **我们的框架偏向"深度优先"**：更精细的 subQUBO 求解（变分优化）、更强的对偶利用（rescaling）、更智能的邻域生成、严格的时间预算管理。适合**严格时间限制、需要高质量局部搜索**的竞赛场景。

两者的核心差距在于 **subQUBO 求解质量**（固定网格 vs 变分优化）和 **时间预算管理**（固定轮数 vs 自适应调度），这正是 sample_B 上 606.79 vs 590.50 差距的来源。
