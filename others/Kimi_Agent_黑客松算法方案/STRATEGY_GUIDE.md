# 2026量子计算大赛 - 混合整数优化赛道 工程实现策略

> **竞赛类型**: 3小时限时黑客松 | **问题类型**: MIQP (Mixed-Integer Quadratic Programming)
> **硬件限制**: 单次量子≤30比特，建议≤20 | **必选框架**: Qiskit或PennyLane

---

## 一、技术栈选型决策

### 1.1 Qiskit vs PennyLane：推荐Qiskit

| 维度 | Qiskit | PennyLane | 黑客松推荐 |
|------|--------|-----------|-----------|
| API成熟度 | 高，QAOA有内置类 | 中，需手写电路 | **Qiskit** |
| 调试便利性 | 文档完善，社区大 | 较小 | **Qiskit** |
| 自动微分 | 不支持（需SPSA/COBYLA） | 支持 | PennyLane |
| 代码复杂度 | 低（几行搞定） | 中（需手动构建电路） | **Qiskit** |
| 性能 | 中等 | 梯度优化快 | 打平 |
| 模拟器稳定性 | Aer非常稳定 | default.qubit稳定 | **Qiskit** |

**结论**: 选择Qiskit。原因：
1. 黑客松最重要的是**快速出可运行代码**，Qiskit的`QAOA`类封装完整
2. 3小时内不需要精细调优，COBYLA无梯度优化足够
3. 出问题能快速Google到解决方案

```python
# Qiskit API（极简）
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=100), reps=2)
result = qaoa.compute_minimum_eigenvalue(hamiltonian)
```

### 1.2 经典LP求解器：推荐SciPy

| 求解器 | 优点 | 缺点 | 推荐度 |
|--------|------|------|--------|
| **SciPy `linprog`** | 已安装，速度快，API简单 | 功能有限 | **★★★★★** |
| CVXPY | 建模优雅 | 需额外安装，可能有依赖冲突 | ★★★☆☆ |
| PuLP | 轻量 | 性能一般 | ★★★☆☆ |
| 纯NumPy | 零依赖 | 自己实现易出bug | ★★☆☆☆ |

```python
# SciPy调用方式
from scipy.optimize import linprog
res = linprog(c=-h, A_ub=G, b_ub=rhs, bounds=[(0,None)]*p, method="highs")
y = res.x  # 最优解
```

### 1.3 并行化策略

**推荐：多进程并行求解多个test实例**

```python
from multiprocessing import Pool

def solve_one(prefix):
    data = load_data(data_dir, prefix)
    return solve_single_instance(data, output_dir)

# 5个test并行跑（假设有5核）
with Pool(processes=5) as pool:
    results = pool.map(solve_one, ["test_1","test_2","test_3","test_4","test_5"])
```

**不推荐并行多个subQUBO**：每个subQUBO求解时间很短（秒级），并行开销反而更大。

---

## 二、代码架构设计

### 2.1 项目结构

```
miqp_solver/
├── config.py              # 全局配置（可调参数集中）
├── main.py                # 主入口，端到端pipeline
├── run_hackathon.py       # 一键运行脚本
├── requirements.txt       # 依赖列表
│
├── core/                  # 核心求解模块
│   ├── __init__.py
│   ├── quantum_solver.py  # QAOA求解器（Qiskit+PennyLane）
│   ├── lp_solver.py       # 经典LP求解器
│   └── benders.py         # Benders分解主循环
│
├── utils/                 # 工具模块
│   ├── __init__.py
│   ├── data_loader.py     # 数据加载与保存
│   ├── blocking.py        # subQUBO分块策略
│   └── validator.py       # 解验证与可行性检查
│
├── visualization/         # Paper图表生成
│   └── paper_viz.py       # 收敛曲线、分块图、电路图
│
└── tests/                 # 测试模块
    └── test_pipeline.py   # 赛前验证pipeline
```

### 2.2 核心接口定义

```python
# ===== utils/data_loader.py =====
def load_data(data_dir: str, prefix: str) -> dict:
    """加载MIQP数据，返回 {Q, c, h, A, G, b, B, bp, n, p, m, m2}"""
    ...

def save_solution(output_dir, prefix, x, y, obj_val, info):
    """保存解为JSON和NPY格式"""
    ...

# ===== utils/blocking.py =====
def coupling_strength_blocks(Q: np.ndarray, block_size: int) -> List[List[int]]:
    """耦合强度分块，将强耦合变量放在同一block"""
    ...

def extract_subqubo(Q, c, fixed_vars, block_vars) -> Tuple[Q_sub, c_sub, var_map, constant]:
    """从完整QUBO提取subQUBO"""
    ...

# ===== core/quantum_solver.py =====
def create_quantum_solver(backend="qiskit", reps=2, shots=1024, maxiter=100) -> Solver:
    """工厂函数，创建量子求解器"""
    ...

def solve_qubo_exact(Q, c) -> Tuple[x_best, obj_best]:
    """经典穷举求解（n<=15）"""
    ...

# ===== core/benders.py =====
class QuantumBendersSolver:
    def __init__(self, data, block_size=15, n_qaoa_blocks=1):
        ...
    
    def solve(self, max_iter=50, tol=1e-3, time_limit=1800) -> dict:
        """主循环，返回 {x, y, objective, history, time}"""
        ...

# ===== utils/validator.py =====
def check_feasibility(data, x, y, tol=1e-6) -> dict:
    """检查MIQP解可行性"""
    ...

def compute_objective(data, x, y) -> float:
    """计算目标函数值"""
    ...
```

---

## 三、3小时时间预算规划

### 3.1 时间线总览

```
比赛开始前 -----------+ 收到test数据 -----------+ 截止提交
                     |                         |
                [收到2个sample]           [收到5个test]
                     |                         |
    准备阶段(2小时)   |    求解阶段(2小时)       |  Paper(1小时)
    ──────────────────┼─────────────────────────┼────────────
                     |                         |
    T-2h: 环境配置    |   T=0h: 加载所有test      |  T+2h: 生成图表
    T-1.5h: 跑通sample|   T+0.5h: 开始批量求解    |  T+2.5h: 写Paper
    T-1h: 调参优化    |   T+1.5h: 收集所有结果    |  T+3h: 提交！
                     |                         |
```

### 3.2 收到test数据前的准备（约2小时可用）

| 时间 | 任务 | 具体操作 | 产出 |
|------|------|----------|------|
| 0-30min | 环境配置 | `pip install qiskit qiskit-algorithms scipy numpy` | 环境就绪 |
| 30-60min | 跑通sample_A | 用穷举法验证pipeline | 确认数学正确 |
| 60-90min | 跑通sample_B | 验证分块+QAOA | 确认架构正确 |
| 90-120min | 参数调优 | 调整QAOA reps, shots, block_size | 最优参数 |

### 3.3 收到test后的3小时分配

| 时间段 | 任务 | 时间预算 | 关键动作 |
|--------|------|----------|----------|
| **第1小时** | 核心算法调试 | 60min | 先用sample_A快速验证，然后sample_B验证分块 |
| **第2小时** | test批量求解 | 60min | 5个test并行跑，每个约10-12分钟 |
| **第2.5-3小时** | Paper+提交 | 60min | 生成图表，撰写文档，打包提交 |

### 3.4 每个test的预期时间预算

| 实例规模 | 求解策略 | 预期时间 | 超时Fallback |
|----------|----------|----------|-------------|
| n=15 | 穷举+LP | 1-5秒 | 穷举不可能超时 |
| n=50 | Benders + 分块QAOA | 5-10分钟 | 贪心+局部搜索 |
| n=80 | Benders + 分块QAOA | 10-15分钟 | 减少QAOA迭代次数 |
| n=100 | Benders + 分块QAOA(少量块) | 12-18分钟 | 经典启发式 |
| n=150 | Benders + 贪心(大部分块) | 15-20分钟 | 纯贪心+局部搜索 |

### 3.5 超时Fallback策略

```python
# 在主循环中实现动态超时
time_per_instance = total_remaining_time / remaining_instances

for prefix in test_prefixes:
    start = time.time()
    
    try:
        # 正常求解
        result = solve_with_qaoa(data, time_limit=time_per_instance)
    except TimeoutError:
        # Fallback 1: 减少QAOA迭代次数
        result = solve_with_qaoa(data, time_limit=time_per_instance,
                                 qaoa_reps=1, maxiter=50)
    except Exception:
        # Fallback 2: 纯经典贪心
        result = greedy_solve(data, time_limit=time_per_instance)
    
    # 如果还有时间，尝试改进
    elapsed = time.time() - start
    if elapsed < time_per_instance * 0.5:
        result = local_search_improve(data, result, time_limit=time_per_instance*0.5)
```

---

## 四、从Sample到Test的快速扩展策略

### 4.1 Sample_A (n=15) - 验证数学正确性

```python
# sample_A直接用穷举，用于验证Benders迭代的数学正确性
def test_sample_a(data):
    """穷举所有2^15=32768种x组合，找到真正的最优解"""
    best_obj = float("-inf")
    best_x = None
    
    for i in range(2**15):
        x = np.array([(i>>j)&1 for j in range(15)], dtype=float)
        y, success, _ = solve_lp_scipy(h, G, b - A@x)
        if success:
            obj = x@Q@x + c@x + h@y
            if obj > best_obj:
                best_obj, best_x = obj, x
    
    return best_x, best_obj  # 作为baseline
```

### 4.2 Sample_B (n=80) - 验证分块逻辑

```python
# sample_B验证分块+QAOA能否在合理时间内找到可行解
blocks = coupling_strength_blocks(Q, block_size=15)
print(f"n=80 -> {len(blocks)} blocks, sizes={[len(b) for b in blocks]}")

# 验证：解合并后是否一致
x_test = np.random.randint(0, 2, 80).astype(float)
for block in blocks[:3]:  # 只验证前3块
    x_sub, _ = solve_subqubo_block(block, x_test)
    x_test = merge_solution(x_test, x_sub, np.array(block))
```

### 4.3 自动化测试流程

```bash
# 一键验证所有组件
python run_hackathon.py --phase prepare

# 输出示例：
# [OK] All tests passed!
#   Data loading:   PASS
#   Exact solver:   PASS
#   LP solver:      PASS
#   Blocking:       PASS
#   subQUBO extract: PASS
#   Full pipeline:  PASS
```

### 4.4 结果验证函数

```python
def quick_check(data, x, y):
    """10秒内完成所有可行性检查"""
    # 1. 二元约束 (1ms)
    assert np.allclose(x, x.astype(int)), "x不是整数"
    assert np.all((x == 0) | (x == 1)), "x不在{0,1}中"
    
    # 2. 线性约束 (1ms)
    viol1 = np.maximum(0, A@x + G@y - b)
    assert np.max(viol1) < 1e-4, f"混合约束违反: {np.max(viol1)}"
    
    # 3. 二元约束 (1ms)
    viol2 = np.maximum(0, B@x - bp)
    assert np.max(viol2) < 1e-4, f"二元约束违反: {np.max(viol2)}"
    
    # 4. 非负约束
    assert np.all(y >= -1e-6), "y有负分量"
    
    # 5. 计算目标值
    obj = x@Q@x + c@x + h@y
    return obj
```

---

## 五、关键代码片段（完整可运行）

### 5.1 数据加载和预处理

```python
import numpy as np
import os

def load_data(data_dir: str, prefix: str):
    """加载MIQP问题数据"""
    def _load(fname):
        path = os.path.join(data_dir, f"{prefix}_{fname}.npy")
        return np.load(path) if os.path.exists(path) else None
    
    Q = _load("Q"); c = _load("c"); h = _load("h")
    A = _load("A"); G = _load("G"); b = _load("b")
    B = _load("B"); bp = _load("bp")
    
    n = Q.shape[0] if Q is not None else 0
    p = h.shape[0] if h is not None else 0
    m = b.shape[0] if b is not None else 0
    m2 = bp.shape[0] if bp is not None else 0
    
    return {"Q":Q, "c":c, "h":h, "A":A, "G":G, "b":b, 
            "B":B, "bp":bp, "n":n, "p":p, "m":m, "m2":m2, "prefix":prefix}
```

### 5.2 QAOA求解subQUBO（Qiskit版本）

```python
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import BackendSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
import numpy as np

def qubo_to_ising(Q, c):
    """QUBO -> Ising哈密顿量"""
    n = Q.shape[0]
    constant = np.sum(Q)/4 + np.sum(c)/2
    linear = np.zeros(n)
    for i in range(n):
        linear[i] = -0.5*c[i] - 0.25*np.sum(Q[i,:]+Q[:,i])
    
    pauli_list, coeffs = ["I"*n], [constant]
    for i in range(n):
        z = ["I"]*n; z[i] = "Z"
        pauli_list.append("".join(z)); coeffs.append(linear[i])
    for i in range(n):
        for j in range(i+1,n):
            if abs(Q[i,j]) > 1e-10:
                z = ["I"]*n; z[i] = "Z"; z[j] = "Z"
                pauli_list.append("".join(z)); coeffs.append((Q[i,j]+Q[j,i])/4)
    
    return SparsePauliOp(pauli_list, coeffs)

def solve_subqubo_qaoa(Q_sub, c_sub, reps=2, shots=1024, maxiter=100):
    """用QAOA求解subQUBO"""
    n = Q_sub.shape[0]
    hamiltonian = qubo_to_ising(Q_sub, c_sub)
    
    sampler = BackendSampler(backend=AerSimulator(), options={"shots": shots})
    optimizer = COBYLA(maxiter=maxiter, rhobeg=0.1)
    
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps)
    result = qaoa.compute_minimum_eigenvalue(hamiltonian)
    
    # 解码最优bitstring
    best_bits = result.best_measurement["bitstring"]
    x_best = np.array([int(best_bits[i]) for i in range(n)], dtype=float)
    obj_best = x_best @ Q_sub @ x_best + c_sub @ x_best
    
    return x_best, obj_best
```

### 5.3 Benders分解主循环

```python
def benders_solve(data, max_iter=50, tol=1e-3, time_limit=1800):
    """
    Benders分解主循环
    外层：迭代更新x，内层：求解LP和subQUBO
    """
    Q, c, h = data["Q"], data["c"], data["h"]
    A, G, b = data["A"], data["G"], data["b"]
    n, p = data["n"], data["p"]
    
    import time
    start = time.time()
    
    # 初始解
    x_best = np.random.randint(0, 2, n).astype(float)
    y_best, best_obj = solve_lp_and_evaluate(data, x_best)
    
    history = {"iterations": [], "objectives": [best_obj]}
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        # Step 1: 固定x，求解LP得到y
        y_new, obj_lp = solve_lp_and_evaluate(data, x_best)
        
        # Step 2: 固定y，求解subQUBO块更新x
        x_new = solve_blocks_qaoa(data, x_best)
        
        # Step 3: 评估
        y_new, obj_new = solve_lp_and_evaluate(data, x_new)
        
        history["iterations"].append(it)
        history["objectives"].append(obj_new)
        
        if obj_new > best_obj:
            improvement = obj_new - best_obj
            best_obj, x_best, y_best = obj_new, x_new.copy(), y_new
            if improvement < tol:
                break
    
    return {"x": x_best, "y": y_best, "objective": best_obj, 
            "history": history, "time": time.time()-start}
```

### 5.4 subQUBO分块函数

```python
def coupling_strength_blocks(Q: np.ndarray, block_size: int):
    """耦合强度贪心分块"""
    n = Q.shape[0]
    assigned = set()
    blocks = []
    coupling = np.abs(Q) + np.abs(Q.T)
    np.fill_diagonal(coupling, 0)
    
    while len(assigned) < n:
        unassigned = [i for i in range(n) if i not in assigned]
        seed = unassigned[0]
        block = [seed]
        assigned.add(seed)
        
        while len(block) < block_size and len(assigned) < n:
            best_var, best_strength = None, -1
            for v in [i for i in range(n) if i not in assigned]:
                strength = sum(coupling[v, b] for b in block)
                if strength > best_strength:
                    best_strength, best_var = strength, v
            if best_var is not None and best_strength > 0:
                block.append(best_var)
                assigned.add(best_var)
            else:
                break
        blocks.append(block)
    return blocks

def random_blocks(n: int, block_size: int, seed=42):
    """随机分块（最快，用于验证）"""
    rng = np.random.RandomState(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    return [indices[i:i+block_size] for i in range(0, n, block_size)]

def extract_subqubo(Q, c, fixed_vars, block_vars):
    """提取subQUBO，fixed_vars=-1表示未固定"""
    Q_sub = Q[np.ix_(block_vars, block_vars)].copy()
    c_sub = c[block_vars].copy()
    fixed_indices = np.where(fixed_vars != -1)[0]
    constant = 0.0
    
    # 固定变量的贡献加到c_sub和constant
    for i, bi in enumerate(block_vars):
        for fj in fixed_indices:
            c_sub[i] += Q[bi, fj]*fixed_vars[fj] + Q[fj, bi]*fixed_vars[fj]
    for fi in fixed_indices:
        for fj in fixed_indices:
            constant += Q[fi, fj]*fixed_vars[fi]*fixed_vars[fj]
        constant += c[fi]*fixed_vars[fi]
    
    return Q_sub, c_sub, np.array(block_vars), constant
```

### 5.5 结果验证函数

```python
def check_feasibility(data, x, y, tol=1e-6):
    """检查MIQP解的可行性"""
    Q, c, h = data["Q"], data["c"], data["h"]
    A, G, b = data["A"], data["G"], data["b"]
    B, bp = data["B"], data["bp"]
    
    results = {}
    results["binary_violation"] = float(np.max(np.abs(x*(1-x))))
    
    if A is not None and G is not None:
        results["ineq_violation"] = float(np.max(np.maximum(0, A@x + G@y - b)))
    else:
        results["ineq_violation"] = 0.0
    
    if B is not None:
        results["bineq_violation"] = float(np.max(np.maximum(0, B@x - bp)))
    else:
        results["bineq_violation"] = 0.0
    
    results["feasible"] = (
        results["binary_violation"] < tol and
        results["ineq_violation"] < tol and
        results["bineq_violation"] < tol
    )
    return results

def compute_objective(data, x, y):
    """计算MIQP目标函数值"""
    obj = x @ data["Q"] @ x + data["c"] @ x
    if data["h"] is not None and y is not None:
        obj += data["h"] @ y
    return float(obj)
```

### 5.6 完整端到端Pipeline

```python
#!/usr/bin/env python
"""端到端MIQP求解Pipeline"""
import numpy as np
import os, time

# === 1. 加载数据 ===
data_dir = "./data"
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

prefix = "sample_A"  # 或 "test_1", "test_2", ...
data = load_data(data_dir, prefix)
print(f"Loaded: n={data['n']}, p={data['p']}")

# === 2. 求解 ===
if data["n"] <= 15:
    # 小规模：穷举
    result = solve_small_instance(data)
else:
    # 大规模：Benders + QAOA
    solver = QuantumBendersSolver(data, block_size=15)
    result = solver.solve(max_iter=30, time_limit=600)

# === 3. 验证 ===
x, y = result["x"], result["y"]
obj = result["objective"]
feas = check_feasibility(data, x, y)
print(f"Objective: {obj:.6f}")
print(f"Feasible: {feas['feasible']}")

# === 4. 保存结果 ===
save_solution(output_dir, prefix, x, y, obj, {
    "time": result["time"],
    "method": result.get("method", "benders")
})
```

---

## 六、Paper撰写快速模板

### 6.1 Paper结构建议（6页标准）

```
1. Introduction (0.5页)
   - 问题背景：MIQP的NP难性质
   - 量子计算动机：QAOA对组合优化的潜力
   - 我们的方法概述

2. Problem Formulation (0.5页)
   - 数学公式
   - 分解策略说明

3. Methodology (2页，重点)
   3.1 Benders Decomposition
       - 分解原理
       - 迭代流程图
   3.2 subQUBO Block Strategy
       - 分块示意图
       - 耦合强度分块算法
   3.3 Quantum Annealing via QAOA
       - QAOA电路图
       - 参数选择说明

4. Experiments (2页)
   4.1 Dataset Description
       - sample_A, sample_B描述
       - test数据规模递增
   4.2 Results
       - 结果汇总表
       - 收敛曲线图
   4.3 Analysis
       - 求解时间分析
       - 量子比特使用效率

5. Conclusion (0.5页)
   - 主要贡献总结
   - 未来改进方向
```

### 6.2 快速生成图表

```python
from visualization.paper_viz import generate_all_paper_figures

# 求解完成后一键生成所有图表
figures = generate_all_paper_figures(
    data=data,
    blocks=solver.blocks,
    history=result["history"],
    results=all_results,
    output_dir="./submission/figures"
)
# 生成：fig_convergence.png, fig_blocks.png, fig_circuit.png, fig_summary.png
```

### 6.3 关键图表说明

| 图表 | 用途 | 生成时间 |
|------|------|----------|
| 收敛曲线 | 展示Benders迭代收敛 | 1秒 |
| 分块热力图 | 展示QUBO矩阵分块结构 | 1秒 |
| QAOA电路图 | 展示量子电路设计 | 1秒 |
| 结果汇总柱状图 | 展示各test实例结果 | 1秒 |

### 6.4 Paper LaTeX片段模板

```latex
\section{Methodology}

Our approach combines Benders decomposition with QAOA to solve the MIQP problem:

\begin{equation}
\max_{x,y} \; x^\top Q x + c^\top x + h^\top y
\end{equation}

\subsection{Benders Decomposition}

We decompose the problem into a master QUBO over $x$ and a linear subproblem over $y$:
\begin{align}
\text{Master:} \quad & \max_x \; x^\top Q x + c^\top x + \eta(x) \\
\text{Sub-LP:} \quad & \eta(x) = \max_y \{h^\top y : Gy \leq b - Ax, y \geq 0\}
\end{align}

\subsection{subQUBO Block Decomposition}

For $n > 20$, we partition variables into blocks of size $\leq 15$:
\begin{equation}
\mathcal{V} = \bigcup_{k=1}^{K} \mathcal{B}_k, \quad |\mathcal{B}_k| \leq 15
\end{equation}

Variables in block $\mathcal{B}_k$ are solved via QAOA while others are fixed.

\subsection{QAOA Circuit Design}

We use QAOA with $p=2$ layers. The circuit consists of:
\begin{itemize}
\item Hadamard initialization for superposition
\item Cost layer: $U_C(\gamma) = e^{-i\gamma H_C}$ with $H_C$ encoding the QUBO
\item Mixer layer: $U_M(\beta) = e^{-i\beta \sum_i X_i}$
\end{itemize}
```

---

## 七、赛前Checklist

### 7.1 比赛前一天
- [ ] 克隆代码仓库到本地
- [ ] 运行 `python tests/test_pipeline.py` 确保全部通过
- [ ] 运行sample_A验证结果正确
- [ ] 运行sample_B验证分块逻辑
- [ ] 熟悉一键运行命令
- [ ] 测试网络环境（比赛平台）

### 7.2 比赛当天（收到test前）
- [ ] 再次运行测试确认环境正常
- [ ] 检查config.py中的参数设置
- [ ] 等待test数据...

### 7.3 收到test后（黄金3小时）
- [ ] **0-10分钟**: 复制test数据到data目录，检查文件完整性
- [ ] **10-20分钟**: 快速跑一个test_1确认pipeline正常
- [ ] **20-120分钟**: 启动批量求解（多进程并行）
- [ ] **120-150分钟**: 收集结果，检查可行性
- [ ] **150-170分钟**: 生成Paper图表，撰写文档
- [ ] **170-180分钟**: 最终检查，打包提交

---

## 八、故障排除指南

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| Qiskit导入失败 | 版本不兼容 | `pip install qiskit qiskit-algorithms qiskit-aer --upgrade` |
| QAOA求解失败 | 哈密顿量过大 | 检查n≤20，分块是否正确 |
| LP不可行 | 约束过于严格 | 检查约束是否被正确加载 |
| 内存溢出 | Q矩阵太大 | 使用稀疏矩阵表示 |
| 求解时间过长 | QAOA迭代太多 | 减少maxiter到50，reps降到1 |
| 结果不可行 | 数值精度问题 | 增加tol到1e-4 |
| test数据格式不同 | 主办方变格式 | 修改load_data函数适配 |

---

> **最后提醒**: 3小时很短，**先求可行解，再求最优解**。
> 如果某个test超时，立即切换到Fallback策略，不要死磕！
