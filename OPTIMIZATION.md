# OPTIMIZATION.md — HTF 全面优化计划

> 本文件记录 v0.12.0 之后的修复/强化优先级。证据分级同 PLAN.md。
> **状态由测试导出，绝不自宣 PASS。**

---

## A 组 — 正确性 / 数值稳定性（最高优先级）

### A-1 Lanczos 重正交化 `[工程]`
**问题**：朴素 Lanczos 经 k>20 步后，浮点误差使 Krylov 向量失去正交性，
Ritz 值出现虚假重复（"幽灵特征值"），Temple 下界随之偏移。  
**修复**：在 `htf/lanczos.py` 的主循环里加入完整 Gram-Schmidt 重正交化（每步对
已生成的所有向量做投影），并用测试验证 n=8、k=30 时 V^T V ≈ I 到机器精度。

### A-2 ZX `zx_to_matrix` 鲁棒性 `[工程]`
**问题**：激进重写（spider_fusion）后 ZX 图不再是"电路形"，`zx_to_matrix`
静默返回错误矩阵而不报错。  
**修复**：在 `zx_to_matrix` 入口处检测图是否为有向无环（电路形）；若不是则
抛出带说明的 `NotImplementedError`，而不是返回错误结果。

### A-3 `circuit_to_diagram` 非相邻双比特门 `[工程]`
**问题**：`circuit_to_diagram` 对非相邻双比特门静默降级为全宽 Box，
丢失结构信息，类型检查形同虚设。  
**修复**：自动插入 SWAP 门将两个比特调换为相邻，再作用双比特门，
最后 SWAP 回来。添加 `adjacent_only=False` 参数（默认启用 SWAP 分解）。

---

## B 组 — 功能完整性

### B-1 ZX 颜色交换规则（H-共轭）`[工程]`
**问题**：现有 3 条规则无法做任何颜色改变，所有 X spider 永远留在图里。  
**修复**：在 `htf/zx.py` 实现 `color_change` 规则：一个 Z spider 若被
偶数个 H box 包围，可转换为 X spider（反之亦然），并消去这些 H box。  
同时实现 `pi_copy` 规则（Z(π) 穿透 X(0) 蜘蛛的复制规则）。

### B-2 CLI 补全 `[工程]`
**问题**：v0.7.0–v0.12.0 新增了 6 个模块，CLI 一个新子命令都没有。  
**修复**：在 `htf/cli.py` 增加：
- `htf lanczos --model --n --k` → 输出 TwoSidedBounds JSON
- `htf qasm-sim --file circuit.qasm` → 输出幺正矩阵 JSON
- `htf zx-simplify --file circuit.qasm` → 输出重写后节点数 + 步数
- `htf inverse --model --n --target-e0` → 输出 InverseDesignResult JSON
- `htf lean-export --model --n --output file.lean` → 生成 Lean 骨架

### B-3 MCP server 补全 `[工程]`
**问题**：MCP server 只有 5 个工具（截至 v0.5.0），新功能全部缺席。  
**修复**：在 `htf/mcp_server.py` 增加：
- `htf_lanczos` — 两侧界
- `htf_qasm_simulate` — QASM 电路幺正模拟
- `htf_zx_simplify` — ZX 重写统计
- `htf_inverse` — 逆向设计

---

## C 组 — 性能

### C-1 `check_u1_invariance` 向量化 `[工程]`
**问题**：当前实现用 `np.nditer` 逐元素循环，对 n=4（4比特，16×16 矩阵）
已有 ~65 k 次 Python 调用，n=6 时慢到不可用。  
**修复**：用 numpy 广播构造全量电荷差矩阵，一次向量化比较，O(d²) 而非
O(d²) Python 调用。

### C-2 `u1_blocks` / `_flat_charges` 稀疏表示 `[工程]`
**问题**：`_flat_charges` 对多 wire 组合用 `ravel()` 展开，空间随 wire 数
指数增长。  
**修复**：用懒惰生成器 + `itertools.product` 替换，仅在需要时枚举扇区，
避免 O(∏d_i) 的预先展开。

### C-3 引擎接入 opt_einsum `[工程]`
**问题**：`htf/engine.py` float 模式用朴素 `np.einsum`，收缩路径不优化；
`opt_einsum` 已在 `pyproject.toml` 列为可选依赖但从未被使用。  
**修复**：在 `contract()` 里检测 `opt_einsum` 是否可用；若可用则用它选择
最优收缩路径（`cotengra` 或 `greedy`）。

---

## D 组 — 文档 / 示例

### D-1 白皮书更新 `[工程]`
**问题**：`docs/whitepaper.en.md` 截止 v0.6.0，缺少 Lanczos / QASM / ZX /
Inverse / Symmetric / Lean 六个模块的描述。  
**修复**：更新白皮书第 4 节（核心能力），补全 v0.7.0–v0.12.0 所有模块。

### D-2 新模块示例脚本 `[工程]`
**问题**：`examples/` 只有 3 个脚本，均止于 v0.4.0。  
**修复**：新增：
- `examples/lanczos_bounds.py`
- `examples/bell_circuit_qasm.py`
- `examples/zx_simplify_demo.py`
- `examples/hamiltonian_learning.py`
- `examples/lean_skeleton_demo.py`

---

## 执行顺序

1. A-1 Lanczos 重正交化（正确性，影响所有 Lanczos 相关功能）
2. A-2 ZX zx_to_matrix 鲁棒性（防止静默错误）
3. C-1 check_u1_invariance 向量化（最快的性能提升）
4. B-1 ZX 颜色交换 + pi_copy 规则（扩展 ZX 实用性）
5. A-3 circuit_to_diagram SWAP 分解（类型安全完整性）
6. B-2 CLI 补全（用户可见性）
7. B-3 MCP server 补全（agent 接口完整性）
8. C-2 u1_blocks 稀疏化（扩展性）
9. C-3 opt_einsum 集成（性能）
10. D-1 白皮书更新
11. D-2 示例脚本
