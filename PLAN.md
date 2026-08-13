# PLAN.md — HTF 研发计划

> 本文件用中文（其余仓库内容以英文为主，便于国际化）。证据分级贯穿全文：
> `[工程]` 可用现有工具实现 · `[研究]` 真开放研究 · `[启发]` 诠释性类比 ·
> `[OUT]` 明确不声称。**状态由测试/检查导出，绝不自宣 PASS。**

## 0. 定位（一句话）

HTF 是**认证模型引擎，不是世界引擎**：它认证数值/截断误差，不认证建模误差；连续极限
`χ→∞` 是本框架**跨不过的墙**（超出工具能力范围）。价值在**有限/局部层的认证**与**工具**。

## 1. 当前状态（v0.6.0）

- [x] Layer 1 拓扑（`htf/topology.py`）：`Wire`/`Box`/`Diagram`，`>>` 与 `@`，构造期类型检查
      （维度不匹配即 `TypeError`——违背结构的图无法编译）。`[工程]`
- [x] Layer 2 函子（`htf/functor.py`）：为 `Box` 赋张量并校验形状。`[工程]`
- [x] Layer 3 引擎（`htf/engine.py`）：`float` 模式收缩（discovery-tier，**无误差界**）；
      `certified` 模式**已落地**：flint Arb 区间算术，返回带严格浮点舍入误差界的 `Certificate`。`[工程]`
- [x] Provenance 证书（`htf/certificate.py`）+ agent 可驱动 CLI（`htf/cli.py`，JSON I/O）。`[工程]`
- [x] 一维格点算子（`htf/lattice.py`）：`laplacian_box`、`heat_step_box`、`state_box`、
      `effect_box`；示例 `examples/heat_equation.py`。`[工程]`
- [x] 结构核验（`htf/structure.py`）：`check_isometry`、`check_unitary`、`check_reflection_positivity`、
      `enforce_isometry`、`enforce_unitary`；proof-carrying 结构报告（`StructureReport`）。`[工程]`
- [x] 二元 MERA 张量网络（`htf/mera.py`）：`MERALayer`、`MERA`、`random_mera`；
      顶向下收缩 `state_vector()`；参数序列化 `to_flat_params`/`from_flat_params`。`[工程]`
- [x] 变分基态（`htf/variational.py`）：`transverse_ising_ham`、`xx_model_ham`、
      `energy_expectation`、`optimize_mera`（L-BFGS-B）、`variational_bound`（认证上界）。`[工程]`
- [x] OS-正性机器核验（`htf/os_axioms.py`）：`transfer_matrix`、`reflection_operator`、
      `check_transfer_positivity`、`check_reflection_symmetry`、`os_positivity_report`；
      三重独立检验（转移矩阵 PSD、[H,R]=0、OS-Gram G=T+RTR≥0）。`[工程]`
- [x] 扩展 CLI（`htf/cli.py`）：`gap`、`variational`、`difficulty`、`os-check`、`benchmark`
      子命令，全部 JSON 输出，供 agent 直接驱动。`[工程]`
- [x] 认证复现基准套件（`htf/benchmark.py`，§4-K）：`run_benchmark`、`BenchmarkReport`、
      `BenchmarkResult`；可重放 JSON 证书；`ising`/`ising_critical`/`xx` 三模型。`[工程]`
- [x] MCP server 包装（`htf/mcp_server.py`，§7）：`MCPServer`（mcp 2.0）+ 5 工具；
      入口点 `htf-mcp`；可选依赖 `mcp[cli]`。`[工程]`
- [x] 开放系统 / CPTP（`htf/open_systems.py`，§4-H）：`density_matrix_from_pure`、
      `partial_trace`、`check_density_matrix`（Hermitian+PSD+单位迹三重核验）、
      `choi_matrix`、`check_kraus_completeness`、`lindblad_superoperator`、
      `lindblad_step`（矩阵指数精确积分）、`steady_state`（约束线性系统求解）。`[工程]`
- [x] 测试（`tests/`）：`python -m pytest -q` **688 个全绿**；总覆盖率 ≥ 98%。

## 2. 核心价值轨道（区分性价值）

- **A. 认证张量引擎。** `[研究]` 每次收缩携带**严格误差界**（键维截断 radii-polynomial /
  Newton–Kantorovich 式验证数值，区间算术 Arb / `python-flint`；有限精度区间传播）。
  可当证明的产出：对具体格点哈密顿量给出谱隙/基态能的**认证有限格点上/下界**（真定理，
  **非**连续 Clay 主张）。与 CAP-for-PDE 内核共用验证数值底层。
- **B. Proof-carrying diagrams。** `[工程]`+`[研究]` 结构性质（RP、规范不变、幺正、OS-正性）
  由类型强制 + 机器核验；把 OS-Gram `min-eig≥0`、精确有理数认证、对易性检验做成一等公民算子。
- **C. 难度地图 / 门控实验室。** `[研究]` 测量隙估计 / 关联长度 / 纠缠标度随 `χ→∞`、格点→连续
  如何退化，区分真效应 vs 有限尺寸/截断假象。
- **D. 复现底座。** `[工程]` 每个认证结果带可重放证书（输入、种子、`χ`、误差界、checker 版本）。

## 3. 路线图（阶段 + 门 gate）

- **Phase 1 — 类型安全弦图骨架（≈2 周）。** `[工程]` 已基本完成（见 §1）。**门：** hello-world
  合法态射跑通、类型不匹配报错、`pytest` 全绿。✅
- **Phase 2 — 一维模型 + 认证模式起步（≈1 个月）。** `[工程]` box 串联一维格点算子（如一维热
  方程 / 薛定谔演化）；**单一里程碑=正确性**（与传统数值解一致）。诚实：格点算子本身**就是**
  差分/谱算子的张量表示，价值在可组合/类型安全，不是"抛弃有限差分"；稳定性取决于格式，框架中立。
  **引入认证模式**：给出收缩结果的区间误差界（`certified` 模式落地）。**门：** 结果与传统解一致 +
  给出可核验误差界。✅
  - `htf/lattice.py`：`laplacian_box`、`heat_step_box`、`state_box`、`effect_box`。
  - `contract(..., mode="certified")`：flint Arb 区间算术，返回带严格浮点舍入误差界的 `Certificate`。
  - 22 个测试全绿；20 步热方程 error_bound ≈ 5.6e-16（约 1 个机器 epsilon）。
- **Phase 3 — MERA 变分 + 首个认证界 + proof-carrying（≈2 个月）。** `[工程]`/`[研究]` 自动生成
  MERA 树（严格等距）+ 变分基态；对小系统给**首个认证谱隙界**（轨道 A）；**首个结构核验**
  （轨道 B，如机器核验转移矩阵网络的反射正性）。"第 5 维涌现"作 `[启发]` 级可视化。**门：** 残量
  包络闭合、结构核验通过。✅
  - `htf/structure.py`：等距/幺正缺陷、RP 核验、SVD retraction；所有检验 < 1e-15。
  - `htf/mera.py`：二元 MERA，`state_vector()` 顶向下收缩，`random_mera`（SVD 构造）。
  - `htf/variational.py`：TFIM/XX 哈密顿量、L-BFGS-B 优化、`variational_bound` 认证上界。
  - 示例 `examples/mera_variational.py`：E_var ≥ E_0（认证上界成立）。
  - 309 个测试全绿；总覆盖率 99%，每模块 ≥ 93%（全超 80% 目标）。
- **Phase 4 — 认证有限格点物理（核心目标）。** `[研究]`+`[OUT]` 对格点哈密顿量（含小格点规范
  理论）给谱隙**认证有限格点上/下界**，严格误差控制下做 `χ→∞` 外推，机器核验 OS-正性/规范不变；
  产出**认证有限格点定理** + 难度地图。**明确不声称（`[OUT]`）：不是连续 Yang–Mills 质量隙的证明。**
  ✅ 已完成（v0.4.0）：
  - `htf/gap.py`：`spectral_gap_exact`、`h2_expectation`、`temple_lower_bound`（刚性有限格点下界）、
    `first_excited_upper`（变分激发态上界）、`certified_gap_upper`（Arb 认证版）、`gap_report`。
  - `htf/scaling.py`：`chi_convergence_study`（通用 `ham_factory` 接口）、`ScalingReport`、
    幂律外推（`[启发]`，非认证）。
  - `htf/difficulty.py`：`entanglement_entropy`、`entanglement_spectrum`、
    `bipartite_entanglement_profile`、`DifficultyReport`、`difficulty_report`（难度分级）。
  - `htf/os_axioms.py`：`transfer_matrix`、`reflection_operator`、`check_transfer_positivity`、
    `check_reflection_symmetry`、`os_positivity_report`（三重 OS-正性机器核验）。
  - CLI 扩展：`gap`、`variational`、`difficulty`、`os-check`，全 JSON 输出。
  - `examples/phase4_certified_physics.py`：全流程 demo，所有断言通过。
  - 586 个测试全绿；总覆盖率 ≥ 98%，每模块 ≥ 93%。
  - **门：** 认证上界成立 ✅；Temple 下界逻辑正确（需 E_var < E_1 条件）✅；难度图产出 ✅；
    OS-正性三重机器核验通过 ✅；CLI 子命令完整 ✅。

## 4. 扩展能力（选做子集，非全做）

E 语义保持图重写（ZX）`[研究]` · F 量子线路互操作（QASM / PyZX / NISQ）`[工程]`/`[研究]` ·
G 对称/规范不变张量作为类型（`U(1)`/`SU(N)` block-sparse）`[研究]` · H 开放系统 / CPTP `[工程]` ·
I 严格双侧界（Lanczos/Anderson 型下界）`[研究]` · J 可微逆向设计 / 哈密顿量学习 `[工程]`/`[研究]` ·
K 认证复现基准套件 `[工程]` · L（远期/投机）导出到证明助手 Lean/Coq `[研究,投机]`。

## 5. 展现形态

三层→三形态：① 拓扑层→**图** + 类型检查报告；② 函子层→**IR**（爱因斯坦索引计算图 + 收缩路径 +
成本估计）；③ 引擎层→**数/张量 + 认证误差区间** + 图表 + 证书。产品壳：库 / Notebook / 可视化节点图
编辑器 / 命令行 checker / Web playground / 导出（TikZ、QASM、证书）/ **agent-可驱动 CLI+MCP**。
可视化"北极星"= Scratch 式友好界面（界面友好可达；领域友好有天花板——物理内禀专家级；对真小白，
对话式 agent 路线更现实）。

## 6. 诚实边界（不可删）

不声称免疫 UV 发散（键维正则化）；不声称证明/逼近连续 Yang–Mills 质量隙（`[OUT]`）；不把 MERA↔AdS
说成已实现的全息求解（`[启发]`）；不声称"抛弃 PDE / 绝对稳定性"；不声称是预测现实的"世界引擎"
（认证数值误差、非建模误差；受面积律纠缠边界限制）。

## 7. TODO（文档/国际化）

- [x] 英文版设计白皮书（`docs/whitepaper.en.md`）。`[工程]`
  - 8 节：定位、边界、架构、核心能力（全部子功能）、CLI/MCP、证据语法、依赖、诚实限制。
- [ ] 节点图可视化前端原型（React Flow 类）。`[工程]`
- [x] MCP server 包装，供 agent 直接连接。`[工程]`
  - `htf/mcp_server.py`：`MCPServer`（mcp 2.0 API）+ `@server.tool` 装饰器；
    5 个工具：`htf_version`、`htf_variational`、`htf_gap`、`htf_os_check`、`htf_benchmark`；
    入口点 `htf-mcp = "htf.mcp_server:main"`；可选依赖 `mcp[cli]`。
- [x] 认证复现基准套件（§4-K）。`[工程]`
  - `htf/benchmark.py`：`run_benchmark`、`BenchmarkReport`、`BenchmarkResult`；
    可重放 JSON 报告；CLI `htf benchmark [--models ising xx]`。
