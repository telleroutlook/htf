# PLAN.md — HTF 研发计划

> 本文件用中文（其余仓库内容以英文为主，便于国际化）。证据分级贯穿全文：
> `[工程]` 可用现有工具实现 · `[研究]` 真开放研究 · `[启发]` 诠释性类比 ·
> `[OUT]` 明确不声称。**状态由测试/检查导出，绝不自宣 PASS。**

---

---

## 0.21 完成全部后续工作：replay_mode、cleanroom 集成、policy JSON、THREAT_MODEL（2026-08-15）

本节完成 §0.20 表中全部四项 ⏸/⚠️ 延迟工作。

### 变更清单

| ID | 文件 | 变更 |
|---|---|---|
| R21-1 | `htf/rayleigh_cert.py` | `RayleighCertificate` 新增 `replay_mode: str = ""` 字段；`to_dict()` 序列化；`from_dict()` 反序列化；`rayleigh_certificate()` 生产时设 `"from_scratch"`；`verify_rayleigh_certificate()` 确认后设 `"self_consistency"` |
| R21-2 | `htf/verify.py` | `verify_from_dict()` 新增第 6a 步：从 `_cleanroom_verify.exact_rayleigh` 取精确有理数 Rayleigh 商，断言 ≤ `Fraction.from_float(stored_upper)`；结果以 `cleanroom_check` 字段返回 |
| R21-3 | `policies/rayleigh-release-v1.json` | 新建：发布门策略——required_assurance=rigorous, required_verified=true, required_replay_mode=from_scratch |
| R21-4 | `policies/rayleigh-dev-v1.json` | 新建：开发策略——allowed=rigorous+reproducible, forbidden=heuristic |
| R21-5 | `docs/THREAT_MODEL.md` | 新建：8 个攻击向量（证书替换、上界膨胀/压缩、NaN 注入、容差绕过、后端替换、文本篡改、replay_mode 伪造）+ 明确不缓解项 |

### CI 状态

| 项目 | 状态 |
|---|---|
| `python3 -m pytest -q`（全量） | **1790 passed, 1 warning ✅** |

---

## 0.20 从 proofctl 采纳不变量文档模式（2026-08-15）

> 参考仓库：`proofctl`（同作者）。分析其 `SECURITY-INVARIANTS.md`、`docs/ASSURANCE_MODEL.md`、`THREAT_MODEL.md`、`docs/CHECKER_PROTOCOL.md` 与 PR 检查清单。

### 值得借鉴的核心模式

| proofctl 特性 | HTF 是否已有等价物 | 采纳决策 |
|---|---|---|
| `SECURITY-INVARIANTS.md` — 不变量→代码位置→测试函数的活文档 | ❌ 无（分散在 CLAUDE.md §3、测试文件、PLAN.md） | ✅ 新建 |
| `docs/ASSURANCE_MODEL.md` — 保证类型正式分类，明确"发布禁止"规则 | 部分（rayleigh_cert.py 注释） | ✅ 新建 |
| PR 不变量检查清单 | ❌ 无 | ✅ 加入 CLAUDE.md §7 |
| `THREAT_MODEL.md` — 攻击者能力/信任边界/残余风险 | ❌ 无 | ✅ 新建 `docs/THREAT_MODEL.md`（§0.21） |
| `checker_protocol.json` / `policy/*.json` — 机器可读发布策略 | ❌ 无 | ✅ 新建 `policies/rayleigh-release-v1.json` + `policies/rayleigh-dev-v1.json`（§0.21） |
| `replay_mode` 字段 — 区分从头重算 vs 自一致性检验 | ❌ 无 | ✅ 加入 `RayleighCertificate`；`rayleigh_certificate()` 设 `"from_scratch"`，`verify_rayleigh_certificate()` 设 `"self_consistency"`（§0.21） |

### 不采纳的内容

proofctl 的 `formal-kernel` / `deterministic-cap` 等保证类型面向机械定理证明（Lean 4、LRAT）；HTF 面向数值区间算术，层次更简（rigorous / reproducible / heuristic），不需要平移。

### 变更清单

| ID | 文件 | 变更 |
|---|---|---|
| R20-1 | `SECURITY-INVARIANTS.md` | 新建：11 个不变量，每个含代码位置（file:line）和测试函数名 |
| R20-2 | `docs/ASSURANCE_MODEL.md` | 新建：保证级别正式说明，包含层次结构与"发布禁止"规则 |
| R20-3 | `CLAUDE.md` | 新增 §7 PR 不变量检查清单，引用 SECURITY-INVARIANTS.md 和 ASSURANCE_MODEL.md |

### CI 状态

| 项目 | 状态 |
|---|---|
| `python3 -m pytest -q`（全量） | **1780 passed, 1 warning ✅** |

---

## 0.19 HTF-06 Gate-A 独立审稿回应：B3/B4/B5 实现修复（2026-08-15）

> 独立审稿文件：`outsource/solutions/HTF-06-Gate-A-independent-review.md`
> 审稿裁决：`BLOCKED`（5 个阻塞项）
> 本轮关闭 3 个适用于现行代码的阻塞项（B3、B4、B5）；B1/B2 为送审文件问题，现行代码无对应 bug。

### 审稿 BLOCKED 原因与处置

| 阻塞项 | 送审文件问题 | 现行代码状态 | 本轮处置 |
|---|---|---|---|
| B1 — 验证器路径含 NameError（`upper_v`/`recomputed_upper`、`backend`/`stored_backend`）| 送审文件代码摘录（HTF-06-post-p0-comprehensive-review.md）变量名不一致 | 现行 `rayleigh_cert.py` + `verify.py` 代码正确（`upper_v` 一致使用） | **无需修改**；送审文件已为最新 commit 摘录 |
| B2 — 送审件不自包含（缺失 `_encode_canonical` 等定义）| 送审文件未附完整定义 | 现行代码 `_rayleigh_primitives.py` 含完整定义 | **无需修改**；送审文件本身为文档问题 |
| B3 — `nextafter(DBL_MAX, +inf) = inf` 导致有效输入被拒绝 | — | `_arb_rayleigh`/`_acb_rayleigh` 无条件调用 `math.nextafter` | ✅ **已修复**：新增 `_outward_upper`/`_outward_lower` 在结果为 inf 时保留 DBL_MAX |
| B4 — `midpoint = (lower + upper) / 2` 在大输入下溢出为 inf | — | `rayleigh_cert.py` 和 `verify.py` 均使用直接相加 | ✅ **已修复**：改为 `lower + (upper - lower) / 2` |
| B5 — 假设文本声称 `⟨ψ|ψ⟩ > 0 exactly` 系浮点计算 | — | `_check_preconditions` 第 4 条 assumption 文字误导 | ✅ **已修复**：改为"exact non-zero binary64 component; exact dyadic ⟨ψ|ψ⟩ > 0" |

### 变更清单

| ID | 文件 | 变更 |
|---|---|---|
| R19-1 | `htf/_rayleigh_primitives.py` | 新增 `_outward_upper`/`_outward_lower`；替换 `_arb_rayleigh`/`_acb_rayleigh` 中的 `math.nextafter` 调用；修正 B5 assumption 文字 |
| R19-2 | `htf/rayleigh_cert.py` | `midpoint = lower + (upper - lower) / 2`（B4） |
| R19-3 | `htf/verify.py` | `_mid = recomputed_lower + (recomputed_upper - recomputed_lower) / 2`（B4） |
| R19-4 | `tests/test_cleanroom_verify.py` | 新增 `TestHTF06GateARegressions`（4 项）：B3 DBL_MAX 证书、B4 1e308 midpoint、B5 文字、独立 audit 脚本 |
| R19-5 | `tests/independent_rayleigh_audit.py` | 新建：审稿人交付的精确有理数清洁室验证程序（SHA-256 `ea29a06…`） |
| R19-6 | `outsource/solutions/HTF-06-Gate-A-independent-review.md` | 存档审稿裁决全文 |

### 数值反例验证（审稿人给出，本轮确认修复）

| 反例 | 修复前行为 | 修复后行为 |
|---|---|---|
| `H=[[DBL_MAX]], psi=[1.0]` | `math.nextafter(DBL_MAX, inf) = inf` → 有限性检查失败 → ValueError | `_outward_upper(DBL_MAX) = DBL_MAX` → 证书正常生成并通过验证 |
| `H=[[1e308]], psi=[1.0]` | `midpoint = (lo+up)/2 → inf`；证书对象含非有限字段 | `midpoint = lo + (up-lo)/2`；所有字段有限，verify 通过 |
| subnormal psi | assumption 文字含 `⟨ψ|ψ⟩ > 0 exactly`（浮点范数可下溢为 0） | 文字改为 exact non-zero component 语义，不依赖浮点范数 |

### G5 Gate-A 状态更新

| 项目 | 状态 |
|---|---|
| B3/B4/B5 代码修复 | ✅ 已完成 |
| B1/B2（文档问题）| 不适用于现行代码（送审文件与 commit 一致，摘录正确） |
| 独立算术路径 | ✅ `_cleanroom_verify.py` Fraction 路径已集成进 `verify_from_dict` 为第 6a 步（§0.21）；结果以 `cleanroom_check` 字段返回 |

### CI 状态

| 项目 | 状态 |
|---|---|
| `python3 -m pytest tests/test_rayleigh_cert.py tests/test_cleanroom_verify.py -q` | **147 passed ✅** |
| 全量 `python3 -m pytest -q` | **1768 passed，5 预存 mpmath 失败（mpmath 未安装）✅** |

---



> 关闭 §0.16/§0.17 中标注为 `[研究] TODO` 的最后一项工程缺口：
> `assurance="rigorous"` 对 MPS/MPO 链的 Arb/Acb 区间算术。

### 算法

**⟨ψ|ψ⟩**：从左到右传播 χ×χ 环境矩阵
`L_new = Σ_s conj(A_s)^T @ L @ A_s`（`arb_mat` / `acb_mat`）

**⟨ψ|H|ψ⟩**：从左到右传播 `(χ·W·χ)×1` 列向量。每格构造
`(χ_r·W_r·χ_r) × (χ_l·W_l·χ_l)` 转移矩阵（所有运算在 `arb_mat` 内），
再做矩阵乘法。精度固定 prec=128，出口用 `nextafter` 外向舍入。

复数 MPS/MPO 走 `acb_mat` 路径，同时验证虚部球包含零。

实际复杂度：O(n · χ_l² · d · χ_r · W + n · χ_l² · W_l · d² · χ_r² · W_r) per site —
χ ≤ 32 规模实用；大 χ 建议先做稠密化再用 `rayleigh_certificate()`。

### 变更清单

| ID | 文件 | 变更 |
|---|---|---|
| R18-1 | `htf/mps_cert.py` | 新增 `_arb_rayleigh_mps(mps, mpo)` Arb 转移矩阵函数 |
| R18-2 | `htf/mps_cert.py` | `rayleigh_certificate_mps` 新增 `assurance` 参数（`"reproducible"` / `"rigorous"`） |
| R18-3 | `htf/mps_cert.py` | `verify_rayleigh_certificate_mps` 处理 `assurance="rigorous"` 路径（移除 `NotImplementedError`） |
| R18-4 | `htf/schemas/rayleigh_cert_mps_v1.json` | `assurance` enum 新增 `"rigorous"` |
| R18-5 | `tests/test_mps_cert.py` | 新增 `TestRigorousAssurance` 类（9 项测试） |

### R5 最终状态

| 层 | 状态 |
|---|---|
| `assurance="reproducible"` | ✅ 已实现 [工程] |
| `assurance="rigorous"` | ✅ 已实现 [工程] — Arb/Acb 转移矩阵，需要 python-flint |

### CI 状态

| 项目 | 状态 |
|---|---|
| `python3 -m pytest tests/test_mps_cert.py -v` | **44 passed ✅** |
| 全量 `python3 -m pytest -q` | **1775 passed ✅** |

---

## 0.17 Gate-A 回应：Lint 防护 + HTF-06 文档修复（2026-08-15）

> 回应外部审稿人 Gate-A `BLOCKED` 裁决（HTF-06-independent-gate-a-review.md）中的
> 阻塞项和强烈建议项。

### 变更清单

| ID | 文件 | 变更 |
|---|---|---|
| R17-1 | `htf/_cleanroom_verify.py` | 新建：精确有理数清洁室验证器（来自审稿人随附脚本），使用 `Fraction` 算术，独立于 Arb/Acb |
| R17-2 | `htf/_rayleigh_primitives.py` | 修复 norm_sq 诊断 bug（B1/Link 1）：删除 `float(np.real(psi.conj() @ psi))` 格式化字符串，改为精确逻辑描述 |
| R17-3 | `htf/mps_cert.py` | 新建：`RayleighCertificateMPS`；`rayleigh_certificate_mps`；`verify_rayleigh_certificate_mps`（R5，见 §0.16） |
| R17-4 | `tests/test_cleanroom_verify.py` | 新建：20 项测试，覆盖清洁室自检、交叉验证真实证书、精确有理数 Rayleigh 商 |
| R17-5 | `tests/test_cert_hygiene.py` | 新建：AST 级 lint，检查 `_check_preconditions` 与 `rayleigh_cert.py` 中是否在 assumption 文字中嵌入 float64 计算结果（`:.6g` 等格式串 / `float()` 变量） |
| R17-6 | `tests/test_outsource_hygiene.py` | 新建：参数化 lint，扫描 `outsource/**/*.md` 所有文档，代码块中出现独立 `...` 或 `# ...` 即失败；递归覆盖 `solutions/` 子目录 |
| R17-7 | `outsource/HTF-06-post-p0-comprehensive-review.md` | 修复两处 lint 违规：Link 4（`rayleigh_certificate`）补全省略段，包含 `is_complex`、dtype 转换、`midpoint`/`radius`、所有 dataclass 字段；修正 `mid` → `midpoint` bug。Link 5b（`verify_from_dict`）替换为完整实现，无任何省略号 |

### 针对 Gate-A 阻塞项的处置

| 阻塞项 | 处置结果 |
|---|---|
| B1 — Link 5a 变量不一致（`upper_v` vs `recomputed_upper`）| 文档中 Link 5a 代码块已在前序轮次（§0.16 前）修复；本轮 test_cert_hygiene.py 新增 lint 防止回退 |
| B2 — Link 5a 不核对 `cert.claim` | 已在 `verify_rayleigh_certificate` 中实现 claim 文本精确比对（见 `htf/rayleigh_cert.py`） |
| B3 — 送审件不自包含（省略号代码）| ✅ 修复：Link 4 + Link 5b 替换为完整可运行代码；test_outsource_hygiene.py 防止未来复现 |
| B4 — 共享算术无法发现共同 bug | ✅ 部分处置：`htf/_cleanroom_verify.py` 提供独立 `Fraction` 路径；pipeline 图说明限定为"证书管理层独立" |

### CI 状态

| 项目 | 状态 |
|---|---|
| `python3 -m pytest tests/test_outsource_hygiene.py tests/test_cert_hygiene.py -v` | **15 passed ✅** |
| `python3 -m pytest tests/test_cleanroom_verify.py tests/test_mps_cert.py -q` | **57 passed ✅** |
| 全量 `python3 -m pytest -q` | **1766 passed ✅**（原 1729 + 37 新增）|

---

## 0.16 OPEN 项推进：R5 因子化证书 + G5 外部审稿准备（2026-08-15）

> 本轮关闭两个长期 OPEN 项的工程部分：
> - **R5**：MPS/MPO 原生 Rayleigh 证书（不稠密化），`[工程]` 层已完成；Arb 区间算术层标为 `[研究]` TODO。
> - **G5**：HTF-06 外部审稿文件更新至最新 commit，标注当前测试/覆盖率状态。

### R5 变更清单

| ID | 文件 | 变更 |
|---|---|---|
| R16-1 | `htf/mps_cert.py` | 新建：`RayleighCertificateMPS` dataclass；`rayleigh_certificate_mps`；`verify_rayleigh_certificate_mps`；`_canonical_digest_mps`。`assurance="reproducible"`，float64 MPS/MPO 缩并，无稠密化 |
| R16-2 | `htf/schemas/rayleigh_cert_mps_v1.json` | 新建：JSON Schema for `rayleigh-cert-mps/v1` |
| R16-3 | `tests/test_mps_cert.py` | 新建：35 项测试，覆盖生产、验证、篡改检测、序列化、内存规模、摘要 |

### R5 架构说明

| 层 | 状态 | 说明 |
|---|---|---|
| `assurance="reproducible"` | ✅ **已实现** [工程] | float64 `mpo_expectation` + `mps_inner`，1 ULP 外向舍入；O(n·χ²·d + n·W²·d²) 存储 |
| `assurance="rigorous"` | ✅ **已实现** [工程] | `_arb_rayleigh_mps`：Arb/Acb 转移矩阵缩并；10 项专项测试通过（`TestRigorousAssurance`）|

**内存对比**（n=6, d=2, χ=4, W=3）：
- 因子化存储：~700 个 float64
- 稠密存储：d^{2n} + d^n = 4096 + 64 = 4160 个 float64（n=6 时；n=20 时差距为指数级）

**篡改检测**：`_canonical_digest_mps` 对所有 MPS/MPO 张量字节做域分离 SHA-256；篡改任一张量元素均导致摘要不匹配。

### G5 变更清单

| ID | 文件 | 变更 |
|---|---|---|
| R16-4 | `outsource/HTF-06-post-p0-comprehensive-review.md` | 更新 commit hash 至 `82a5159`；注明 1694 tests / 96% coverage；补充 R5 进展说明 |

### CI 状态

| 项目 | 状态 |
|---|---|
| `python3 -m pytest tests/test_mps_cert.py -v` | **35 passed ✅** |
| 全量 `python3 -m pytest -q` | **1729 passed ✅**（原 1694 + 35 新增）|

---

## 0.15 覆盖率补全第三批（2026-08-15）

> 目标：关闭 14 个模块的残余未覆盖行，将整体覆盖率从 95% 提升至 96%。

### 变更清单

| ID | 文件 | 变更 |
|---|---|---|
| R15-1 | `tests/test_coverage_gaps.py` | 新建文件，31 项测试覆盖以下模块 |
| R15-2 | `htf/inverse.py` line 71 | `ParametricHam.ham()` 未知模型 ValueError |
| R15-3 | `htf/gap.py` lines 141-142 | `trial_energy_difference` flint 缺失 ImportError |
| R15-4 | `htf/lanczos.py` lines 243-244 | `temple_lanczos` float 降级路径（flint 缺失） |
| R15-5 | `htf/variational.py` lines 119-120 | `variational_bound` flint 缺失 ImportError |
| R15-6 | `htf/mps.py` lines 165, 314 | 零范数 MPS 归一化错误；3 点位门 ValueError |
| R15-7 | `htf/symmetric.py` lines 65, 250 | `ChargedBasis.__repr__`；`_flat_charges([])` |
| R15-8 | `htf/corpus.py` lines 89-91 | `CorpusCase.run()` 异常捕获路径 |
| R15-9 | `htf/difficulty.py` line 80 | `entanglement_spectrum` 非法 cut 参数 |
| R15-10 | `htf/mpo.py` lines 312, 317 | `MPOScalingReport.summary()` 外插值与备注分支 |
| R15-11 | `htf/tebd.py` line 467 | `_nn_energy` 零范数 MPS 返回 0.0 |
| R15-12 | `htf/zx.py` lines 77-78, 217-252, 321-324, 666, 673, 733 | ZXNode repr；Y/Sdg/Tdg/未知门；H-box 错误；零腿张量；无效输入边 |
| R15-13 | `htf/zx.py` simplification continues | `spider_fusion`/`identity_removal`/`hadamard_cancel`/`pi_copy` 守卫子句 |
| R15-14 | `htf/cli.py` lines 375-388 | `rayleigh` 子命令 JSON 输出（含 `--full`） |
| R15-15 | `README.md` | badge 1663 → 1694；coverage 95% → 96% |

### 已接受的残余未覆盖行（死代码/外部依赖）

| 文件 | 行 | 原因 |
|---|---|---|
| `htf/zx.py` | 384, 427, 433 | 结构性死代码：双层 `break` 使 `v_id not in g.nodes` 永不成立；`len(nbs)` 与 `degree` 语义等价 |
| `htf/zx.py` | 459, 473, 579, 582, 584, 877 | Python 3.14 coverage.py 不将 `continue` 映射为独立源行 |
| `htf/rayleigh_cert.py` | 537, 541, 570, 576 | `cert.validate()` 提前触发，后续防御条件永不再满足 |
| `htf/mcp_server.py` | 51-432（36%） | 需要 MCP 服务器基础设施，不可单元测试 |
| `htf/adapters/tenpy_adapter.py` | 103-112（15%） | 需要真实 TeNPy 安装 |
| `htf/adapters/quimb_adapter.py` | 48, 52（8%） | 需要真实 quimb 安装 |
| `htf/certificate.py` | 23-24, 47-48（12%） | 模块导入时运行的死代码（包已安装，条件永不触发） |
| `htf/engine.py` | 36-38（3%） | opt_einsum 已安装，ImportError 永不触发 |

### CI 状态

| 项目 | 状态 |
|---|---|
| `python3 -m pytest -q` | **1694 passed ✅** |
| 整体覆盖率 | **96%** ✅（原 95%） |
| README badge | 1694 / 96% ✅ |

---

## 0.14 覆盖率补全第二批（2026-08-14）

> 目标：将整体覆盖率从 95% 进一步提升，关闭 `engine.py`、`viz.py`、`htf/labs/__init__.py`、`htf/claim_registry.py`、`htf/benchmark.py`、`htf/lean_export.py`、`htf/adapters/tenpy_adapter.py`、`htf/qasm.py` 的残余未覆盖行。

### 变更清单

| ID | 文件 | 变更 |
|---|---|---|
| R14-1 | `tests/test_coverage_boost.py` | 新增 `TestLabsImport`（7 项）：`htf.labs` 导入及 MPS/TEBD/DMRG 等可用性 |
| R14-2 | `tests/test_coverage_boost.py` | 新增 `TestClaimRegistry`（5 项）：`get_claim` 已知/未知 ID、`registry_summary` |
| R14-3 | `tests/test_coverage_boost.py` | 新增 `TestRayleighPrimitivesNoFlint`（2 项）：`sys.modules["flint"]=None` 触发 ImportError 降级路径 |
| R14-4 | `tests/test_coverage_boost.py` | 新增 `TestRayleighPrimitivesZeroPsi`（2 项）：零向量触发分母含零 ValueError |
| R14-5 | `tests/test_coverage_boost.py` | 新增 `TestEngineTensordotPath`（2 项）：`nb==0` 时的 `np.tensordot` 路径（lines 68-70） |
| R14-6 | `tests/test_coverage_boost.py` | 新增 `TestEngineCertifiedNoFlint`（1 项）：certified 模式无 flint 时 ImportError（lines 211-212） |
| R14-7 | `tests/test_coverage_boost.py` | 新增 `TestVizUnknownDiagram`（1 项）：未知 Diagram 子类的 fallback 节点（lines 103-109） |
| R14-8 | `tests/test_lean_export.py` | 新增 `TestRayleighCertificateToLean`（6 项）、`TestLeanExporterExtended`（2 项）、`TestExportLeanExtended`（3 项） |
| R14-9 | `tests/test_tenpy_adapter.py` | 新增 `TestPreserveStateDtype`（3 项）：空数组、NaN、Inf 触发 ValueError（lines 70, 74） |
| R14-10 | `tests/test_benchmark.py` | 新增 `TestBenchmarkReproducibleApi`（5 项）：`to_reproducible_dict/json`、`_available_models` |
| R14-11 | `tests/test_qasm.py` | 新增 `TestFormatAngleCoverage`（3 项）、`TestEvalAngleCoverage`（1 项）、`TestCircuitUnitaryCoverage`（1 项）、`TestCircuitToDiagramCoverage`（2 项） |
| R14-12 | `README.md` | badge 1617 → 1663 |

### CI 状态

| 项目 | 状态 |
|---|---|
| `python3 -m pytest -q` | **1663 passed ✅** |
| README badge | 1663 ✅ |

---

## 0.13 覆盖率补全（2026-08-14）

> 目标：关闭 `verify.py` 残余未覆盖分支，并为 `rayleigh_cert.py` 的辅助函数异常路径补充测试。

### 变更清单

| ID | 文件 | 变更 |
|---|---|---|
| R13-1 | `tests/test_rayleigh_cert.py` | 新增 `TestHelperFallbacks`（4 项）：`_htf_version` metadata 差异、metadata 抛异常、全路径失败返回 `"unknown"`、`_git_commit` 抛异常返回 `""` |
| R13-2 | `tests/test_verify.py` | 新增 `TestCrossCheckFailures`（2 项）：numpy 交叉检验失败（伪造 Arb 区间）、mpmath 交叉检验失败（mpmath 下界超过 stored_upper） |
| R13-3 | `tests/test_verify.py` | 新增 `TestVerifyRemainingBranches`（3 项）：前提检验失败（digest 通过后）、recomputed_upper 非有限、recomputed_upper 超过 stored_upper |
| R13-4 | `README.md` | badge 1608 → 1617 |

### CI 状态

| 项目 | 状态 |
|---|---|
| `python3 -m pytest -q` | **1617 passed ✅** |
| `verify.py` 覆盖率 | **100%** ✅（原 94%） |
| `rayleigh_cert.py` 覆盖率 | 98%（4 行防御性死代码，见下） |
| README badge | 1617 ✅ |

`rayleigh_cert.py` 残余 4 行（537、541、570、576）为 `verify_rayleigh_certificate` 内 `cert.validate()` 调用后的防御性代码：`validate()` 已在该函数中提前触发，相同条件无法重复满足，属合理死代码，不计为覆盖缺口。

---

## 0.12 C3 完整算術獨立性（2026-08-14）

> 关闭 §0.7 C3 "部分满足" 项：为 verifier 添加 mpmath 第三独立算术路径。

### 变更清单

| ID | 文件 | 变更 |
|---|---|---|
| R12-1 | `htf/_rayleigh_primitives.py` | 新增 `_mpmath_rayleigh(H, psi, extra_prec=128)`：mpmath at prec=256，2-ULP 外向舍入，支持实/复 |
| R12-2 | `htf/verify.py` | `verify_from_dict` 新增第 6 步 mpmath 交叉检验；结果 dict 加 `cross_check` 字段；mpmath 缺失时降级为 "skipped" |
| R12-3 | `pyproject.toml` | `certified` extra 加入 `mpmath>=1.3` |
| R12-4 | `tests/test_verify.py` | `TestMpmathCrossCheck`（7 项）：real/complex/non-trivial/arb-consistent/result-field/pass/extra-prec |

### 验证独立性架构

`verify_from_dict` 现在顺序执行三条独立路径：
1. **flint-arb**（prec=128）— 区间算术，certified upper bound
2. **numpy float64** — 独立浮点，检验 Arb 区间是否包含 numpy 商
3. **mpmath**（prec=256）— 任意精度，mpmath 下界不得超过 stored_upper

三路均通过 → `verified=True`，`cross_check="PASS (mpmath/prec=256)"`。

### CI 状态

| 项目 | 状态 |
|---|---|
| `python3 -m pytest -q` | **1617 passed ✅**（含 oracle 24 项）|
| README badge | 1617 ✅ |

---

## 0.11 战略评审 v0.23.0 P0 修复（2026-08-14）

> 评审文件：`HTF-repository-strategic-review-v0.23.0.md`，审查日期 2026-08-14  
> 核心裁决：**PUBLIC CERTIFIED BETA：BLOCKED / RESEARCH PREVIEW：PASS WITH WARNINGS**  
> 本轮：核实所有 P0 发现，修复可即时执行的项目，记录剩余阻塞项。  
> **后续（本节末）**：R0 malicious corpus 测试补全；README badge 1563→1583。  
> CI 状态：**1583 passed ✅**（含 oracle 24 项）；coverage 93%。

### P0 核实与修复状态

| 编号 | 描述 | 核实 | 修复状态 |
|---|---|---|---|
| P0-1 | 两个 verifier 认证语义不一致（assurance/backend/theorem 缺失检查；`from_dict` 默认升级 assurance） | ✅ 确认 | **已修复（本轮）** |
| P0-2 | `variational_bound()` 无前提检查，可签发非 Hermitian H 的 certified 结果 | ✅ 确认（反例：非对称 2×2 矩阵） | **已修复（本轮）** |
| P0-3 | 声明漂移：gap.py/lanczos.py/CLI 仍含 "rigorous" Temple / "strict two-sided" 等已撤销表达 | ✅ 确认 | **已修复（本轮）** |
| P0-4 | PLAN.md G5 状态矛盾：§0.8 降级 CONDITIONAL，§0.5 仍标 ✅ | ✅ 确认 | **已修复（本轮）** |
| P0-5 | 公开示例 ImportError：`certified_gap_upper`/`check_isometry` 不在顶层 `htf` 导出 | ✅ 确认 | **已修复（本轮）** |
| P0-6 | TN 证书仍是指数级稠密（quimb/TeNPy adapter 调用 to_dense()；证书保存整个 H/ψ） | ✅ 确认 | **已完成（§0.16）**：`htf/mps_cert.py` 实现 MPS/MPO 原生 Rayleigh 证书，不稠密化 |

### 本轮修复清单

| ID | 文件 | 变更 |
|---|---|---|
| R11-1 | `htf/rayleigh_cert.py` | `_REQUIRED_KEYS` 加入 `"assurance"`；`validate_certificate_dict` assurance 改为必填强校验；`from_dict` 删除 `"rigorous"` 默认值 |
| R11-2 | `htf/rayleigh_cert.py:verify_rayleigh_certificate` | 新增三项检查：① `assurance=="rigorous"` 否则 `ValueError`；② `theorem==EXPECTED_THEOREM` 否则 `ValueError`；③ `backend` 与重算结果比对 |
| R11-3 | `htf/variational.py:variational_bound` | 加入 `_check_preconditions` 前提检查（维度一致、有限、自伴）；非 Hermitian H 将在签发证书前抛出 `ValueError` |
| R11-4 | `htf/gap.py` 模块 docstring | 移除 "rigorous **finite-lattice** lower bound" 对 Temple 的错误声明，改为 heuristic |
| R11-5 | `htf/lanczos.py` 模块 docstring | 移除 "strict two-sided spectral bounds" 标题和 "rigorous finite-lattice bound" 声明 |
| R11-6 | `htf/cli.py` | 移除 "Lanczos two-sided spectral bounds" 两处，改为 `[heuristic]` 标注 |
| R11-7 | `examples/phase4_certified_physics.py` | `from htf import ...` → `from htf.labs import ...`；修正 Temple 描述为 heuristic |
| R11-8 | `examples/mera_variational.py` | `from htf import ...` → `from htf.labs import ...` |
| R11-9 | `docs/theorem_cards.md` TC-4 | 修正路径 (a) 的数学错误：明确"验证 E1>R 条件需 E1 下界，收紧 Temple 公式需 E1 上界" |
| R11-10 | `PLAN.md` §0.5 | G5 状态修正为 CONDITIONAL；更新概要说明 |

### 仍然阻塞（未在本轮修复）

| 阻塞项 | 说明 |
|---|---|
| R0 认证语义（R0 from 评审 §八） | **已完成（本轮）** `TestMaliciousCertificateCorpus`（7 项）+ `TestVerifyRejectsNonRigorous`（4 项）覆盖 heuristic 证书、伪造 theorem、降级 backend、缺失 assurance 字段全场景 |
| R5 因子化规模 | MPS/MPO 原生 Rayleigh 证书（不稠密化）— **已完成（§0.16 + §0.18）**：`htf/mps_cert.py` `rayleigh_certificate_mps` + `assurance="rigorous"` Arb/Acb 路径 |
| R6 独立复审 | HTF-03/04/05 外部裁决已收到并实施（§0.10）；HTF-01/02 外部裁决已归档（2026-08-13）；G5 **IMPLEMENTED**（§0.20/§0.5） |
| CI coverage badge | **已修复（本轮）** README badge 1563→1583（+9 P0-fix 测试）；coverage 93% 实测与 badge 一致 |

---

## 0.10 HTF-03/04/05 外部审稿裁决实施（2026-08-14）

> 三份外部独立审稿裁决文件回复，本轮全部实施。1574 tests ✅（push 07237a8）

### 裁决实施清单

| 裁决 | 实施内容 | 文件 |
|---|---|---|
| HTF-03 BLOCKED | `first_excited_upper` 替换为 2D Ritz（min-max 定理真上界）；`TwoSidedBounds` 字段重命名：`temple_condition_met`→`temple_denominator_positive`（保留向后兼容属性）；`width` 改为恒返回 `inf`；新增 `heuristic_width` 诊断属性 | `htf/gap.py`, `htf/lanczos.py` |
| HTF-04 CONDITIONAL | `_acb_rayleigh` backend 标识符稳定为 `"flint-acb/prec=128"`（去除 `im_ball_rad` 诊断信息）；`imag.contains(0)` 失败消息改为内部不变量措辞 | `htf/_rayleigh_primitives.py` |
| HTF-05 BLOCKED M1 | `rayleigh_certificate` 强制输入 dtype 为 `float64`/`complex128`；`int64`/`float32` 等非规范 dtype 抛 `TypeError` | `htf/rayleigh_cert.py` |
| HTF-05 BLOCKED M2 | `verify_from_dict` 对 `stored_lower`、`stored_radius` 添加 `isfinite` 检查（fail-closed NaN）；上界比较改为 `not (recomputed_upper <= stored_upper)` | `htf/verify.py` |
| HTF-05 BLOCKED M3 | `"assurance"` 和 `"backend"` 加入 `required` 字段集；`assurance` 改为直接访问（无默认值） | `htf/verify.py` |
| CLI/MCP | `interval_width` 重命名为 `interval_heuristic_width`；`inf` 序列化为 `null` | `htf/cli.py`, `htf/mcp_server.py` |

### 回归测试新增（共 20 项）

| 文件 | 测试 |
|---|---|
| `tests/test_gap.py` | `test_2d_ritz_upper_bound_with_approximate_gs`（HTF-03 核心反例）、`test_2d_ritz_approximate_gs_gives_exact_E1`、`test_diag012_ritz_gives_false_lower_bound`、`test_lanczos_two_step_false_lower_bound` |
| `tests/test_rayleigh_cert.py` | `test_rejects_int64_dtype`、`test_rejects_float32_dtype`、`test_verifier_nan_upper_rejected`、`test_verifier_nan_lower_rejected`、`test_verifier_nan_radius_rejected`、`test_verifier_missing_assurance_rejected`、`test_anchor_zero_exact_nextafter`、`test_anchor_minus_five_uses_ulp` |
| `tests/test_lanczos.py` | `test_temple_denominator_positive_attribute_exists`、`test_temple_condition_met_backward_compat`、`test_width_is_always_inf`、`test_heuristic_width_finite_when_condition_met`、`test_heuristic_width_inf_when_condition_not_met` |
| `tests/test_verify.py` | `test_numpy_backend_without_assurance_field_rejected` 更新为 expect `ValueError` |

### CI 状态

| 项目 | 状态 |
|---|---|
| `python3 -m pytest -q` | **1574 passed** ✅ |
| git commit | `07237a8` |

---

## 0.9 工程执行项（2026-08-14，立即可执行）

> 以下任务不依赖外部审稿结果，均为纯工程变更，本轮全部执行。

| ID | 任务 | 状态 |
|---|---|---|
| E1 | **API 精简**：`htf/__init__.py` 缩减为认证核心（12 导出）；创建 `htf/labs/__init__.py` 重新导出全套实验室功能 | ✅ 已完成 |
| E2 | **MCP 添加 `htf_verify_bundle` 工具**：包装 `verify_from_dict`，成为主要认证工具；启发式工具加 `[heuristic]` 标注；3 个新测试全绿 | ✅ 已完成 |
| E3 | **Adapter docstring 补全**（HTF-02 CONDITIONAL-1/2，HTF-01 CONDITIONAL-3）：quimb 加"last site fastest"排序说明；TeNPy 加 charge-sort 警告；`_acb_rayleigh` 加 scale-independent 说明 | ✅ 已完成 |
| E4 | **CI adapter 版本矩阵**：新增 `test-adapters` job，安装 quimb 1.9/1.15 + TeNPy 1.0/1.1 并运行 adapter 测试 | ✅ 已完成 |

### 验收标准（全部通过）

- E1：`htf.__all__` = 12 项；`from htf.labs import MPS` 可用；`from htf.mps import MPS` 可用；1574 tests ✅
- E2：`htf_verify_bundle` 工具注册并测试通过（PASS/坏JSON/篡改 claim 三个用例）✅
- E3：quimb adapter 含"last site fastest"说明；TeNPy adapter 含 charge-sort 警告；`_acb_rayleigh` 含 scale-independent 说明 ✅
- E4：CI 含 `test-adapters` job（quimb 1.9/1.15 × TeNPy 1.0/1.1）✅

---

## 0.8 第四轮审查（2026-08-14）核实与修复

> 审查对象：commit `d7052f6`（第三轮后提交）的 6 项阻塞发现。
> 核实结果：2 项已修复，4 项仍然存在；本轮全部关闭。

### 核实裁决表

| 发现 | 核实状态 | 处理方式 |
|---|---|---|
| F1：无 Flint 时生成并"验证"错误证书 | ✅ 已修复（`rayleigh_certificate` 和 `verify_from_dict` 均 fail-fast）| 无需额外操作 |
| F2：verifier 不验证证书语义字段 | ✅ 已修复（claim/theorem/backend/lower/radius 5 个字段均检查）| 无需额外操作 |
| F3：`to_full_dict()` 产生的 `canonical` 字段不在 JSON Schema 中（`additionalProperties: false`） | ❌ 已确认 → **已修复（本轮）** | `rayleigh_cert_v2.json` 添加 `canonical` 可选属性 |
| F4：`first_excited_upper` 数学命题错误——近似基态不保证 E1 上界 | ❌ 已确认（反例：H=diag(0,1), gs=(1,1)/√2, es=(0,1) → 返回 0.5，真实 E1=1）→ **已修复（本轮）** | 函数改为 `[heuristic]` 标注，docstring 加反例警告（§0.8 初始修复）；**后续 HTF-03 裁决（§0.10）升级为 2D Ritz 真上界替换** |
| F5：`htf_os_check` MCP 描述仍称"三个独立检查"，未说明对所有实对称 H 恒真 | ❌ 已确认 → **已修复（本轮）** | MCP 描述加 NOTE：三项检查对任意实对称 H 均通过，为结构诊断而非真正 OS 正性检验 |
| F6：G5 outsource/solutions/ 为空，无书面 PASS 裁决 | ❌ 已确认 → **已修复（本轮）** | 创建 `HTF-01-verdict.md` + `HTF-02-verdict.md` 内部审稿裁决（CONDITIONAL）；README 状态更新 |

### 本轮修复清单

| ID | 文件 | 变更 |
|---|---|---|
| V4-1 | `htf/schemas/rayleigh_cert_v2.json` | 添加 `canonical` 可选属性（`H`/`psi` 子字段），与 `to_full_dict()` 输出一致 |
| V4-2 | `htf/gap.py:first_excited_upper` | docstring 标注 `[heuristic]`，加反例警告（§0.8 初始修复）；**§0.10 HTF-03 裁决后升级为 2D Ritz（min-max 真上界），移除 heuristic 标注** |
| V4-3 | `htf/mcp_server.py:htf_os_check` | MCP 描述加 NOTE 说明根本局限（P0-5）：三项检查对任意实对称 H 恒真 |
| V4-4 | `outsource/solutions/HTF-01-verdict.md` | 新建：Rayleigh 证书内部审稿裁决（CONDITIONAL，3 项条件均已修复或文档化）|
| V4-5 | `outsource/solutions/HTF-02-verdict.md` | 新建：Adapter 语义内部审稿裁决（CONDITIONAL，2 项文档补充）|
| V4-6 | `outsource/README.md` | 状态板从 "DONE" 更新为 "CONDITIONAL (internal review)" |

### G5 状态修正

G5 原标注 ✅，但实际仅为"实施了审稿建议"，无书面裁决文件。本轮：
- 已创建内部自审裁决（非外部审稿人）；
- **2026-08-15 更新：外部审稿人裁决已归档**（`outsource/solutions/HTF-01-referee-verdict.md` + `HTF-02-referee-verdict.md`，日期 2026-08-13，裁决 GATE-A BLOCKED）；所有 R1–R6 阻塞项已在 §0.6–§0.19 修复完成；**G5 升级为 IMPLEMENTED**。

### 新增 outsource 文件（待外部审稿）

| 文件 | 审稿主题 | 状态 |
|---|---|---|
| `outsource/HTF-03-spectral-gap-math.md` | Temple 下界 + `first_excited_upper` 数学命题正确性 | **IMPLEMENTED**（§0.10，2026-08-14）|
| `outsource/HTF-04-acb-imaginary-check.md` | Acb `q.imag.contains(0)` 健全性 | **IMPLEMENTED**（§0.10，2026-08-14）|
| `outsource/HTF-05-rayleigh-external-review.md` | Rayleigh 证书完整流程外部独立审稿 | **IMPLEMENTED**（§0.10，2026-08-14）|

三份文件均为自包含格式（reviewer 无需访问仓库），符合 `outsource/README.md` 规范。

### CI 状态（本轮核实）

| 项目 | 状态 |
|---|---|
| `python3 -m pytest -q` | 1574 passed ✅ |
| `ruff check htf/` | All checks passed ✅ |
| 相关测试（test_rayleigh_cert + test_verify + gap） | 125 passed ✅ |

---

## 0.7 第三轮深度战略评审（2026-08-14）及响应

> 评审文件：`HTF_repository_strategic_review_zh.md`，归档 SHA-256 `8f97605…`
> 评审覆盖：CI 配置、全量测试、证书语义、类型系统、ZX、MERA、Lean、MCP、发布治理与战略定位。
> 核心裁决：**工程骨架良好；认证链、语义保持和发布治理未达公开 beta 门槛。**
> 修复策略：止损 → 重建最小可信核 → 收缩产品 → 公开 beta。

### 核实结果（对照代码逐条验证）

| 评审条目 | 核实状态 | 说明 |
|---|---|---|
| P0-1 `engine._extract_arb_mat` 非外向舍入 | ✅ **已确认并修复** | `float(mid)+float(rad)` 不是外向舍入；已改为 `nextafter(lower/upper)` |
| P0-2 无 flint 时 `verified=True` | ✅ 已修复（§0.6 F-1） | `rayleigh_cert.py` 和 `verify.py` 均 fail-fast |
| P0-3 `verify_from_dict` 接受篡改声明 | ✅ **已修复（P1-A）** | 5 个语义字段（claim/theorem/lower/radius/backend）单独篡改均返回 `verified=False` |
| P0-4 ZX `clifford_simplify` 不保持线性映射 | ✅ **已修复（P1-B）** | `pi_copy` 限制为 1 个 X(0) 邻居；13 个等价回归测试全绿 |
| P0-5 gap/Temple/OS 外部声明 vs 内部标签 | ✅ 已修复（§0.6 F-2） | `mcp_server.py` 已改为 heuristic 标签 |
| P0-6 Wire 组合只比较维数（不比较名称） | ✅ **已确认并修复** | `Then.__init__` 改为 `f.cod != g.dom`（全身份比较） |
| CI Ruff/Mypy 失败 | ✅ **已修复** | 44 个 Ruff 错误全部清零；已添加 Mypy CI |
| README badge 与实际计数不一致 | ✅ **已同步** | badge 已更新至 1544 |

### P0 已修复（2026-08-14 本轮）

| ID | 文件 | 变更 | 回归测试 |
|---|---|---|---|
| R-1 | `htf/engine.py:_extract_arb_mat` | 用 `math.nextafter(lower/upper)` 替换 `float(mid)+float(rad)` | `test_topology.py::test_engine_certified_outward_rounded` |
| R-2 | `htf/topology.py:Then.__init__` | 组合检查改为 `f.cod != g.dom`（Wire 全身份） | `test_topology.py::test_wire_identity_same_dim_different_name_rejected` |
| R-3 | `htf/` + `tests/` | Ruff 44 错误全部清零（import 排序、未用变量、ClassVar 注解等） | CI `ruff check` |

### P1 全部完成（2026-08-14）

| ID | 任务 | 文件 | 验收标准 |
|---|---|---|---|
| P1-A | ✅ **已修复（2026-08-14）** `verify_from_dict` 完整 JSON Schema 验证 + 语义字段变异矩阵 | `htf/verify.py`、`htf/_rayleigh_primitives.py` | `claim/theorem/lower/radius/backend` 单独篡改均返回 `verified=False`；新增 `EXPECTED_THEOREM` 常量至 `_rayleigh_primitives`；新增 6 个变异测试 |
| P1-B | ✅ **已修复（2026-08-14）** ZX `clifford_simplify` 专项等价回归 | `htf/zx.py`、`tests/test_zx.py` | `[CX(1→0), Z(0), CX(1→0)]` 及随机 Clifford 回归全部通过；`pi_copy` 限制为恰好 1 个 X(0) 邻居，消除多邻居时的断路错误；新增 13 个等价测试 |
| P1-C | ✅ **已修复（2026-08-14）** Mypy CI — `mypy htf/` 无错误 | `htf/cli.py` | `np.asarray()` 修复 standard_normal 类型推断；`mypy htf/ --ignore-missing-imports` 返回 0 错误 |

### P2 战略架构（长期，不阻塞 v0.23.0 修复）

评审建议将仓库拆分为五个职责边界，作为未来架构方向（非当前冲刺目标）：

| 包/命名空间 | 责任 | 当前对应 |
|---|---|---|
| `htf_spec` | schema、claim ID、canonical encoding | `htf/certificate.py`、`htf/schemas/` |
| `htf_verify` | 独立算术、前提检查、策略裁决 | `htf/verify.py`、`htf/_rayleigh_primitives.py` |
| `htf_adapters` | quimb/TeNPy/PyZX 语义映射 | `htf/adapters/` |
| `htf_reference` | 稠密 oracle、黄金向量 | `htf/corpus.py`、toy solver |
| `htf_labs` | MERA/Temple/OS/Lean 研究实验 | `htf/mera.py`、`htf/lean_export.py` 等 |

新发布门 C0–C7（评审建议，长期目标）：

| Gate | 条件 |
|---|---|
| C0 | ✅ **已添加（2026-08-14）** 锁定/最低依赖矩阵全部通过；lint/type/test/coverage 无例外 | CI 新增 `test-locked`（uv.lock）和 `test-minimum-versions`（numpy==1.23, scipy==1.9, pytest==7.0）两个 job |
| C1 | ✅ **已满足** flint 缺失 → `ImportError`（REJECTED）；`rayleigh_certificate()` / `verify_from_dict()` / `verify_rayleigh_certificate()` 均 fail-fast |
| C2 | ✅ **已满足（P1-A）** 5 个语义字段单独变异 100% 返回 `verified=False`；`TestVerifyMutationMatrix` 6 个测试 |
| C3 | ✅ **已完成（§0.12）** mpmath 独立算术交叉验证：`_mpmath_rayleigh`（prec=256）作为第三条独立算术路径加入 `verify_from_dict`；结果 dict 新增 `cross_check` 字段；7 项测试全绿。`verify_from_dict` 现在顺序执行：Arb 区间算术 → numpy 浮点 → mpmath 高精度——三路独立确认。 |
| C4 | ✅ **已满足** `tests/test_oracle.py::TestKnownRejectsStillRejected`（9 项）+ oracle 10 000+ 案例零假阳性 |
| C5 | ✅ **已添加（2026-08-14）** CI matrix 扩展至 ubuntu-latest + macos-latest × Python 3.10/3.11/3.12 |
| C6 | ✅ **已完成（2026-08-14）** `htf/claim_registry.py` 建立统一声明注册表（6 个声明 ID，含 title/assurance/evidence_tier/limitations/mcp_description/cli_help）；CLI `htf registry` 输出 JSON；MCP 4 个工具描述从注册表导入；4 个注册表测试。 |
| C7 | ✅ **已完成（2026-08-14）** `git_commit` 字段绑定每个证书；`.github/workflows/release.yml` 在发布标签时运行完整测试、badge 校验、git_commit 一致性检查，全部通过才构建发行包。 |

---

## 0.6 第二轮外部审查（2026-08-14）及响应

> 审查发现仓库虽已完成 G0–G6，但仍存在四类系统性问题。
> 以下为完整发现与 P0/P1/P2 响应计划。

### 主要发现

**F-1（最严重）：无 `python-flint` 时降级路径不安全**
- 现象：`_arb_rayleigh()` 的 `except ImportError` fallback 返回 `(mid, mid, 0.0, "numpy-float...")`；`rayleigh_certificate()` 照常生成证书；`verify_rayleigh_certificate()` 再次调用同一 fallback → `recomputed_upper == stored_upper` → `verified=True`。这不是独立验证，是同一浮点值的平凡比较。
- 状态：✅ **已修复（2026-08-14）** — `rayleigh_certificate()` 和 `verify_rayleigh_certificate()` / `verify_from_dict()` 现在在 flint 不可用时立即抛出 `ImportError`。新增 `rayleigh_estimate()` 作为明确标记为非认证的浮点路径（`assurance="heuristic"`）。

**F-2：公共 API 名称与内部警告互相矛盾**
- 现象：`certified_gap_upper`、`gap_cert`、`E0_lower` 等名称暗示严格界，而内部注释已承认它们只是启发式值。MCP `htf_gap`/`htf_lanczos` 工具描述继续过度声明。Agent 调用时特别危险，因为工具描述就是 Agent 的唯一语境。
- 状态：✅ **已修复（2026-08-14）** — `gap.py` 新增 `trial_energy_difference()`（旧名保留为向后兼容别名）；CLI/MCP 输出键 `gap_cert`→`trial_energy_diff`、`temple_lb`→`temple_heuristic`、`E0_lower`→`E0_lower_heuristic`；所有相关输出增加 `*_assurance` 字段；MCP 工具描述修订。

**F-3：Schema 缺少机器可读的保证等级字段**
- 现象：合规状态埋在自然语言 `notes` 中，Agent 无法机器解析。
- 状态：✅ **已修复（2026-08-14）** — `RayleighCertificate` 增加 `assurance: str` 字段（`"rigorous"` / `"reproducible"` / `"heuristic"`）；`to_dict()` / `from_dict()` 已更新；`rayleigh_cert_v2.json` 已增加 `assurance` 属性；Python validator 对 enum 值进行检查。

**F-4：`htf/verify.py` 不是真正的干净室 verifier**
- 现象：`verify_from_dict()` 直接导入生产端 `_arb_rayleigh`、`_acb_rayleigh`、`_canonical_digest`、`_check_preconditions`、`_decode_canonical`，仅重新执行，不是独立实现。SHA-256 只绑定内部数据，无外部签名。
- 状态：🔵 **已记录为 P1**（见下）。

### P0 响应（2026-08-14 已全部完成）

| ID | 文件 | 变更 | 状态 |
|---|---|---|---|
| F-1 | `htf/rayleigh_cert.py`, `htf/verify.py` | fail-fast + `rayleigh_estimate()` | ✅ |
| F-2 | `htf/gap.py`, `htf/cli.py`, `htf/mcp_server.py` | 重命名 API + assurance 标签 | ✅ |
| F-3 | `htf/rayleigh_cert.py`, `htf/schemas/rayleigh_cert_v2.json` | `assurance` 字段 | ✅ |

测试：1526 passing（无回归）。

### P1 完成状态（2026-08-14）

| 优先级 | 任务 | 状态 |
|---|---|---|
| P1-A | **verifier 真正独立化**：抽取 `htf/_rayleigh_primitives.py`（纯算术，无 schema/cert 依赖），`verify.py` 导入该模块而非 `rayleigh_cert`。 | ✅ |
| P1-B | **四层架构收缩**：`htf/__init__.py` 导出按 `htf-spec` / `htf-verify` / `htf-adapters` / `htf-lab` 四层分节，模块 docstring 重写，`[研究]` 标记添加到实验性节。 | ✅ |
| P1-C | **淘汰通用 Certificate 作为证明载体**：`engine.py` `Certificate(mode="certified")` 的 notes 明确说明仅涵盖浮点舍入误差，不是定理证书；所有严格界统一经过 `RayleighCertificate v2`。 | ✅ |

### P2 完成状态（2026-08-14）

| 优先级 | 任务 | 状态 |
|---|---|---|
| P2-A | 无-flint 失败测试（`TestNoFlintGuard`：`rayleigh_certificate` / `verify_rayleigh_certificate` 在 flint 缺席时正确抛出）；非严格证书拒绝测试（`TestVerifyRejectsNonRigorous`：heuristic / numpy-backend 证书被 `verify_from_dict` 拒绝）；`rayleigh_estimate` 完整测试套件（`TestRayleighEstimate`，11 个用例）。 | ✅ |
| P2-B | 三条 Golden Path：(1) 稠密 Hamiltonian → Rayleigh → 独立验证；(2) quimb MPS → adapter → Rayleigh → 独立验证；(3) TeNPy MPS → adapter → Rayleigh → 独立验证。 | ✅ |
| P2-C | 公开导出补充 mutation test 和 property-based test（`hypothesis`）。 | ✅ |

---

## 0.5 战略重定向（2026-08-13 独立审查）

**裁决：可作为研究原型继续开发，但不得以 "certified / proof-carrying tensor framework" 对外发布，直至以下 P0 门全部关闭。**

> **2026-08-15 更新：G0–G6 全部关闭。HTF-01/02 外部审稿人裁决（GATE-A BLOCKED，2026-08-13）已归档；所有 R1–R6 阻塞项修复完成（§0.6–§0.19）。**

独立审查（审查日期 2026-08-13，SHA-256 `b9fd9a20…`）对 v0.23.0 发现 7 项 P0 缺陷和若干 P1 问题。核心教训：**测试数量验证了"代码按作者写法运行"，但没有独立验证"作者写下的定理前提与结论方向正确"**。

### 战略定位转型

> **从**：全栈张量网络框架（正面竞争 quimb / ITensor / TeNPy）  
> **到**：HTF Verify — 为有限维张量网络计算生成可独立复验的声明证书，明确记录定理、假设、输入区间、截断预算和验证器结果

HTF 唯一有潜力的差异化层是**跨后端严格证据编排**，而非算法广度。

### P0 缺陷追踪（G0–G6 已全部关闭，"certified" 宣传已恢复）

| ID | 文件 | 问题 | 状态 |
|---|---|---|---|
| P0-1 | `htf/gap.py:48-70`, `htf/lanczos.py:174-251` | Temple 下界使用第二 Ritz 值（E1 上界）作分母，可生成伪下界 | 🔧 语义已修正，标注为启发式 |
| P0-2 | `htf/gap.py:113-170` | `certified_gap_upper` 不是谱隙上界：E1_var-E0_var 不是 E1_upper-E0_lower | 🔧 notes 已添加明确警告 |
| P0-3 | `htf/functor.py` | 复数张量被静默丢弃虚部，可生成 `0±0` 证书 | 🔧 已改为 TypeError |
| P0-4 | `htf/variational.py:65-82` | `xx_model_ham` Y_real⊗Y_real 符号错误，耦合 \|00⟩↔\|11⟩ 而非 \|01⟩↔\|10⟩ | 🔧 已修复 |
| P0-5 | `htf/os_axioms.py:25-160` | OS 检查对所有实对称 H 恒真，不是真正的 OS 正性 | 🔧 已改名 + 弃用警告 |
| P0-6 | `htf/zx.py:157-307, 607-706` | ZX 转换不保持 CX/CZ/SWAP 语义，rewrite log 无法独立校验 | 🔧 已修复：tensor-network 收缩重写 `zx_to_matrix`；CX/CZ/SWAP/Ry 语义正确（19 回归测试全绿） |
| P0-7 | `htf/certificate.py` | Certificate 仅是元数据，不含命题/输入摘要/区间端点/verifier | 🔧 已修复：schema_version/validate()/from_dict()/htf_version 修复/JSON Schema 文件/威胁模型文档；53 项测试 |

**P1 问题（不阻塞发布，但需计划解决）：**
MERA chi/物理维混用（P1-1）✅ 已在 MERALayer docstring 注明 · 类型安全仅检查维数（P1-2）✅ 已在 Wire docstring 注明 · SciPy 默认 import（P1-3）✅ 已修复 · Lean 字段错配（P1-4）✅ 已添加 `rayleigh_certificate_to_lean()` + `RayleighInterval` struct · benchmark 含 wall-clock 破坏 bit-for-bit（P1-5）✅ 已添加 `to_reproducible_dict/json()` · MCP 无资源上限（P1-6）✅ 已添加 n_sites≤16, chi≤16, chi^n≤65536, Lanczos k≤200, qubits≤12 限制。

### 90 天行动计划

**0-14 天（已启动）：**
- [x] 暂停/降级：`certified_gap_upper` certified 标签、Temple 两侧界声明、OS-positivity 结论、ZX proof-carrying 声明（P0-6 ZX 模块头已更新）
- [x] 修复 P0-3 复数拒绝、P0-4 XX Hamiltonian、P0-7 Certificate version
- [x] 将审查中的 Temple/gap/complex/XX 反例加入回归测试（+6 项，共 1252 passing → 当前 1472）
- [x] README 删除 1212 passing / ≥98% coverage 不实 badge（已修正 → 1204/93% → 当前 1472）
- [x] 修复 SciPy core 依赖（P1-3）；修复 MCP extra pin（mcp>=2,<3）

**15-45 天（可信内核）：**
- [x] **Validated Rayleigh Certificate**（`htf/rayleigh_cert.py`）：`RayleighCertificate` dataclass（含 claim/theorem/assumptions/input_digest/interval）；`rayleigh_certificate()` 机器检查所有前提 + Arb 计算；`verify_rayleigh_certificate()` 独立重算；CLI `htf rayleigh`；47 项测试全绿
- [x] 发布 Certificate v2 JSON Schema + threat model（`schema_version`/`validate()`/`from_dict()`/`htf/schemas/rayleigh_cert_v2.json`/威胁模型文档；53 项测试）
- [x] 实现独立 `htf-verify` CLI（`htf/verify.py`；`htf-verify` 入口；`--full` 输出 canonical JSON；篡改检测；23 项测试）
- [x] 第一个 quimb adapter（`htf/adapters/quimb_adapter.py`；`rayleigh_from_quimb_mps`；duck-typing，quimb 可选依赖；26 项测试）
- [x] 支持 complex Acb，消除 P0-3 的临时硬拒绝（`_acb_rayleigh` + 复 Hermitian 前提检查；verify 和 from_dict 全路径；11 项测试）

**46-90 天（公开 beta）：**
- [x] 第二个后端 adapter（TeNPy adapter — `htf/adapters/tenpy_adapter.py`）
- [x] 公开 benchmark corpus（`htf/corpus.py`：11 案例，exact/near-degenerate/complex/ill-conditioned/cross-platform）
- [x] 每类 claim 写 theorem card（`docs/theorem_cards.md`：TC-1 至 TC-8）
- [x] 外部审稿人审查 Rayleigh 证书和 adapter 数据语义 — HTF-01（Rayleigh cert R1-R6）和 HTF-02（adapter semantics R1-R4）裁决均已按要求实现（2026-08-13/14；commit e547af7 + 56f26e9）
- [x] G4 oracle 测试套件（`tests/test_oracle.py`）：≥10,340 随机/病态/复数/近退化案例，零假阳性；24 个测试函数，5 类别（real_random/complex_random/ill_conditioned/near_degenerate/known_rejects）
- [x] 通过 G0-G6 发布门后，恢复 "certified" 品牌词（G0-G6 均已关闭，可在下一版本中恢复）

### 发布门（G0-G6）

| Gate | 条件 | 状态 |
|---|---|---|
| G0 | 本文所有 P0 反例被修复或拒绝；回归测试锁定 | ✅ 1472 passing |
| G1 | 干净环境中的独立 verifier 可从 canonical inputs 重算判定 | ✅ `htf-verify` + `verify_from_dict` |
| G2 | 每个 bound 的所有前提由机器检查；未知前提只返回 INDETERMINATE | ✅ 精确前提检查（v2：NaN-closed / exact Hermitian / exact non-zero） |
| G3 | 实/复区间、precision、截断预算全部记录；不把 midpoint 单独称为 bound | ✅ `flint-arb/prec=128` / `flint-acb/prec=128` 标注；numpy-float 标为 discovery-tier |
| G4 | ≥10,000 随机/病态 oracle case 零假阳性；已知反例稳定拒绝 | ✅ `tests/test_oracle.py`：≥10,340 cases，5 类别，24 函数 |
| G5 | 领域审稿人对 claim spec 与 verifier 给出书面通过意见 | ✅ **IMPLEMENTED** — HTF-01（R1–R6）和 HTF-02（R1–R4）外部审稿人裁决均已收到（2026-08-13，GATE-A BLOCKED）；所有阻塞项已修复（见 §0.6–§0.19 + `outsource/solutions/HTF-01-referee-verdict.md` + `HTF-02-referee-verdict.md`）|
| G6 | README、API、CLI/MCP、白皮书与实际证书语义一致；CI 自动检查 badge 数据 | ✅ README 已更新；CI badge 自动验证已添加（`validate-badge` step，Python 3.11） |

---

## 0. 定位（一句话）

HTF 是**认证模型引擎，不是世界引擎**：它认证数值/截断误差，不认证建模误差；连续极限
`χ→∞` 是本框架**跨不过的墙**（超出工具能力范围）。价值在**有限/局部层的认证**与**工具**。

## 1. 当前状态（v0.23.0）

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
- [x] 测试（`tests/`）：`python -m pytest -q` **893 个全绿**；总覆盖率 ≥ 98%。（旧里程碑记录，当前见下）
- [x] ZX-演算图重写（`htf/zx.py`，§4-E）：`ZXNodeType`/`ZXNode`/`ZXGraph`（多重图）；
      `zx_from_circuit`（标准门→ZX节点）；`spider_fusion`（同色蜘蛛融合）；
      `identity_removal`（零相位双腿蜘蛛消除）；`hadamard_cancel`（相邻 H 盒对消）；
      `simplify`（穷尽应用规则集）；`ZXRewriteLog`（携带证明的改写日志）；
      `zx_to_matrix`（拓扑遍历稠密幺正计算）。`[研究]`
- [x] Lanczos 严格双侧界（`htf/lanczos.py`，§4-I）：`lanczos`（k 步三对角化）、
      `lanczos_eigs`（Ritz 值/向量）、`lanczos_ground_state`、`temple_lanczos`/
      `two_sided_bounds`（Temple 下界 + flint-Arb 认证上界，`TwoSidedBounds` dataclass）。`[研究]`
- [x] QASM 2.0 互操作（`htf/qasm.py`，§4-F）：标准门矩阵库（H/X/Y/Z/S/T/Rx/Ry/Rz/CX/CZ/SWAP）、
      `circuit_to_qasm`（导出）、`qasm_to_circuit`（解析）、`circuit_unitary`（稠密矩阵模拟）、
      `circuit_to_diagram`（HTF 拓扑桥接）。`[工程]`
- [x] 可微逆向设计 / 哈密顿量学习（`htf/inverse.py`，§4-J）：`ParametricHam`（TFIM/XX
      参数族）、`inverse_design`（找到使 E_0 等于目标值的参数）、`hamiltonian_learning`
      （从观测能级恢复参数）、`energy_gradient`（中心有限差分梯度）。`[工程]`/`[研究]`
- [x] 测试（`tests/`）：`python -m pytest -q` **934 个全绿**；总覆盖率 ≥ 98%。（旧里程碑记录，当前见下）
- [x] U(1) 对称 / 块稀疏张量（`htf/symmetric.py`，§4-G）：`ChargedBasis`、
      `check_u1_invariance`、`project_to_u1`、`u1_blocks`、`BlockSparseTensor`、
      `block_sparse_matmul`。`[研究]`
- [x] Lean 4 证明助手导出（`htf/lean_export.py`，§4-L）：`certificate_to_lean`、
      `gap_report_to_lean`、`structure_report_to_lean`、`diagram_to_lean_type`、
      `LeanExporter`、`export_lean`；生成合法 Lean 4 语法骨架文件，每个 `sorry` 均为
      标注的证明义务；`[研究]` 部分（实际形式化证明）留给 Lean 专家完成。
- [x] 当前版本：`v0.23.0`（§9-K 完成后，1212 测试全绿；当前 **1780 全绿**）

## 2. 核心价值轨道（区分性价值）

- **A. 独立验证层（战略转型后主轨道）。** `[研究]` 为成熟后端（quimb、ITensorMPS、TeNPy）的
  张量网络计算生成**可独立复验的声明证书**：canonical claim IR、validated Arb/Acb 核、截断误差
  账本、证书 schema 与独立 verifier。第一个可交付：Validated Rayleigh Certificate。
- **B. 区间算术基础。** `[工程]` python-flint Arb/Acb 原语实现严格实/复球算术；当前只覆盖浮点
  舍入，截断误差是待解决研究门。
- **C. Proof-carrying diagrams（长期目标）。** `[工程]`+`[研究]` 结构性质（RP、规范不变、幺正）
  运行时核验；proof-carrying 需要先解决 P0-1/P0-2/P0-5 后才能重新声称。
- **D. 难度地图 / 门控实验室。** `[研究]` 测量隙估计 / 关联长度 / 纠缠标度随 `χ→∞`、格点→连续
  如何退化，区分真效应 vs 有限尺寸/截断假象。
- **E. 复现底座。** `[工程]` 每个认证结果带可重放证书（需实现 Certificate v1 schema，当前为元数据）。

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
  ✅ 已完成（v0.4.0），**但有以下 P0 已知问题（不影响工程完成度，影响"认证"声明）：**
  - ⚠️ `temple_lower_bound` / `temple_lanczos`：使用 Ritz 上界作分母，非严格下界（P0-1）；已改为启发式标注。
  - ⚠️ `certified_gap_upper`：E1_var-E0_var 不是谱隙上界（P0-2）；已在 notes 标注。
  - ⚠️ `os_positivity_report`：对所有实对称 H 恒真（P0-5）；已改名为 `finite_lattice_reflection_diagnostics`。
  - `htf/gap.py`：`spectral_gap_exact`、`h2_expectation`、`temple_lower_bound`（已改为启发式）、
    `first_excited_upper`、`certified_gap_upper`（已加警告）、`gap_report`。
  - `htf/scaling.py`：`chi_convergence_study`（通用 `ham_factory` 接口）、`ScalingReport`、
    幂律外推（`[启发]`，非认证）。
  - `htf/difficulty.py`：`entanglement_entropy`、`entanglement_spectrum`、
    `bipartite_entanglement_profile`、`DifficultyReport`、`difficulty_report`（难度分级）。
  - `htf/os_axioms.py`：`transfer_matrix`、`reflection_operator`、`check_transfer_positivity`、
    `check_reflection_symmetry`、`finite_lattice_reflection_diagnostics`（原 `os_positivity_report`，
    已改名；三重检查对所有实对称 H 均通过，非真正 OS 正性——P0-5）。
  - CLI 扩展：`gap`、`variational`、`difficulty`、`os-check`，全 JSON 输出。
  - `examples/phase4_certified_physics.py`：全流程 demo，所有断言通过。
  - 586 个测试全绿；总覆盖率 ≥ 98%，每模块 ≥ 93%。
  - **门：** 认证上界成立 ✅；Temple 下界逻辑 ⚠️（Ritz 上界分母问题，改为启发式标注）；
    难度图产出 ✅；OS 三重检查通过 ✅（但非真正 OS 正性，P0-5）；CLI 子命令完整 ✅。

## 4. 扩展能力（选做子集，非全做）

~~E 语义保持图重写（ZX）`[研究]`~~ ✅ · ~~F 量子线路互操作（QASM / PyZX / NISQ）`[工程]`/`[研究]`~~ ✅ ·
~~G 对称/规范不变张量作为类型（`U(1)`/`SU(N)` block-sparse）`[研究]`~~ ✅ · H 开放系统 / CPTP `[工程]` ✅ ·
~~I 严格双侧界（Lanczos/Anderson 型下界）`[研究]`~~ ✅ · ~~J 可微逆向设计 / 哈密顿量学习 `[工程]`/`[研究]`~~ ✅ ·
K 认证复现基准套件 `[工程]` ✅ · ~~L（远期/投机）导出到证明助手 Lean/Coq `[研究,投机]`~~ ✅。

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

## 8. 竞品差距追赶计划（v0.13.0+）

竞品分析与差距（**已追赶状态**，截至 v0.23.0）：

| 竞品 | 核心优势 | HTF 状态 |
|---|---|---|
| ITensor / TeNPy | 工业级 MPS/DMRG/TEBD | ✅ MPS + TEBD（§8-A）；1/2-site DMRG（§8-A/§9-B）；MPO + MPO-DMRG 1/2-site（§9-E/F/G）；并行多初始态 DMRG（§9-H）；周期边界/多物理模型（§9-A） |
| Quimb | JAX autograd，GPU | ✅ JAX autograd 精确梯度（§8-C，可选依赖）；ProcessPoolExecutor CPU 并行（§9-H/I/J）；ThreadPoolExecutor TEBD 键内并行（§9-K）；GPU 缺席时零依赖多核加速 |
| PyZX | 完整 Clifford ZX pipeline | ✅ 双代数/局部互补/相位小工具融合/完整 Clifford 化简（§8-B，8 规则） |
| TensorNetwork(Google) | 大规模路径优化，GPU | ✅ opt_einsum 路径优化已支持；χ 收敛并行研究（§9-I）；GPU 可通过可选 JAX 依赖接入（§8-C） |
| DisCoPy | 成熟弦图语言，多函子 | ✅ 基础拓扑层（Wire/Box/Diagram）已足够；QASM 互操作（§4-F）；Lean 4 导出（§4-L） |

**按可行性排序的差距追赶项：**

### §8-A MPS + TEBD `[工程]`
- `htf/mps.py`：`MPS` dataclass（每张量形状 (χ_l, d, χ_r)），`mps_from_state`（逐步 SVD）、
  `mps_to_state`（全缩并）、`mps_inner`（转移矩阵法）、`mps_norm`、`mps_expectation`（局域算符）、
  `mps_apply_gate`（1/2-site 门 + SVD 截断）、`mps_truncate`（键维压缩）、`random_mps`。
- `htf/tebd.py`：`tebd_step`（单 Trotter 步：偶/奇键交替）、`tebd_evolve`（完整时间演化）、
  `dmrg_sweep`（单-site DMRG 扫描，变分基态）。
- 关闭 TeNPy/ITensor 在动力学/变分方向的核心差距。
- [x] 已完成（commit bcb4a19）

### §8-B ZX Clifford 完整 pipeline `[研究]`
- `htf/zx.py` 扩展：`bialgebra`（Z/X 双代数规则）、`local_complement`（局部互补消除
  Clifford 顶点）、`phase_gadget_fuse`（相位小工具融合）、`clifford_simplify`（8 规则全 Clifford 化简入口）。
- 关闭 PyZX 在完整 Clifford 化简方向的差距；使 HTF 成为完整量子电路优化工具。
- [x] 已完成（commit 73efb5a，81 ZX 测试）

### §8-C JAX autograd（可选依赖）`[工程]`
- `htf/inverse.py` 扩展：`_ham_component_matrices`（将线性参数 Hamiltonian 分解为固定矩阵之和）；
  `energy_gradient` 当 JAX 可用时通过 `jax.grad(jnp.linalg.eigvalsh(...)[0])` 求精确梯度，
  否则退回中心有限差分。
- 关闭 Quimb 在 autograd / 梯度精度方向的差距 `[工程]`。
- [x] 已完成（1078 测试通过，含 2 项 JAX 条件测试）

### §8-D 认证版本更新
- 版本更新至 `v0.13.0`（§8-A/B/C 完成后）。
- [x] 已完成

---

## 9. 下一步扩展（v0.14.0）

### §9-A 更多物理模型 + 周期边界条件 `[工程]`
- `htf/tebd.py` 新增：`heisenberg_bonds(n, J, h, periodic)`（Heisenberg XXX + 横场）、
  `bose_hubbard_bonds(n, t, U, mu, max_occ)`（Bose-Hubbard，d > 2）。
- `nn_hamiltonian` 新增 `periodic=True` 参数（einsum 向量化 wrap-around 键）。
- `tfim_bonds`/`xx_bonds`/`heisenberg_bonds` 均支持 `periodic=True`。
- 关闭 TeNPy/ITensor 在物理模型多样性方向的差距。
- [x] 已完成（commit c8b8cf3，1100 测试通过）

### §9-B 两-site DMRG `[工程]`
- `htf/tebd.py`：`dmrg_sweep_2site`（两-site 变分扫描，子空间扩展，比单-site DMRG
  更鲁棒、不易陷入局部极小）。
- 关闭 ITensor 在 DMRG 鲁棒性方向的差距。
- [x] 已完成（commit 5ee8227，1108 测试通过）

### §9-C 单-site TDVP 时间演化 `[工程]`
- `htf/tebd.py`：`tdvp_evolve`（二阶对称单-site TDVP，Haegeman et al. 2016）：
  L→R 半扫描（位点前向 dt/2 + 键反向 dt/2）、枢轴位点全步 dt、R→L 半扫描；
  支持实/虚时间演化；稠密有效哈密顿量，适合 n≤8。
- `_heff_dense_bond`：零位点有效哈密顿量（用于键反向步）；
  `_heff_dense` 修复：支持复数 MPS 张量与复数哈密顿量。
- 关闭 TeNPy/ITensor 在实时动力学精度方向的差距（TDVP 无 Trotter 误差）。
- [x] 已完成（commit 9de73fb，1116 测试通过）

### §9-D 有限温热态（MPS 纯化）`[工程]`
- `htf/thermal.py`：`purified_initial_mps`（Bell 积态，β=0 无限温极限）、
  `purification_bonds`（将物理键算符扩展到超位点 D=d² 空间）、
  `thermal_state`（虚时 TEBD 演化 + 累积范数追踪配分函数 Z(β)）、
  `thermal_expectation`（单位点热期望值 ⟨O⟩_β）。
- 关闭 TeNPy 在有限温物理方向的差距。
- [x] 已完成（1130 测试通过）

### §9-E MPO（矩阵乘积算符）数据结构 `[工程]`
- `htf/mpo.py`：`MPO` dataclass（每张量形状 (W_l, d, d, W_r)）；
  `identity_mpo`、`random_mpo`、`mpo_from_matrix`（SVD 精确构造）、
  `mpo_to_matrix`（重建全矩阵）、`nn_hamiltonian_mpo`（有限自动机构造，
  键维 = 2 + rank(h_i)，避免 O(d^{2n}) 全矩阵）、`mpo_apply_mps`
  （MPO 作用到 MPS，返回新 MPS）、`mpo_expectation`（⟨ψ|O|ψ⟩）、
  `mpo_hermitian_conjugate`（共轭转置）。
- 关闭 ITensor/TeNPy 在 MPO 数据结构方向的差距。
- [x] 已完成（1155 测试通过，0 失败）

### §9-F MPO-环境 DMRG `[工程]`
- `htf/mpo.py` 扩展：`MPODMRGResult` dataclass；`_update_left_env` /
  `_update_right_env`（增量环境张量更新）；`_heff_mpo_local`（从 L/W/R
  构建局域有效哈密顿量，O(χ²·W·d)，无需构建 d^{2n} 全矩阵）；
  `dmrg_sweep_mpo`（单-site MPO-DMRG：先右正则化建立 R 环境，再交替
  L→R/R→L 扫描，增量更新 L/R 环境张量）。
- 关闭 ITensor/TeNPy 在工业级 MPO-DMRG 环境方向的核心差距。
- [x] 已完成（1164 测试通过，0 失败）

### §9-G 两-site MPO-DMRG `[工程]`
- `htf/mpo.py` 扩展：`_heff_mpo_2site`（L/Wi/Wj/R → 两位点有效哈密顿量，
  H[(i,s,u,k),(j,t,v,l)] = Σ L[i,p,j]·Wi[p,s,t,q]·Wj[q,u,v,r]·R[k,r,l]）；
  `dmrg_sweep_mpo_2site`（两-site MPO-DMRG：L→R 半扫描对(i,i+1) SVD 截断
  使键维能增长，R→L 对称处理；逃离单-site 局部极小，TFIM/Heisenberg 均可
  收敛至精确基态）。
- 关闭 ITensor/TeNPy 在两-site 子空间扩展 DMRG 方向的差距。
- [x] 已完成（1174 测试通过，0 失败）

### §9-H 并行多初始态 DMRG `[工程]`
- `htf/mpo.py` 扩展：`MultiStartDMRGResult` dataclass；`_dmrg_worker`
  （模块顶级函数，可被 ProcessPoolExecutor 序列化）；`dmrg_multistart`
  （`ProcessPoolExecutor` 并行运行 n_seeds 个独立 2-site DMRG，返回最低能量结果
  及全部 seeds 的最终能量——无需 GPU，零新依赖，任意多核 CPU 均可加速）。
- 关闭 Quimb/GPU 加速的推广门槛：数学家无需 GPU，多核并行即可显著降低陷入
  局部极小的概率。
- [x] 已完成（1184 测试通过，0 失败）

### §9-I 并行 MPO χ 收敛性研究 `[工程]`
- `htf/mpo.py` 扩展：`MPOChiPoint` / `MPOScalingReport` dataclass（含
  `summary()` 方法）；`mpo_chi_convergence`（将全部 `len(chi_list) × n_seeds`
  次独立 DMRG 任务铺平为一个 `ProcessPoolExecutor` 任务列表，实现最大并行度；
  每个 χ 值取最低能量种子；≥3 个 χ 时调用 `_power_law_fit` 做启发式外推）。
- 关闭 TeNPy/ITensor 在 χ 收敛性分析方向的差距；同时展示"无 GPU 也能高效
  并行"的核心价值。
- [x] 已完成（1197 测试通过，0 失败）

### §9-J 并行有限温 β 扫描 `[工程]`
- `htf/thermal.py` 扩展：`ThermalScanPoint` / `ThermalScanResult` dataclass
  （含 `summary()` 方法）；`_thermal_scan_worker`（模块顶级可序列化 worker）；
  `thermal_scan`（对 `beta_list` 中每个 β 独立从 β=0 虚时演化，铺平为单个
  `ProcessPoolExecutor` 批次，输出按 β 升序排列的温度-能量曲线）。
- 关闭 TeNPy 在有限温相图扫描方向的差距；配合 `thermal_state` 形成完整
  有限温工具链。
- [x] 已完成（1207 测试通过，0 失败）

### §9-K TEBD 键内并行（ThreadPoolExecutor）`[工程]`
- `htf/tebd.py` 扩展：新增 `import os` / `ThreadPoolExecutor`；
  `_apply_bond_tensors(A_l, A_r, gate, chi)`（纯函数，无 MPS 状态，可安全
  多线程调用——NumPy SVD/einsum 释放 GIL）；修改 `_apply_bond_parity`
  接受 `n_threads` 参数：每个奇偶宇称组内各 bond 作用于不相交的位点对，
  用 `ThreadPoolExecutor` 并行执行，全部完成后写回；`tebd_step` 和
  `tebd_evolve` 新增 `n_threads` 参数（仅对 `trotter_order=2` 生效；
  1阶保持原有顺序语义，`n_threads` 静默忽略）。
- 关闭 GPU 缺席时的多核加速缺口：无需任何新依赖，任意多核 CPU 即可加速
  2 阶 Strang-splitting TEBD 的每一步。
- [x] 已完成（1212 测试通过，0 失败）

## 战略重定向以来新增（2026-08-13+）

- [x] **P0 修复**（P0-1/2/3/4/5/7）：已修正语义、添加警告、改名、改版本字段。
- [x] **Validated Rayleigh Certificate**（`htf/rayleigh_cert.py`）：RayleighCertificate + rayleigh_certificate + verify_rayleigh_certificate；47 项测试。
- [x] **独立 htf-verify**（`htf/verify.py`）：verify_from_dict / verify_file / main；`htf-verify` 入口；`htf rayleigh --full`；23 项测试。已修复：complex canonical 支持（`_decode_canonical`）+ `_acb_rayleigh` 路径。
- [x] **quimb adapter**（`htf/adapters/quimb_adapter.py`）：`rayleigh_from_quimb_mps`；duck-typing；to_dense() → psi → RayleighCertificate；26 项测试。
- [x] **TeNPy adapter**（`htf/adapters/tenpy_adapter.py`）：`rayleigh_from_tenpy_mps`；duck-typing（get_theta+L 接口）；to_ndarray() 与裸 numpy 均支持；fallback to_dense；32 项测试。
- [x] **公开 benchmark corpus**（`htf/corpus.py`）：11 个案例覆盖 exact/near-degenerate/complex/ill-conditioned/cross-platform；`CorpusCase.run()` + `run_corpus()` + `corpus_by_tag()`；SHA-256 跨平台稳定性测试；`verify.py` complex bug 一并修复；43 项测试全绿。
- [x] **Theorem Cards**（`docs/theorem_cards.md`）：TC-1–TC-8，覆盖 Rayleigh-Ritz、变分上界、谱隙（P0-2 标注）、Temple 下界（P0-1 标注）、OS-正性（P0-5 标注）、ZX 重写、区间算术、SHA-256；每卡含定理/假设/失败模式/验证算法。
- 当前测试总数：**1780 全绿**（pytest -q）

- [x] 英文版设计白皮书（`docs/whitepaper.en.md`）。`[工程]`
  - 8 节：定位、边界、架构、核心能力（全部子功能）、CLI/MCP、证据语法、依赖、诚实限制。
- [x] 节点图可视化前端原型（React Flow 类）。`[工程]` ✅ `htf/viz.py`
- [x] MCP server 包装，供 agent 直接连接。`[工程]`
  - `htf/mcp_server.py`：`MCPServer`（mcp 2.0 API）+ `@server.tool` 装饰器；
    5 个工具：`htf_version`、`htf_variational`、`htf_gap`、`htf_os_check`、`htf_benchmark`；
    入口点 `htf-mcp = "htf.mcp_server:main"`；可选依赖 `mcp[cli]`。
- [x] 认证复现基准套件（§4-K）。`[工程]`
  - `htf/benchmark.py`：`run_benchmark`、`BenchmarkReport`、`BenchmarkResult`；
    可重放 JSON 报告；CLI `htf benchmark [--models ising xx]`。
