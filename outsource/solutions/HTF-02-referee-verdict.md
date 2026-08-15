# HTF-02 — Adapter Data Semantics：独立 Gate-A 审稿裁决

**审稿日期：** 2026-08-13  
**送审件：** `HTF-02-adapter-semantics.md`  
**裁决：** **GATE-A BLOCKED**  

## 1. 结论摘要

当前两条适配器不能按送审件写出的无条件语义承诺通过 Gate-A。问题不只是“补一句基顺序说明”：

1. **TeNPy 路径不是一个可靠的全波函数导出器。** 它没有限制 `bc == "finite"`，没有按标签固定轴顺序，没有去掉并验证两端虚拟腿，也没有处理 `Site.sort_charge()` 引入的局部基置换。TeNPy 官方的全波函数函数恰好显式执行这些步骤。当前路径因此会接纳 iMPS、segment MPS、派生的多物理腿 MPS，且可能把内部 charge-sorted 基中的向量与常规基中的 `H` 配对。
2. **两条路径都会在 `max|Im ψ| <= imag_tol` 时把原状态静默替换成 `Re ψ`。** 对“为 MPS 所表示的同一个状态出具严格证书”的承诺，这是实质性错误；`imag_tol` 还是绝对阈值，连整体缩放/全局相位下的适配器行为都不保持不变。
3. **基顺序只能是调用者前提，现有代码无法从一个裸 `numpy.ndarray` 的 `H` 中验证它。** 维数相等不能发现站点置换或局部基置换。该前提必须成为明确、可测试、可记录的接口契约。
4. **送审件中的 TeNPy 数值 mock 不可能是所声称的四站点标准 MPS 全链 theta。** `L=4` 的单物理腿 MPS 应有四个物理轴，典型形状为 `(1,d_0,d_1,d_2,d_3,1)`；所给 `(1,4,1)` 只含一个物理轴。该测试只证明 duck-typed mock 能被 `ravel()`，没有测试 TeNPy 语义。

因此，当前状态是 **BLOCKED**。完成第 8 节列出的代码、契约和测试修复后，有限、标准单物理腿 MPS 的适配器可以重新送审。

## 2. 审查边界与版本锚点

本审查接受 HTF-01 已建立的 Rayleigh–Ritz/证书核心性质，不重新证明它；这里只审查进入证书核心的 `(H, ψ)` 是否具有送审件声称的语义。

送审件没有给出 quimb/TeNPy 的版本、commit 或官方链接，故“always”“from docs”不能跨版本验证。本次外部核对基于 2026-08-13 可访问的官方资料：

- quimb 最新文档显示为 `1.15.1.dev12+g2bd344ce1` 附近的开发文档；
- TeNPy 最新文档显示为 `1.1.1.dev66+f1223d6` 附近的开发文档；
- 这些版本号是审查时的资料锚点，不是建议项目依赖开发版。

项目必须另行钉住所支持的发行版本或 commit，并在该版本上运行真实后端集成测试。尤其是 quimb 的更新日志明确记录过带惰性 qubit 置换的 `CircuitPermMPS.to_dense()` 顺序错误修复；这说明对所有带 `to_dense()` 的 duck type 作跨版本“always”声明是不成立的。

## 3. 上游 API 的独立核对

### 3.1 quimb

当前 quimb 的 `TensorNetworkGenVector.to_dense()` 文档说明：

- 不显式传 `inds_seq` 时，物理指标按对象的 `sites` 顺序分成一个组，只包含仍存在的站点；
- 默认返回 ket/列向量，形状是 `(D, 1)`，而不是送审件声称的无条件 `(D,)`；
- 对该结果 `.ravel(order="C")` 会得到长度 `D` 的一维数组。

所以，对普通 `MatrixProductState`、站点依次为 `0,1,...,L-1`、每个物理轴的整数索引就是调用者所称局部计算基时，最后站点变化最快的结论成立。可是 quimb 本身只知道物理轴索引和 `sites` 顺序，不知道调用者口中的“标准计算基”或几何/逻辑 qubit 顺序。MPS 还可以只覆盖 `L` 中的一部分站点；适配器又仅检查 `hasattr(to_dense)`，会接纳语义更宽的 wrapper。

**核对结果：** `to_dense().ravel()` 的机械展开对普通 MPS 是正确的；“总是标准计算基、总与 H 匹配”不成立，必须作为调用者契约。送审件关于返回形状的文字需改为 `(D,1)`（当前默认）或更稳妥的“可转成总大小为 `D` 的数组”。

### 3.2 TeNPy

TeNPy 官方资料给出以下事实：

- `MPS` 同时表示 finite MPS、segment MPS 和 infinite MPS；对 iMPS，`L` 是单位胞长度，不是有限系统的总站点数。
- `get_theta(i,n)` 返回带标签 `vL,p0,...,p{n-1},vR` 的局部/全段波函数；外侧虚拟腿并不对所有边界条件都平凡。
- `Site.sort_charge()` 会改变局部基顺序；自 TeNPy 1.0 起其默认值为 `True`。`Site.perm` 记录内部 charge-sorted 基相对未守恒/常规局部基的置换。
- TeNPy 官方 `tenpy.algorithms.exact_diag.get_full_wavefunction()` 要求 `psi.bc == "finite"`，按标签把轴排成 `vL,p0,...,vR`，把两端大小为 1 的虚拟轴压掉，并默认撤销各 `Site.perm`，最后才按 NumPy/Kronecker 顺序拉直。
- TeNPy 自己的 exact-diagonalization 路径同样先拒绝非 finite MPS，再取全链 theta、固定/切掉 `vL,vR`。

这与送审实现有直接差异。`get_theta(0,L).to_ndarray().ravel()` 对**有限、标准单物理腿、轴序如文档、并且 H 使用 TeNPy 内部局部基顺序**的对象可以给出正确向量；它不是无条件的 TeNPy 全波函数导出契约。

## 4. 四条链接逐项裁决

| 链接 | 裁决 | 独立判断 |
|---|---|---|
| Link 1 — quimb `to_dense()` | **PARTIAL / CONDITIONAL** | 普通 MPS 的张量收缩与按 `sites` 展开成立；当前默认形状是 `(D,1)`。局部基、逻辑站点顺序、缺失站点和 wrapper 语义不能由适配器推断；H 同序是调用者前提。 |
| Link 2 — TeNPy `get_theta(0,L)` | **REFUTED（按无条件陈述）** | 只对 finite、标准单物理腿 MPS 且双方采用同一内部基时成立。现代码不拒绝 iMPS/segment，不撤销 charge sorting，不显式按腿标签转轴，也不排除多物理腿派生类。 |
| Link 3 — gauge / normalization | **PARTIAL** | 理想 Rayleigh 商对任意非零复标量严格不变；完整收缩对保持物理态且元数据一致的 gauge 变换不变。可是证书的浮点/区间 `upper` 不必逐比特相等；附件未提供核心实现，无法确认该实现性声明。更严重的是绝对 `imag_tol` 使适配器预处理本身不具缩放或全局相位不变性。 |
| Link 4 — complex handling | **REFUTED** | 阈值内无条件取实部会为另一个状态出证书；提高 `imag_tol` 可放大错误，`NaN`/`inf` 还会绕过比较。对严格的“同一 MPS 状态”承诺不可接受。 |

### Link 3 的精确限定

对 Hermitian `H` 和 `α != 0`，理想商满足

\[
R_H(\alpha\psi)
=\frac{(\alpha\psi)^*H(\alpha\psi)}{(\alpha\psi)^*(\alpha\psi)}
=R_H(\psi).
\]

这确认的是数学量，不确认 `rayleigh_certificate(...).upper` 在有限精度或区间外包下逐值相同。独立计算中，实缩放和复缩放后的双精度结果相差可达约 `2.22e-16`；正确测试应检查两者都包住同一精确 Rayleigh 商，或在已证明的舍入规则下比较，而不是无依据地断言 `upper` 完全相等。

TeNPy 的 `get_theta()` 官方语义是构造 n-site wavefunction，因此在合法 canonical/form 元数据下，A/B/混合 canonical 表示的 gauge 改变不会改变物理 theta，除浮点误差外成立。要排除两类边界情形：`form=None`/元数据不一致，以及 DMRG mixer 暂态中矩阵型 `S` 触发带 cutoff 的伪逆。不能把结论写成对任意内部张量修改无条件成立。

## 5. Gate-A 问题逐项回答

### Q1 — 基顺序一致性

**答案：不是代码保证，而是未写明的调用者前提。** 裸矩阵没有站点、局部基标签、fermionic swap 规则或 lattice-to-MPS 映射；`H.shape[0] == len(psi)` 只能检查总维数，不能发现任何置换。

最低限度必须在两条公开函数的 docstring 中加入：

> `H` MUST be expressed in exactly the tensor-product basis used by the extracted state vector. For quimb this is the order of the present `mps.sites` and each physical-index order used by `to_dense()`. For TeNPy this adapter accepts only finite, standard one-physical-leg MPS and uses sites `0..L-1`, the pre-charge-sort local basis (`undo_sort_charge=True`), and C-order flattening. The adapter cannot infer or verify this convention from a plain ndarray; any lattice/site/local-basis permutation must be applied consistently to both `H` and `psi`.

仅加维数检查不是运行时修复。更强的设计是要求一个可序列化的 `BasisSpec`（站点顺序、各 `d_i`、局部标签/置换策略）并把其 digest 写入证书；H 构造器与适配器应共享该对象。

### Q2 — TeNPy 非均匀局部维数

对 finite、标准单物理腿 MPS，按 `vL,p0,...,p{L-1},vR` 排轴并去掉平凡边界腿后，总大小确为

\[
D=\prod_{i=0}^{L-1} d_i.
\]

NumPy 默认 `ravel(order="C")` 使最后物理轴变化最快，所以非均匀维数本身没有问题。现代码没有超出后续 `len(psi)==H.shape[0]` 的语义检查；从所引片段甚至无法独立确认该后续检查的实现。

必须在适配器边界显式检查：

- `mps_like.bc == "finite"`；
- 标准单物理腿 MPS，或明确支持并记录派生类的所有物理腿；
- `raw.size == math.prod(mps_like.dim)`；
- `H.ndim == 2`、`H` 方阵、`H.shape[0] == raw.size`；
- 所选 TeNPy basis policy 与 H 构造路径一致。

最后一项不能由长度检查替代。

### Q3 — 静默复数截断

**答案：是阻塞性的语义缺口，不是可忽略的浮点便利。** 即使虚部很小，严格证书也必须准确说明证书针对的是哪个数值向量。把 `ψ` 改成 `Re ψ` 后，digest 与 Rayleigh 商都属于另一个状态。

独立反例取

\[
Y=\begin{pmatrix}0&-i\\ i&0\end{pmatrix},\quad
H=10^{12}Y,\quad
\psi=(1,i10^{-12})^T.
\]

此时 `max|Im ψ|=10^{-12}<10^{-10}`，现代码会接受并取实部；独立计算得到

- 原状态 `R_H(ψ) = 2.0`；
- 截断状态 `R_H(Re ψ) = 0.0`。

这说明“虚部小”本身不给出 Rayleigh 误差的绝对上界；还必须控制 `||H||` 和状态误差。若用户设 `imag_tol=1`，对 `ψ=(1,0.5i)` 与 `H=Y`，原商为 `0.8`，截断后为 `0.0`。

此外，绝对阈值导致同一射线的两个缩放表示可能一个被拒绝、一个被接受并截断；整体复相位也会改变处理结果。这直接破坏适配器层面的 normalization/phase invariance。

最低修复是：

- 首选：保留 complex amplitudes，转换为 `complex128` 并交给已支持复态的 `rayleigh_certificate`；
- 若证书核心不能处理复态：对任意非零虚部 fail closed，不能使用容差静默投影；
- 如另设显式 `project_to_real=True` 工具，它必须改名、在 notes 中声明 `state_transform=real_projection`，且不得再声称证书对应原 MPS 状态。若仍要为原状态给严格界，必须加入基于 `||H||` 和 `||ψ-Reψ||` 的经证明误差项。

### Q4 — H 的责任归属

对一个给定的 Hermitian 数组，证书核心可以给出代数上有效的 Rayleigh 上界；适配器无法证明这个数组就是生成 MPS 时的物理 Hamiltonian。现送审件在审稿说明中提到了这一点，但所示公开函数/notes 没有形成可执行契约。

必须在 docstring 中明确外部来源责任，并在 `cert.notes` 或结构化 provenance 中记录至少：后端及版本、站点顺序、局部维数、TeNPy 的 `undo_sort_charge` policy、`H_source=caller`。H 的内容 digest 应由证书层绑定；digest 证明“使用了哪一个 H”，不证明“这个 H 物理上正确”。

### Q5 — 最终裁决

**BLOCKED。** quimb 的普通有限 MPS 收缩路径在明确基契约下可用；TeNPy 当前路径和两条路径的复数投影都存在“传入状态与被认证状态不相同”的可复现失败模式。不能以测试数目或“通常只是数值噪声”覆盖该缺口。

## 6. 数值锚点与送审测试核查

### 6.1 两站点锚点

对 `H=diag(0,1,1,2)`、`ψ=(1,0,0,0)`，独立 NumPy 计算确认：

- `ψ†Hψ = 0`；
- `ψ†ψ = 1`；
- `RQ = 0`；
- `eigmin(H) = 0`。

因此数学锚点正确。`cert.upper <= 1e-9` 涉及未给出的 Arb/证书实现，不能仅由本附件独立确认。

### 6.2 C-order 与非均匀维数

对形状 `(1,2,3,1)`、物理元素按 `10*i+j` 标号的 theta，独立拉直结果为

```text
[0, 1, 2, 10, 11, 12]
```

确认最后物理指标变化最快；这只确认轴顺序固定以后 `.ravel()` 的机械事实，不确认 H 与这些轴的物理标签一致。

### 6.3 维数相等不能发现基置换

令标准态 `ψ_std=(1,0)`，内部基置换 `P=[[0,1],[1,0]]`，`H=diag(1,0)`。则

```text
R_H(ψ_std)   = 1.0
R_H(Pψ_std)  = 0.0
```

两个向量长度相同，维数检查完全通过；这正是 TeNPy `Site.perm` 或站点顺序不一致会产生的类别。

### 6.4 mocks 与测试数声明

- quimb mock 返回 `(2,2)`，而当前 quimb 默认 ket 是 `(4,1)`；`ravel()` 后锚点仍对，但该 mock 没有测试 quimb 的站点/物理指标顺序。
- TeNPy mock 的 `L=4` 与 `(1,4,1)` 不符合标准四站点 theta 的腿结构。若要模拟两站点、每站点 `d=2`，最低限度应使用 `L=2` 和形状 `(1,2,2,1)`；但真实 TeNPy 集成测试仍不可由 mock 替代。
- “1427 tests green / 26 / 32”没有随附件提供 commit、日志或测试代码，故裁决为 **INCONCLUSIVE**。即使数字属实，这两个给出的 mock 也没有覆盖当前阻塞问题。

## 7. 非循环性与可信边界

未发现使用 Rayleigh–Ritz 结论反向证明适配器正确的形式循环。但送审论证多次从“测试通过”跳到“后端语义正确”，而给出的 mocks 正是自行规定了期望输出；这不是数学循环，却是**测试替身与被测契约同源**的问题，不能作为独立后端证据。

本裁决没有确认以下未提供实现的事项：`rayleigh_certificate` 的 Hermiticity/有限值/零向量检查、区间端点方向、digest 覆盖字段、scaled input 的 `upper` 是否逐值相同。它们仍由 HTF-01 或证书核心审查负责。

## 8. 解除 BLOCKED 的最小修复

以下四组修改均为重新送审前置条件。

### R1 — 用后端认可的有限态导出语义

TeNPy 支持版本若提供官方 helper，优先采用等价于：

```python
from tenpy.algorithms.exact_diag import get_full_wavefunction

def _extract_tenpy_state_vector(mps_like):
    if getattr(mps_like, "bc", None) != "finite":
        raise ValueError("TeNPy adapter accepts only bc='finite'")
    raw = np.asarray(
        get_full_wavefunction(mps_like, undo_sort_charge=True)
    )
    if raw.ndim != 1:
        raise ValueError("TeNPy full wavefunction must be one-dimensional")
    if hasattr(mps_like, "dim"):
        expected = math.prod(int(d) for d in mps_like.dim)
        if raw.size != expected:
            raise ValueError("TeNPy local dimensions do not match state size")
    return _preserve_state_dtype(raw)
```

若所钉版本没有该 helper，项目内实现必须逐项复制其**语义**而不是私下假定 ndarray 布局：检查 finite、按腿标签转轴、验证并 squeeze 两端虚拟腿、选择并记录是否撤销 `Site.perm`、拒绝多物理腿派生类或完整支持其 basis spec。

### R2 — 禁止隐式 `Re ψ`

两条路径共享：

```python
def _preserve_state_dtype(raw):
    raw = np.asarray(raw).reshape(-1, order="C")
    if raw.size == 0:
        raise ValueError("empty state vector")
    dtype = np.complex128 if np.iscomplexobj(raw) else np.float64
    raw = raw.astype(dtype, copy=False)
    if not np.all(np.isfinite(raw)):
        raise ValueError("state vector contains non-finite values")
    return raw
```

删除 `imag_tol` 的静默投影语义。若为兼容保留参数，应立即弃用并使任何非零虚部 fail closed；不要让 `NaN`、`inf` 或大阈值成为绕过路径。

### R3 — 固化 basis/H 契约与前置检查

在调用证书核心前显式验证 H 为二维方阵且维数与 `psi.size` 一致；核心层继续负责 Hermiticity 和严格算术。把第 5 节 Q1 的 basis contract 逐字或等价地加入两条 docstring，并把 basis/provenance 元数据写入证书。

### R4 — 用真实后端和对抗性测试替换语义不足的 mocks

至少增加：

1. 真实 quimb MPS：非对称幅度向量，逐项与显式 Kronecker 基比较；覆盖非均匀维数、站点子集/顺序策略及所支持的精确版本。
2. 真实 finite TeNPy MPS：与官方 `get_full_wavefunction(..., undo_sort_charge=True)` 逐项比较；覆盖 `Site.perm != identity` 和非均匀 `mps.dim`。
3. `bc='infinite'`、`bc='segment'`、多物理腿派生类必须 fail closed。
4. 复态保真测试：使用本报告的 Pauli-Y 反例，确认虚部被保留；若选择 real-only policy，则确认任何非零虚部都拒绝。
5. 基置换 mutation guard：只置换 H 或只置换 ψ 时，测试必须检测 basis-spec/digest 不匹配，而不是仅依靠总维数。
6. gauge/缩放测试：比较被认证的数学 Rayleigh 值或区间包含关系，不要求未经证明的 `upper` 逐比特相等。
7. 记录并钉住 quimb、TeNPy、NumPy 的版本/commit；送审件中的每个外部 API 声明附官方 URL 和版本。

## 9. 重新送审的通过条件

只有在 R1–R4 全部完成，并提交真实后端测试证据后，HTF-02 才可重新裁决。届时可望得到的范围化结论是：

> 对受支持版本中的有限、标准单物理腿 MPS，在显式声明且与 H 一致的 tensor-product basis 中，适配器无损提取实或复状态向量；证书针对该数值向量与调用者提供的 H。适配器验证代数一致性，但不证明 H 的物理来源。

任何比这更宽的承诺——特别是“任意 TeNPy MPS”“任意带 `to_dense()` 的对象”或“容差内复态仍是同一状态”——目前均无证据支持。

## 10. 核对来源

- [quimb `TensorNetworkGenVector.to_dense()` 官方文档](https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/tnag/core/index.html#quimb.tensor.tnag.core.TensorNetworkGenVector.to_dense)
- [quimb `MatrixProductState` 官方文档](https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/tn1d/core/index.html#quimb.tensor.tn1d.core.MatrixProductState)
- [quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html)
- [TeNPy `MPS` 官方文档](https://tenpy.readthedocs.io/en/latest/reference/tenpy.networks.mps.MPS.html)
- [TeNPy `get_full_wavefunction()` 官方文档](https://tenpy.readthedocs.io/en/latest/reference/tenpy.algorithms.exact_diag.get_full_wavefunction.html)
- [TeNPy `exact_diag.py` 上游实现](https://github.com/tenpy/tenpy/blob/main/tenpy/algorithms/exact_diag.py)
- [TeNPy `Site` 与 `perm`/`sort_charge` 官方文档](https://tenpy.readthedocs.io/en/latest/reference/tenpy.networks.site.Site.html)
- [TeNPy `Array.to_ndarray()` 官方文档](https://tenpy.readthedocs.io/en/latest/reference/tenpy.linalg.np_conserved.Array.html)
- [NumPy `ravel()` 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.ravel.html)
