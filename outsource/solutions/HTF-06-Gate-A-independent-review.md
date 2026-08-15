# 通用送审头（必附）

> 请对所附问题做**独立审稿**，而不是替作者完成或默认其结论成立。
>
> 1. **先核对精确陈述**：逐一核对附件中所有定义、代码片段、声明和链接。凡作为前提使用的数学定理，须核对其**精确陈述、假设是否覆盖此处所用的对象**。
> 2. **逐步检查，不预设成立**：对每一步给出独立判断。凡载荷性的等式、界、数值锚点，用**独立计算**核验，不要凭记忆；发现与附件不符时，报告独立算得的精确值。
> 3. **无循环性红线**：确认没有任何一步在证明某事时假设了它本身，或借助本任务范围之外的未证结论。
> 4. **诚实的裁决空间**：允许并鼓励 `CONFIRMED / PARTIAL / REFUTED / CONDITIONAL / INCONCLUSIVE + 精确定位`。不要为了通过而通过。
> 5. **交付可独立审查的成稿**：结论须逐项对应附件的验收标准；对 `PARTIAL/REFUTED/CONDITIONAL`，给出确切反例、间隙或最小修改。

# HTF-06 — Post-P0 Rayleigh 证书流水线独立 Gate-A 审稿裁决

审阅对象：`HTF-06-post-p0-comprehensive-review(1).md`（658 行）  
送审规范：`REVIEW_PROMPT(1).md`  
审阅日期：2026-08-15  
独立验证程序：`independent_rayleigh_audit.py`  
程序 SHA-256：`ea29a06788bfa3cd1b206b9f41be641d5d7e2f2829707a215df32f423ff4e8a2`

## 1. 总裁决

**Gate-A：`BLOCKED`。**

Rayleigh–Ritz 的数学核心正确；对附件给出的二进制浮点矩阵，把各分量解释为其精确二进有理数后，A1–A5 足以使该数学定理适用。附件中 Arb/Acb 的球算术思路，以及从 Arb 上端点经“最近舍入再向上走一个 binary64 数”导出有限上界的论证，在函数成功返回且所依赖库实现符合文档时也是保守的。

但整条证书链不能通过 Gate-A，原因不是措辞偏好，而是可定位、可复现的阻塞项：

1. 两条验证器成功路径按所附源码均含未定义变量，不能成功验证正常证书；
2. 送审件声称“全部定义和代码自包含”，却遗漏线格式安全所必需的编码器、解码器、证书类和完整 schema；
3. 对所有满足 A1–A5 的有限输入“产生证书”的覆盖性主张有明确反例；
4. 生产器可在端点有限时生成 `midpoint=inf, radius=inf` 的所谓 rigorous 证书对象，而验证器不检查这些字段；
5. 当前验证器与生产器共用同一算术实现，不能发现系统性算术错误，因此不能称为算术独立验证。

上述第 1 项单独已经足以给出 `BLOCKED`；其余项目表明仅修变量名仍不足以通过 Gate-A。

## 2. 精确命题、量词和对象

附件的数学主张应规范化为以下条件命题，而不能写成对所有抽象自伴算子的无条件生产保证。

令 $n\ge 1$，令 $H\in\mathbb C^{n\times n}$ 和 $\psi\in\mathbb C^n\setminus\{0\}$。这里每个实部、虚部都取自一个**有限 binary64 值所代表的精确二进有理数**。假设 $H=H^*$。令

\[
q(H,\psi)=\frac{\psi^*H\psi}{\psi^*\psi},\qquad
E_0=\lambda_{\min}(H).
\]

若证书中有限 binary64 数 `upper` 满足精确不等式

\[
q(H,\psi)\le \operatorname{value}_{\mathbb R}(\texttt{upper}),
\]

则

\[
E_0\le q(H,\psi)\le \texttt{upper}.
\]

这里的 $E_0$ 只能解释为证书所存**有限矩阵 $H$** 的最小特征值。它不是未经额外证明的无限系统基态能量，也不自动包括模型截断、基底截断、MPS/MPO 截断、离散化、输入构造或 binary64 量化误差。

### 2.1 自包含证明

有限维 Hermitian 谱分解给出 $H=U\operatorname{diag}(\lambda_1,\ldots,\lambda_n)U^*$，其中所有 $\lambda_i\in\mathbb R$，并令 $E_0=\min_i\lambda_i$。写 $c=U^*\psi\ne0$，则

\[
q(H,\psi)
=\frac{\sum_i\lambda_i|c_i|^2}{\sum_i|c_i|^2}
\ge \frac{\sum_iE_0|c_i|^2}{\sum_i|c_i|^2}=E_0.
\]

因此，只要独立算术确认 `upper` 不小于精确 Rayleigh 商，目标上界即成立。这个证明没有把 $E_0\le\texttt{upper}$ 当作前提，非循环性成立。

### 2.2 引用核对

附件只写了 “Courant & Hilbert, *Methods of Mathematical Physics*, Vol. I, §VI.1”，没有年份、版次、页码或定理编号。1953 年英文修订版第 VI 章 §1（印刷页约 398 起）讨论的是自伴二阶微分方程及边界条件下的经典极值性质，并非附件所写的任意有限复 Hermitian 矩阵命题的逐字陈述。更直接的有限维来源是同卷第 I 章 §4.1、印刷页 31 起的 “Minimum–maximum property of eigenvalues / Characterization …”；该处结果未以附件可核对的定理编号给出。

结论：**原引用位置相关，但精确元数据和对象覆盖说明不足**。这不推翻数学命题，因为上节已经给出有限维自包含证明；建议把引用改为：

> R. Courant and D. Hilbert, *Methods of Mathematical Physics*, Vol. I, English ed., Interscience, 1953, Ch. I, §4.1, pp. 31–33 (unnumbered minimum–maximum result); finite-dimensional specialization proved explicitly here.

核对来源：[1953 年卷一扫描本](https://ia601708.us.archive.org/31/items/in.ernet.dli.2015.140700/2015.140700.Methods-Of-Mathematical-Physics-Vol1.pdf)、[Wiley 卷一说明](https://www.wiley-vch.de/de/fachgebiete/naturwissenschaften/physik-11ph/mathematische-physik-11ph3/methods-of-mathematical-physics-978-3-527-41447-5)。

## 3. 阻塞项与最小反例

### B1 — 两条验证器路径均有未定义变量（阻塞）

**Link 5a，附件行 328–340：**

- 分支条件使用 `is_complex`，函数体内没有定义；
- 重算值赋给 `upper_v`，随后却比较和打印 `recomputed_upper`。

在不存在同名模块全局变量时，正常路径首先在 `is_complex` 处抛出 `NameError`；即使补上它，仍会在 `recomputed_upper` 处抛出 `NameError`。依赖偶然的模块全局变量也不是正确修复。

**Link 5b，附件行 371–374、473–483：**

- 已读取的变量名是 `backend`；
- 后端交叉检查却使用未定义的 `stored_backend`。

因此 rigorous 证书执行到该处会抛出 `NameError`，无法到达附件行 498–508 的成功返回。

这还与附件行 533–534 “所有锚点由 `rayleigh_certificate + verify_from_dict` 产生”以及行 652 “1766 tests passing”发生直接证据冲突。可能性只有三种：送审摘录与 commit 不同、测试没有执行成功分支、或测试/覆盖率陈述没有对应当前代码。没有仓库 URL、不可变源码包、测试日志和依赖锁，无法替作者选择其中一种。

**最小修复：**

```python
# verify_rayleigh_certificate: decode 后加入
is_complex = np.iscomplexobj(H) or np.iscomplexobj(psi)

# 并统一变量名
_, recomputed_upper, _, recomputed_backend = (
    _acb_rayleigh(H, psi) if is_complex else _arb_rayleigh(H, psi)
)

# verify_from_dict: 读取时就使用后续同名变量
stored_backend = full_cert.get("backend", "")
```

然后必须增加真实/复数各一条端到端成功测试；测试必须断言执行到 `verified=True`，不能只断言未在更早处失败。

### B2 — 序列化链并不自包含（阻塞）

附件行 57 声称 “All definitions and code (self-contained)”，但至少下列载荷性定义缺失：

- `_encode_canonical`；
- `_decode_canonical`；
- `RayleighCertificate` 及其 `canonical`/`to_dict` 行为；
- `SCHEMA_VERSION` 的精确值；
- `EXPECTED_THEOREM` 常量与展示字符串是否逐字相同；
- v2 JSON schema、数字字段的允许类型、重复键策略和 canonical 数组的完整线格式。

因此无法审核：complex128 如何编码、shape 与 payload 长度是否一致、是否允许丢失符号零、十进制是否精确 round-trip、超长/错形 payload 是否在分配前拒绝、以及 decode 后 dtype 是否仍被限制为 float64/complex128。Link 3 的**内存中 hash preimage**可以审核，但 “序列化安全 + 独立验证器从字典恢复相同输入”不能确认。

**最小修复：**把上述完整定义、一个真实证书 JSON、对应预期 digest 和正式 schema 全部加入下一版送审件；不得要求审稿人猜测仓库实现。

### B3 — 满足 A1–A5 的有限输入不一定产生证书（覆盖性反例）

取

```python
H = np.array([[np.finfo(np.float64).max]], dtype=np.float64)
psi = np.array([1.0], dtype=np.float64)
```

它满足 A1–A5，且精确 Rayleigh 商等于

```text
0x1.fffffffffffffp+1023 = 1.7976931348623157e308.
```

对这个 1×1 计算，Arb 端点是精确的 binary64 最大有限值；但附件行 142 无条件执行

```python
math.nextafter(DBL_MAX, math.inf) == math.inf
```

随后行 144–145 拒绝。故“对有限自伴 $H$ 和非零 $\psi$，HTF 产生证书”按自然全称理解为假。

这不是不健全：拒绝比错误接受安全；但它推翻了当前覆盖范围的精确陈述。

**最小修复二选一：**

1. 把主张改成偏函数：仅当区间计算和有限 binary64 导出成功时产生证书，并明确可能拒绝满足 A1–A5 的输入；或
2. 使用真正的定向转换，或只在最近舍入值位于端点内侧时才 `nextafter`。若端点已经精确等于 `DBL_MAX`，应保留 `DBL_MAX`，不应无条件扩成 `inf`。

FLINT 文档说明 exact input 且结果可精确表示时输出保持 exact，并说明球结果包含精确运算结果；参见 [FLINT `arb` 文档](https://flintlib.org/doc/arb.html)。

### B4 — 有限端点可产生无限 midpoint/radius（阻塞）

取 `H=[[1e308]]`, `psi=[1.0]`。在 1×1 精确路径上，端点导出为相邻有限数：

```text
lower = 9.999999999999998e307
upper = 1.0000000000000002e308
```

但附件行 282–285 的 binary64 运算得到：

```text
(lower + upper) / 2 = inf
radius = inf
```

生产器只检查 Link 2 端点有限，不检查最终 `midpoint` 和 `radius`。两条验证器又都不重算或检查 `lower/midpoint/radius`。因此，在修复 B1 后，一个带非有限元数据的 `assurance="rigorous"` 对象仍可能被标记 verified。

**最小修复：**若 v2 的可信声明只有 `upper`，删除或明确标为非可信展示字段；若这些字段属于证书区间，则用 `Fraction.from_float` 对两个端点做精确中点/半径构造并向外导出，检查全部字段有限且满足精确包含关系，验证器也必须重算它们。

### B5 — “假设已验证”文本可陈述假命题（重大）

取 `H=[[0.0]]`，并取 `psi=[0x0.0000000000001p-1022]`，即最小正次正规 binary64。它精确非零，精确范数平方为 $2^{-2148}>0$，但附件行 100 的 NumPy 点积下溢为 `0.0`，行 106 写出：

```text
<psi|psi> = 0 > 0
```

数学检查 A5 本身仍正确，因为它使用分量的精确非零检查；错误在于证书的 assumptions 文本冒充已计算的正范数。

**最小修复：**删除这个普通浮点范数值，改为：

```text
|psi> has at least one exact non-zero binary64 component; hence the exact dyadic norm square is > 0
```

另将附件行 251 的 `.ravel()` 改为保留原维数的 `np.asarray(psi)`；否则二维输入会被悄悄展平，A2/A5 对“原始输入是一维向量”的表述不实。

## 4. 逐链接裁决

| 链接 | 裁决 | 独立判断 |
|---|---|---|
| Link 1 — preconditions | `PARTIAL` | 对 canonical 后的有限 binary64/complex128 数组，square、matching length、exact Hermitian、exact non-zero 足以应用有限维定理。但 `.ravel()` 改变原始输入语义，展示范数可下溢/上溢且陈述错误。 |
| Link 2a/2b — Arb/Acb | `PARTIAL` | 球算术包含原理及成功返回时的上界健全性确认；`float(upper()) + nextafter(+∞)` 是保守导出。无条件多走一 ULP 导致 DBL_MAX 反例，依赖版本未绑定，固定精度可能拒绝或给出过宽界。 |
| Link 3 — digest | `PARTIAL` | 内存 preimage 的域分离、路径标志、shape、长度前缀、大端 binary64 设计无明显结构碰撞；但 codec 缺失，且无密钥 hash 只提供自洽性，不提供来源认证。 |
| Link 4 — production | `REFUTED` | `upper` 的数学方向可正确；但 `1e308` 反例生成无限 midpoint/radius，完整证书类/schema 又缺失，不能确认 rigorous v2 对象的语义。 |
| Link 5 — verification | `REFUTED` | 两个给出的实现均有成功路径 `NameError`；即使修复，仍不检查完整 interval，且共享算术不能发现系统性 primitive bug。 |

## 5. 对 Q1–Q7 的直接回答

### Q1 — 前提是否充分？

**数学上：是。实现表述上：部分。**

有限 binary64/complex128 分量可视为精确二进有理数。先拒绝 NaN/Infinity 后，`np.array_equal(H, H.conj().T)` 的数值相等足以说明这些精确值构成 Hermitian 矩阵；至少一个分量非零则精确范数平方严格为正。因此定理适用。

但代码先 `.ravel()`，所以它没有检查调用者原始 `psi` 是一维；普通浮点 `norm_sq` 也不能作为严格证明值。A1–A5 应被描述为**canonical 后数组**的前提，或移除展平。

### Q2 — `float(endpoint)` 后 `nextafter` 是否向外安全？

**是，成功得到有限结果时安全；但无条件扩一 ULP 不完备。**

python-flint 的 `upper()` 先在当前精度生成一个向 $+\infty$ 舍入的 exact Arb 浮点上界；其 `arb.__float__` 调用 `arf_get_d(..., ARF_RND_NEAR)`，即正确的 binary64 最近偶数舍入。若最近值已在端点外侧，再向上一步仍在外侧；若最近值在内侧，正确最近舍入保证相邻向上 binary64 已越过端点。因此一步足够。下端点同理。

来源：[python-flint `arb.pyx`](https://github.com/flintlib/python-flint/blob/main/src/flint/types/arb.pyx)、[FLINT `arf_get_d` 与舍入模式](https://flintlib.org/doc/arf.html)。

边界缺陷是：端点已经等于最大有限 binary64 时，无条件 `nextafter` 变成 `inf` 并被拒绝。建议定向转换或条件扩张。

### Q3 — Acb 虚部包含零检查会怎样？

在 A4 确实执行、输入球包含精确二进值、Acb 运算满足其包含语义的前提下，精确商是实数，所以结果复矩形必须包含该实数，虚部球必含 0。固定精度只会使球更宽，不会把被包含的精确值排除；因此不应对真正 Hermitian 输入产生数学意义上的假阳性失败。

虚部检查**不能替代 Hermiticity 检查**。若绕过 A4，非 Hermitian 矩阵也可对某个试态给出实商。例如

```python
H = np.array([[1, 1], [0, 1]], dtype=np.complex128)
psi = np.array([1, 0], dtype=np.complex128)
```

此时 $H\ne H^*$，但试态商为 1，虚部检查会通过。这不是当前流水线内的错误，因为 Link 1 应先拒绝它；它说明 SC-5/SC-6 只能称为 sanity guard。

Acb 是实部、虚部两个 Arb 球组成的矩形包含，参见 [python-flint `acb` 文档](https://python-flint.readthedocs.io/en/latest/acb.html)。

### Q4 — digest 是否足以绑定 `(H, psi)`？

**作为内容寻址/内部一致性：设计合理。作为认证或当前线格式证明：不足。**

- 域分离、R/C 标志、shape、字段名和长度前缀消除了所列的拼接、维度和实/复别名歧义；
- 大端 `>f8` 给出平台无关的 binary64 字节；有限性检查排除了 NaN payload 的额外复杂性，符号零仍可区分；
- SHA-256 的常规 preimage/second-preimage 假设适用，参见 [NIST FIPS 180-4](https://csrc.nist.gov/pubs/fips/180-4/upd1/final)。

但该 digest 不是 MAC 或数字签名。攻击者可以修改 H/psi，重新计算 digest 和一个新上界，生成另一份内部自洽证书；没有外部可信 digest/签名时，验证器无法证明这是原始发布者的输入。所谓 length-extension 也不是这里的主要问题：系统没有把 SHA-256 当作 secret-prefix MAC，且验证的是整个结构化 preimage；真正缺少的是认证。

更直接的 Gate-A 间隙是 `_encode_canonical/_decode_canonical` 未给出，无法确认线格式按上述字节语义无损往返。

### Q5 — 语义字段是否足以防止弱路径冒充 rigorous？

**否。**

修复变量名后，`backend == recomputed_backend` 能阻止仅篡改证书字符串的简单攻击，因为本地代码实际调用某一 primitive 后再比较标签。但字符串本身不是算术质量证明：同一或被替换的 primitive 完全可以执行较弱算法并仍返回固定标签。`"flint-arb/prec=128"` 也没有记录 python-flint/FLINT 版本、构建或实现 hash。

信任根是“本地验证器代码 + 它实际加载的算术库 + 输入 codec”，不是证书中的 `assurance/theorem/backend` 文本。应记录并锁定依赖版本，并用不同实现重算精确商或区间；版本字段只改善可追溯性，不能替代独立算术。

此外：对象验证器不核对 `claim`；两条验证器都不核对 `lower/midpoint/radius`。若这些字段保留为证书语义的一部分，必须重算或明确宣布不可信。

### Q6 — 当前独立性实际能保证什么？

在先修复 B1 的前提下：

- **(a) 改小 `upper`：**若小到低于重算上界会被发现；改成任何仍覆盖重算上界的更松值会通过，但数学主张仍真。当前设计验证“有效性”，不是逐字节不可篡改性。
- **(b) 只改 canonical H/psi：**旧 digest 会失配；若攻击者连 digest 一起重算，验证器会验证新的输入与上界。没有外部签名/可信 digest 时，不能判定其来源被篡改。
- **(c) `_arb_rayleigh/_acb_rayleigh` 的系统性 bug：**不能发现，生产器与验证器重放同一错误。

达到算术独立性的最小方案恰好适合此输入域：binary64/complex128 都是二进有理数，可用 Python `Fraction` 或整数尾数/指数**精确**计算 $\psi^*H\psi/\psi^*\psi$，再与 `Fraction.from_float(stored_upper)` 比较。随本审稿交付的 `independent_rayleigh_audit.py` 实现了这一路径，不导入 HTF 或 python-flint。要成为正式 v2 验证器，还必须补齐 B2 的 codec/schema，才能无猜测地读取真实证书。

### Q7 — 限制是否诚实、完整？

所列四项基本诚实，但**不完整**。至少应补充：

1. 只认证证书中 exact binary64/complex128 矩阵和向量；任何输入构造、量化、离散化、基底/体积/能量截断、物理建模误差均不在界内，不只是 bond dimension $\chi$；
2. 当前验证器只有 certificate-management independence，没有 arithmetic independence；
3. SHA-256 digest 不提供发布者认证或防止攻击者重签发另一份自洽证书；
4. 满足 A1–A5 仍可能因有限导出、固定精度或 denominator ball 包含零而被拒绝；
5. `prec=128` 标签未绑定 python-flint/FLINT 版本；
6. 当前摘录未定义线格式 codec，不能声称序列化已独立审核；
7. 若保留 midpoint/radius，它们可溢出且当前不受验证；
8. 未说明输入尺寸/JSON payload 的资源上限与拒绝服务边界。

“固定 128 位不适应 H 的 condition number”也不够精确：本计算的可用宽度还取决于试态尺度、分母下界和二次型求和中的消去。区间包含性仍可正确，但结果可能无用或被拒绝。

## 6. 独立数值交叉核验

独立程序没有调用附件函数，也没有调用 Arb/Acb；它把每个 binary64 分量转换为 `Fraction.from_float` 的精确有理数，精确计算复二次型，并用另一条 `numpy.linalg.eigvalsh` 路径对给定小矩阵的特征值作交叉检查。

| Anchor | 独立精确 Rayleigh 商 | 独立 (E_0) | 对附件陈述的判断 |
|---|---:|---:|---|
| 1: `diag(0,1,2)`, `[1,0,0]` | $0$ | $0$ | 确认。因 Arb 路径全 exact，所给代码导出 `upper = 0x0.0000000000001p-1022 = 5e-324`，故 $0\le upper$。 |
| 2: 同矩阵，等分量试态 | (1)（即使 `1/sqrt(3)` 已先舍入，三个分量仍完全相同，比例精确消去） | (0) | 确认。精确路径预期 `upper = 0x1.0000000000001p+0 = 1.0000000000000002`。 |
| 3: `[[1,i],[-i,1]]`, `[1,0]` | $1$ | 特征值 $0,2$，故 $E_0=0$ | 确认。全 exact，代码导出 `upper = 1.0000000000000002`，故 $0\le upper$。 |
| 4: `diag(0,1e-15)`, 等分量试态 | `2535301200456459 / 5070602400912917605986812821504`，其 binary64 值为 `5e-16` | (0) | 数学锚点确认。实际 Arb 球端点未在附件中给出，故不能核对主机声称的具体宽度；只要函数成功返回，向外上端点应为正。 |

至少 Anchors 1 与 3 已由不同算术路径精确重算，满足送审验收项。附件没有提供主机生成的实际证书 JSON 或精确 `lower/upper`，所以无法做逐字节输出比对；不能把“≈”当作一个可核验的数值锚点。

## 7. 独立验证程序与变异守卫

`independent_rayleigh_audit.py` 仅使用标准库和 NumPy，核心算术用 `fractions.Fraction`，覆盖：

- dtype、shape、有限性、exact Hermitian、exact non-zero；
- real/complex 精确 Rayleigh 商；
- 附件定义的结构化 SHA-256 preimage；
- schema/theorem/assurance/backend/claim/digest/upper 的内存模型检查；
- Anchors 1 与 3 的精确值和独立特征值交叉检查；
- schema、assurance、theorem、backend、claim、digest 的逐字段变异；
- 降低 upper 的变异；
- 只改 H 不改 digest 的变异；
- 同时改 H 与 digest、但保留不再成立的旧 upper 的对抗性变异；
- 非 Hermitian 输入拒绝、real/complex digest 路径分离。

执行结果：

```text
ALL_CHECKS_PASSED
```

SHA-256：

```text
ea29a06788bfa3cd1b206b9f41be641d5d7e2f2829707a215df32f423ff4e8a2
```

程序没有个人绝对路径、用户名、公司名或内部主机名。由于送审件缺失 `_encode_canonical/_decode_canonical` 和完整 schema，程序明确不猜测 v2 线格式；这正是 B2。补齐 codec 后，应把该 exact-rational 核心接到真实 v2 解析器上并加入仓库回归测试。

## 8. Gate-A 最小修复清单

以下全部完成并重新送审前，不应解除 `BLOCKED`：

1. 修复 Link 5a 的 `is_complex` / `upper_v`–`recomputed_upper`，以及 Link 5b 的 `backend`–`stored_backend`；加入 real/complex 两条端到端成功测试。
2. 提供 commit `82a5159` 的可核对仓库 URL 或不可变源码包、测试命令、依赖锁和测试日志；解释为何所附“完整源码”与 `verify_from_dict` 成功锚点声明矛盾。
3. 补入完整 v2 schema、证书类、encode/decode 和真实 JSON/digest fixture；验证 shape、dtype、payload 长度和非有限值。
4. 对 `DBL_MAX` 反例采用条件/定向 binary64 导出，或把生产保证明确改为可能拒绝的偏函数。
5. 修复或删除 midpoint/radius；若保留为可信字段，生产器和验证器均以精确关系检查，并增加 `1e308` 回归测试。
6. 删除普通浮点 norm 的“严格假设”文本，并停止静默 `.ravel()`，或明确 flatten 是输入规范的一部分。
7. 对 `claim/lower/midpoint/radius` 的信任边界作出唯一、机器可检验的定义；不得一边序列化为 rigorous interval，一边在验证时忽略。
8. 引入不同算术实现。对当前二进浮点输入域，优先采用 exact-rational 重算；若因规模采用另一套区间库，则必须独立于生产 primitive。
9. 在证书中记录 python-flint 与 FLINT 的精确版本；后端标签不得被描述为算术质量证明。
10. 扩充限制说明，覆盖输入/模型误差、认证缺失、可能拒绝、共享算术和资源上限。

## 9. 验收标准逐项映射

| 附件验收项 | 本审稿交付 |
|---|---|
| 1. Overall verdict | `BLOCKED` |
| 2. Links 1–5 | `PARTIAL, PARTIAL, PARTIAL, REFUTED, REFUTED` |
| 3. Q1–Q7 | 已逐题直接回答 |
| 4. Conditional exact edit | 本件不是 `CONDITIONAL`；仍给出精确变量名修复和全部复审条件 |
| 5. Blocked counterexample/gap + repair | B1–B5 含动态故障、三个数值反例和最小修复 |
| 6. Anchors 1 and 3 independent recomputation | exact `Fraction` + NumPy 独立路径完成 |
| 附加块 B | 已交付可运行脚本、变异守卫、成功标志和 SHA-256；线格式兼容被 B2 客观阻断 |

## 10. 非循环性最终检查

本审稿没有假设目标不等式。独立链条是：

1. 从 binary64 位值构造精确二进有理数；
2. 独立检查 Hermitian 与非零；
3. 精确计算 Rayleigh 商 (q)；
4. 独立比较 $q\le upper$；
5. 用有限维谱分解推出 $E_0\le q$。

因此数学审查无循环。当前 HTF 运行时验证器重用生产算术不是逻辑循环，但它是共同失效模式，不能被描述为算术独立。

**最终裁决维持：`BLOCKED`。**
