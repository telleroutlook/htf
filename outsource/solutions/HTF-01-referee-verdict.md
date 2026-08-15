# HTF-01 独立审稿裁决：Rayleigh Certificate

**审稿对象：** `HTF-01-rayleigh-cert-audit.md` 所给出的六链接流水线与
`rayleigh-cert/v1` 模式  
**审稿日期：** 2026-08-13  
**最终裁决：** **GATE-A BLOCKED**  
**可否把现有 v1 证书作为 certified evidence：** **否；在下列修复完成并重新送审前，只能标为 discovery-tier。**

---

## 1. 执行摘要

所引 Rayleigh–Ritz 数学事实本身成立，也与 RH、ζ 零点或任何谱实性猜想无关；但所给实现链不能推出它声称的结论。至少有四个彼此独立的阻塞性错误：

1. **A3 放行非自伴矩阵。** `max|H-H†| ≤ 1e-10` 不是定理的假设。存在通过 A1–A4、但使所写不等式直接失败的 2×2 反例。
2. **Arb/Acb 球到 Python float 的导出不是外向舍入。** 在完全合法的 `ctx.prec=64` 下，`float(mid)+float(rad)` 严格小于真 Rayleigh 商 `1/3`。
3. **v1 摘要编码有结构性同摘要。** 一个有效的 3×3 实输入与一个有效的 2×2 复输入能产生逐字节相同的摘要原文，而其 Rayleigh 商分别是 0 和 1；这不需要攻破 SHA-256。
4. **验证器的容差可认证假命题。** 对 `H=[1]`, `ψ=[1]`，存储 `upper=1-5×10^-15` 会通过所给容差比较，但真实 `E0=RQ=1`，所以证书文字 `E0 ≤ upper` 为假。

另有 NaN/Inf fail-open、复数带符号零不能按位重放、摘要端序未固定、证书安全字段未被语义核对、`verified` 可由对象自身声明、以及“自包含无密钥摘要不构成恶意篡改认证”等问题。因此这不是只补一句说明即可放行的 `CONDITIONAL`；当前代码路径确实会产生或接受不正确证书，必须判为 `BLOCKED`。

---

## 2. 审查范围、复现环境与独立性

本裁决只审查送审件中逐字给出的定义和代码；未假定仓库中未引用的实现会补救这些问题。特别地：

- `cert.validate()` 的实现没有给出；不能把未展示的语义检查当作已成立。
- 模式清单没有列出验证器实际依赖的 `_H_canonical`、`_psi_canonical`；本裁决按“它们确实被序列化”这一对作者最有利的解释继续审查。
- 实验环境为 CPython 3.12.13、NumPy 2.3.5、python-flint 0.9.0、FLINT 3.6.0；关键反例使用精确有理数复核，不依赖 NumPy 特征值输出作为证明。
- 独立脚本 `HTF-01-independent-audit.py` 未复制仓库实现；它用 `fractions.Fraction` 从 binary64 的精确有理值重算 Rayleigh 商，并用 python-flint 复现球算术导出错误。

脚本成功输出：

```text
ALL_CHECKS_PASSED
```

脚本 SHA-256：

```text
e8c94cf6cec274785400c6386598d96ae3c4f34db0c347d4ce29ae5ece1ff952
```

---

## 3. 数学定理核对

### 3.1 精确陈述

令 `H=H† ∈ C^{n×n}`，特征值按 `λ0≤⋯≤λn-1` 排列，且 `ψ≠0`。取酉对角化
`H=U diag(λj) U†` 并令 `c=U†ψ`，则

\[
\frac{\langle\psi,H\psi\rangle}{\langle\psi,\psi\rangle}
=\frac{\sum_j\lambda_j|c_j|^2}{\sum_j|c_j|^2}.
\]

右端是各 `λj` 的凸组合，所以

\[
\lambda_0\le R_H(\psi)\le\lambda_{n-1}.
\]

因此送审件的有限维结论正确；对 Hermitian `H`，商本身已严格为实数，外加 `Re` 是冗余而非错误。证明中不可删除的假设是：

- `H` **精确自伴**；
- `ψ` **精确非零**；
- 证书所指的 `H,ψ` 与算术实际处理的 `H,ψ` 是同一对对象。

### 3.2 引用核对

Courant–Hilbert, *Methods of Mathematical Physics*, Vol. I 的第 VI 章确实讨论变分法与特征值问题，引用方向合理；但送审件只写 `§VI.1`，没有版次、页码或定理号，不能作为精确的机器可审计定位。建议改为完整版本信息并附页码，同时保留上面的有限维谱分解证明，使本证书不依赖版次差异。

### 3.3 非循环性

**CONFIRMED。** 定理及本裁决的所有反例只使用有限维线性代数、IEEE-754 值的精确有理表示、SHA-256 和区间算术；没有使用 RH、RH 等价命题、ζ 零点位置或本任务范围之外的谱猜想。

---

## 4. 六个链接逐项裁决

| 链接 | 裁决 | 独立结论 |
|---|---|---|
| 1. 前提检查 | **REFUTED** | A3 不是自伴性；NaN 可穿透比较；A4 是易上溢/下溢的浮点代理。 |
| 2. 实 Arb 算术 | **REFUTED（核心 Arb 运算本身 confirmed）** | `arb_mat` 运算给出包含球，但 `mid±rad` 转 float 会丢失外包性。 |
| 3. 复 Acb 算术 | **REFUTED（共轭行向量 confirmed）** | `ψ†` 构造正确；自伴前提、零分母检查、虚部判据和 float 导出不正确。 |
| 4. SHA-256 摘要 | **REFUTED** | 摘要算法本身可用，但 v1 消息编码非域分离、非定长标记且端序未固定，存在结构性同摘要。 |
| 5. 编码/解码 | **REFUTED** | 普通有限实 float64 可往返；复数用算术重组会丢失 `-0.0` 位并使未篡改证书摘要失配。 |
| 6. 独立验证器 | **REFUTED / 部分定义缺失** | 容差接受假上界；NaN fail-open；安全字段未绑定；`validate()` 未给出。 |

### 4.1 Link 1 — 前提检查

#### 阻塞反例：近似对称不能替代自伴

取

\[
\varepsilon=5\times10^{-11},\qquad
H=\begin{pmatrix}0&\varepsilon\\0&0\end{pmatrix},\qquad
\psi=\binom{1}{-1}.
\]

逐项结果为：

```text
max|H-H^T| = 5e-11 ≤ 1e-10       A3 放行
<psi,psi>  = 2                    A4 放行
spec(H)    = {0,0}
Re(<psi,H psi>/<psi,psi>) = -2.5e-11
```

该 `H` 不是自伴矩阵，故定理中的 `E0` 根本没有定义。即使强行把最小实特征值取为 0，所声称的不等式也变成

```text
0 ≤ -2.5e-11
```

而直接为假。虚部检查不能补救：这个反例全实，Rayleigh 商的虚部恰为 0。

#### NaN/Inf fail-open

`nan > 1e-10` 与 `nan < 1e-30` 都是 `False`。因此下面两类输入均可穿过所给逻辑：

```python
H = [[nan]], psi = [1]
H = [[0]],   psi = [nan]
```

实际 python-flint 计算会把相应商传播为 `nan`；后续 `nan > stored+tol` 仍为 `False`，形成端到端 fail-open。

#### A4 的准确评价

- `norm_sq ≥ 1e-30` 在无 NaN 时确实蕴含当前浮点计算没有得到零，所以是比定理更强的筛选；它会错误拒绝很小但非零的合法向量。
- 它不是可靠性证明：NumPy 点积可上溢为 `inf` 或下溢为 0；Arb 的任意指数算术与这个阈值没有同一误差模型。
- 对 binary64 输入，正确的定理前提检查是先拒绝所有非有限分量，再用 `np.any(psi != 0)` 检查精确非零，并在球除法前另查分母球不含 0。

**Link 1 裁决：REFUTED。**

### 4.2 Link 2 — 实 Arb 算术

[FLINT 对 `arb_t` 的定义](https://flintlib.org/doc/arb.html)明确保证：对输入球中任意点做精确运算，其结果被输出球包含。因此矩阵乘法、内积和除法的**球内部计算原则正确**。问题出在球被导出成普通 float 的最后三行。

[python-flint 的 `arb.__float__` 源码](https://github.com/flintlib/python-flint/blob/0.9.0/src/flint/types/arb.pyx)使用 `ARF_RND_NEAR`，即最近舍入；`float(quotient.rad())` 也不是向上导出。分别最近舍入后再做一次 binary64 加法，不能证明所得端点包住原球。

#### 阻塞反例：合法高精度使上界向下舍入

取

```text
H = diag(1,0,0),  psi = (1,1,1),  ctx.prec = 64.
```

精确 Rayleigh 商为 `1/3`。实际 python-flint 0.9.0 给出：

```text
q.mid().man_exp() = (6148914691236517205, -64)
q.rad().man_exp() = (1, -65)
float(mid)         = 0x1.5555555555555p-2
float(rad)         = 0x1.0000000000000p-65
float(mid)+float(rad)
                   = 0x1.5555555555555p-2
```

最后的 float 严格小于 `1/3`，差值为

\[
-\frac1{54043195528445952}\approx-1.85037\times10^{-17}.
\]

因此函数返回的 `interval.upper` 不是认证上界。`ctx.prec=100,200,500` 同样复现；代码没有固定或记录全局精度，不能把默认 53 位的偶然行为当作规范。

#### 分母

对有限的精确 binary64 点输入和非零 `ψ`，数学分母严格为正；但验证程序仍必须在除法前检查 `denominator.contains(0)`，并在除法后拒绝非有限/不确定球。实际对含 0 的 Arb/Acb 分母做除法会得到 `nan` 球，而所给代码没有 fail-closed 检查。

#### 最小算术修复

使用 [python-flint 的 `lower()` / `upper()`](https://python-flint.readthedocs.io/en/latest/arb.html)取得定向 Arb 端点后，还必须把 Arb→binary64 的最近舍入再向外推进。例如对有限结果：

```python
lower = math.nextafter(float(q.lower()), -math.inf)
upper = math.nextafter(float(q.upper()),  math.inf)
```

这可能多放宽 1 ulp，但保持声音性。若要求最紧端点，应比较精确 `man_exp()` 后只在必要时推进。随后以实际存储的 `lower/upper` 重新定义 `midpoint/radius`，并验证其包络关系。

**Link 2 裁决：底层 Arb 包含运算 CONFIRMED；所返回的证书区间 REFUTED。**

### 4.3 Link 3 — 复 Acb 算术

- `psi_dag = [conj(psi[i])]` 是正确的 1×n 共轭转置；这一点 **CONFIRMED**。
- 对精确 Hermitian `H`，`Re(<ψ,Hψ>/<ψ,ψ>)` 等于 Rayleigh 商；这一点 **CONFIRMED**。
- Acb 是实部球×虚部球的矩形包络，[官方文档](https://python-flint.readthedocs.io/en/latest/acb.html)支持这种解释。

但整条链接仍失败：

1. A3 已允许非 Hermitian `H`，而 `imag_bound≤1e-8` 不可能反推自伴性；上面的全实反例虚部为零。
2. `abs(float(mid_im))+float(rad_im)` 仍包含最近舍入，`1e-8` 是任意数值阈值，不是定理假设。
3. 正确做法是在精确自伴检查后要求 `q.imag.contains(0)`；若不包含 0 就 fail closed，而不是允许固定阈值。
4. 实部端点沿用了 Link 2 的非外向 float 导出，所以同样不是 certified endpoint。
5. 除法前没有检查复分母球不含 0，除法后也未拒绝 `nan`/无限球。

**Link 3 裁决：REFUTED。**

### 4.4 Link 4 — 摘要安全与规范性

#### SHA-256 本身

SHA-256 是 [NIST FIPS 180-4](https://csrc.nist.gov/pubs/fips/180-4/upd1/final) 规定的散列算法；在消息编码无歧义且摘要有可信锚点时，把不同消息保持同一摘要被认为在计算上不可行。问题不在 SHA-256，而在送审件喂给它的消息没有域分离、类型、形状和固定端序。

#### 阻塞反例：无需 SHA 碰撞的相同原文

以下两对输入都通过“方阵、长度、精确 Hermitian、非零”检查：

| 输入 | `H` | `psi` | Rayleigh 商 |
|---|---|---|---:|
| 实路径，n=3 | `diag(1,0,1)` | `(0,1,0)` | 0 |
| 复路径，n=2 | `diag(1,0)` | `(1+i,0)` | 1 |

v1 实路径产生 12 个 float64；v1 复路径也产生 12 个 float64。按所给拼接次序，两者的 96 个原文字节**逐字节完全相同**，所以摘要也相同；在本次 little-endian 复现环境中为：

```text
5d49ac866e9b48df5b3e9ccdd996bea7b8dac77dd84ef55f66dafebb8b4efabb
```

这回答了 Q4 的“是否需要长度前缀”：在当前跨实/复域协议中，**需要域标签、形状和边界；否则已有结构性捷径。**

#### 端序与数值锚点不一致

[NumPy dtype 文档](https://numpy.org/doc/stable/reference/arrays.dtypes.html)说明未显式指定时采用硬件原生端序。送审代码的 `astype(np.float64).tobytes()` 因此不是跨平台 canonical bytes；附件锚点却手写了 big-endian `3f f0 ...`。

对锚点 `H=diag(0,1), psi=(1,0)`：

```text
附件所列 big-endian 字节的 SHA-256:
037b991da8e0441b30d0128476abafc45310ce3da3957b93fae02d5891c26bc1

常见 little-endian 主机上实际代码的 SHA-256:
6f574f263c46e7cad7afd874638ab085257979c17e71e703bd2c5560010b52b8
```

锚点的 `RQ=0` 与 Arb `upper=0` 已确认；摘要字节说明与代码不一致。

#### “tamper-detectable”的边界

自包含 JSON 中的无密钥摘要只能检查内部一致性或偶然损坏。攻击者可同时替换 `(H,ψ)` 和 `input_digest`，无需求碰撞。要声称对恶意篡改可检测，必须满足至少一项：

- 验证器从可信清单/调用参数取得预期摘要；或
- 对包含输入、区间、模式版本和关键元数据的规范化证书做数字签名/受认证封装。

**Link 4 裁决：REFUTED。**

### 4.5 Link 5 — 编码与解码

对**有限 float64 实值**，`numpy.float64 → Python float → numpy.float64` 数值与位模式可保持；CPython float 使用 IEEE-754 binary64，并用可往返的最短十进制表示。NumPy 的 [`tolist()` 文档](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tolist.html)对一般 dtype 警告“可能丢精度”，所以模式仍应把输入 dtype 限定为 binary64，而不能泛称任意 ndarray。

复路径存在具体反例。解码式

```python
np.array(real) + 1j * np.array(imag)
```

是浮点算术，不是按分量装配；它会改变复数实部或虚部的 `-0.0` 符号位。取复数零矩阵，并让 `psi[0]=1-0j` 的虚部位为负零：

```text
原始摘要: 520ed84f78c15b465e2894663d4123653c8dc5dd07b51607c111005b87091bab
解码摘要: 0a61b55b94db9397645a0b77a30c7756279850b2d256284f0ffec6d4ae5c3f77
```

以上两个哈希值来自本次 little-endian 复现环境；端序改变时具体哈希会改变，但失配仍存在。这是未篡改证书也会触发的摘要失配。最小修复是分配 `complex128` 后分别赋值：

```python
z = np.empty(real.shape, dtype=np.complex128)
z.real = real
z.imag = imag
```

或者在编码前明确把所有 `±0` 规范化为 `+0`，并让摘要与重放共同使用同一规则。若真正要求按位可移植，使用固定端序 16 进制字节或 `float.hex()` 比依赖普通 JSON number 更清楚。

NaN 与 Infinity 还会破坏标准 JSON 互操作；[RFC 8259](https://www.rfc-editor.org/info/rfc8259/)明确不允许这两类数值。故有限性检查既是数学要求也是序列化要求。

最后，`_H_canonical` 与 `_psi_canonical` 必须成为公开、版本化的 replay payload 字段；否则列出的 v1 模式不是验证器所需数据的完整模式。

**Link 5 裁决：REFUTED。**

### 4.6 Link 6 — 独立验证器

#### 阻塞反例：容差把假上界判为真

取

```text
H = [1], psi = [1], E0 = RQ = 1.
cert.upper = 1 - 5e-15
```

binary64 中 `cert.upper<1` 严格成立。所给代码计算

```python
tol = max(abs(cert.upper) * 1e-14, 1e-15)
if 1.0 > cert.upper + tol:
    reject
```

此条件为 `False`，故验证器接受；但证书声明 `1 ≤ cert.upper` 为假。认证不等式不能使用“接近即相等”的容差。序列化字段本来就是 binary64，可精确比较；若要跨格式，应用有向端点或精确十六进制/有理编码，不应放宽逻辑命题。

#### 其他失败

- `upper_v=nan` 时比较为 `False`，随后仍设置 `verified=True`。
- 只比较 `upper`，没有核对 `lower/midpoint/radius` 的包络关系，也没有核对 `claim` 中的数值是否等于 `interval.upper`。
- 没有证明 `backend` 是精确枚举；`startswith("flint-arb")` 可被任意后缀伪装。
- 没有核对 `theorem`、`assumptions`、`notes` 与实际检查相符。
- 没有要求 producer 使用的精度、python-flint/FLINT 版本被记录。
- `verified` 是证书内可写布尔值；验证器还原地把它设为真。应返回独立验证结果，而不把作者可写字段当作信任根。
- `cert.validate()` 未提供，故其是否拒绝 NaN、未知字段、矛盾区间或缺失 replay payload 无法确认。

最小逻辑修复是：先得到**外向舍入**的重算上端点 `U`，再严格要求 `U ≤ stored_upper`，不加容差；或者直接把 binary64 输入转为精确有理数并检查 `RQ ≤ stored_upper`。所有比较必须先拒绝 NaN/Inf。

**Link 6 裁决：REFUTED。**

---

## 5. 对 Q1–Q5 的直接回答

### Q1 — 四项前提是否充分？

**否。** A3 的近似自伴检查是决定性缺口；还缺有限性、受支持 dtype/精度转换、精确非零和球分母非零检查。`1e-30` 阈值既非定理假设，也不能证明数值可靠。

### Q2 — 实 Arb 是否给出 certified interval？

**中间球给出包含；返回的 float 区间不给出。** `arb_mat` 运算按 Arb 规则传播舍入误差，但 `float(mid)±float(rad)` 不是外向舍入，并已有 `RQ=1/3, ctx.prec=64` 的反例。除法前应显式拒绝含 0 分母球。

### Q3 — 复 Acb 是否正确？

**`psi_dag` 正确，整体链接不正确。** 对精确 Hermitian `H`，实部就是 Rayleigh 商；但固定 `1e-8` 虚部阈值不能认证自伴性，同样存在端点导出和零分母问题。

### Q4 — 摘要与往返是否安全？

**否。** SHA-256 本身没有发现实用碰撞攻击；但当前消息编码已有实/复跨域结构性同原文，端序未固定，复带符号零不能按位往返。自包含无密钥摘要还不能认证恶意篡改者同时改写摘要的情况。

### Q5 — 限制是否诚实？

**部分诚实，但模式不足。**

- `E0≤upper` 的文字本身没有声称 gap、MPS 截断误差或接近真实基态，所以这部分没有越界。
- 但“没有越界”不等于“已清楚披露”：`notes` 是任意字符串，模式没有必填的 `guarantee_scope` / `non_guarantees`。
- `numpy-float` 下写 `radius=0.0` 会向只看 JSON 的读者暗示精确性；应写 `certified=false` 且 `radius=null/unknown`，不能把发现层浮点误差伪装成零半径。
- `verified` 与 `certified` 必须分离：前者最多表示某验证器运行过，后者才表示严格数学不等式已被声音地验证。
- `claim/theorem` 在**精确 Hermitian、非零、同一输入、外向端点**这些修复后没有超出 Rayleigh–Ritz；在当前 A3 下则已超出。

---

## 6. 必须完成的最小修复清单

这些是从 `BLOCKED` 回到可重新送审状态的最低要求，不是可选增强。

### R1 — 版本升级并冻结输入语义

- 新建 `rayleigh-cert/v2`；不得让已存在的 v1 证书自动继承 certified 状态。
- 明确定义输入为有限 `float64/complex128` 数组，或定义从更宽 dtype 到 binary64 的受检转换；禁止静默不安全 cast。
- 模式公开列出 replay payload、形状、域、端序、证书精度和后端版本。

### R2 — 精确前提检查

按此顺序执行：

1. 形状和受支持 dtype；
2. `np.isfinite(H).all()`、`np.isfinite(psi).all()`；
3. `np.array_equal(H, H.conj().T)` 的精确自伴性；
4. `np.any(psi != 0)` 的精确非零；
5. Arb/Acb 分母球 `contains(0)` 时 fail closed。

若业务输入只有近似 Hermitian 矩阵，必须先显式生成并摘要绑定
`Hsym=(H+H†)/2`，且证书对象改为 `Hsym`；不能继续把原 `H` 写进同一 claim。

### R3 — 修复球端点导出

- 在局部上下文中固定工作精度并记录它；不要依赖外部可变 `ctx.prec`。
- 使用 `q.real.lower()/upper()`，再以 `nextafter` 或精确 `man_exp` 比较实现 binary64 向外舍入。
- 拒绝 NaN/无限/不确定输出；若允许 `+inf` 作为真但无用的上界，必须用标准可序列化表示而非非标准 JSON 数字。
- 由最终存储端点重新计算能覆盖两端的 `midpoint/radius`，并在验证器中检查关系。
- 对复路径要求精确自伴输入，并验证 `q.imag.contains(0)`；删除 `1e-8` 作为声音性判据。

### R4 — 新摘要编码

摘要原文至少包含：

```text
domain_tag = "rayleigh-cert-input/v2"
real_or_complex_tag
H.ndim, H.shape, psi.ndim, psi.shape
fixed dtype tags
fixed endian (例如 >f8)
每个字段的标签和字节长度
H/psi 各分量的 C-order bytes
```

必须加入上文 n=3 实 / n=2 复同摘要案例作为回归测试。若要抵抗恶意篡改，还要从可信外部取得预期摘要或验证对整份证书的签名。

### R5 — 位安全重放

- 复数用分别赋值 `.real/.imag` 的方式重建，或先规范化 `±0` 并在摘要与编码两侧共同执行。
- 拒绝 NaN/Inf；固定 JSON/二进制规范与端序。
- 编码后立即执行 `digest(original)==digest(decoded)` 的实、复、subnormal、最大有限值和四种带符号零组合测试。

### R6 — 验证器严格化

- 删除 `tol`；严格检查 certified 上界。
- 所有非有限比较 fail closed。
- 核对 `claim` 数值、四个 interval 字段关系、精确 backend 枚举、版本、工作精度、假设清单与 replay payload。
- 忽略传入的 `verified`，返回独立的验证结果；不要把证书自身布尔值当作证据。
- `numpy-float` 生产物不得带 certified claim；只有独立严格重放成功后才能另行出具认证结果。

---

## 7. 必须加入的对抗回归测试

| 测试 | 修复后预期 |
|---|---|
| `H=[[0,5e-11],[0,0]], psi=(1,-1)` | 前提检查拒绝 |
| H 或 psi 含 NaN/Inf | fail closed |
| `H=diag(1,0,0), psi=(1,1,1), prec=64/200` | 存储 `upper ≥ 1/3` |
| Arb/Acb 分母球含 0 | 除法前拒绝 |
| n=3 实 / n=2 复结构性同原文案例 | v2 摘要不同 |
| 复数任一分量为 `-0.0` | 编码—解码摘要完全相同 |
| `H=[1], psi=[1], upper=1-5e-15` | 验证失败 |
| 只改 `claim`、`backend`、`radius` 或 replay payload | 验证失败 |
| 把 producer backend 改成 `numpy-float` | 不得被标为 certified |

独立脚本已经覆盖上述关键反例以及“扰动输入摘要必须失败、扰动 claim 必须失败、减小上界必须失败”三类变异守卫。

---

## 8. 数值锚点最终核验

对送审锚点 `H=diag(0,1), psi=(1,0)`：

```text
numerator   = 0 exactly
denominator = 1 exactly
RQ          = 0 exactly
Arb lower   = 0
Arb upper   = 0
Arb radius  = 0
```

所以 `interval.upper ≤ 1e-9` 成立，且远弱于实际可得的精确 0。唯一错误是附件把 big-endian 手写字节称作当前代码的 canonical bytes；当前实现使用 native-endian，确切摘要已在 §4.4 给出。

---

## 9. Gate-A 最终裁决

### **GATE-A BLOCKED**

阻塞位置是 **Link 1、Link 2/3 的端点导出、Link 4/5 的输入绑定，以及 Link 6 的严格比较**。任一项单独都足以阻止 PASS；其中 A3 反例、`1/3` 向下导出反例和容差假上界反例都已经直接推翻流水线的声音性。

在 R1–R6 完成、全部对抗回归测试通过并重新独立送审以前：

- 不得声称 v1 证书是 certified；
- 不得把 `verified=True` 解释为数学上已证明；
- 可保留 v1 输出作 discovery/debug 记录，但必须显式标注非认证。
