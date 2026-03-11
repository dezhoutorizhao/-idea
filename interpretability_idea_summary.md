# 神经网络可解释性研究 Idea 总结

## 文档目的

本文档系统总结三个候选研究方向：

1. `Idea 1 + Idea 2 hybrid`：`Invariant Mechanism Discovery under Semantic and Representational Symmetries`
2. `Idea 1 单做`：`Symmetry-Calibrated Causal Interpretability`
3. `Idea 2 单做`：`Equivalence-Class Causal Tracing`

核心目标不是做一个“更花哨的解释算法”，而是正面解决当前机制可解释性研究中的两个基础性漏洞：

- 很多解释只在单个 prompt、单个表示坐标系下成立，缺乏可识别性
- 很多所谓“机制”其实只是局部实现、脆弱实现或坐标幻觉，而不是真正稳定的任务机制

---

## 一、问题背景：为什么需要新的机制可解释性方法

### 1.1 当前主流范式的默认假设

当前大量神经网络可解释性工作，尤其是 mechanistic interpretability 工作，通常隐含接受以下假设：

1. 如果在某个 prompt 上，通过 patching / ablation / attribution 找到一组有较强 causal effect 的组件，那么它们就是该任务的“机制”
2. 如果某个 neuron / head / feature 在当前表示空间中看起来可解释，那么这种解释就是可靠的
3. 如果某个 circuit 在一个实验 setup 中成立，那么它大概率具有更普遍的机制意义

这些假设使得研究者能够快速构建“任务 -> 电路 -> 可视化”的故事，但问题在于：这些故事可能并不稳定，也不一定可泛化。

### 1.2 当前范式的两个根本漏洞

#### 漏洞 A：语义脆弱性

同一个任务、同一个语义目标，仅仅改变 prompt 表述方式，解释结果就可能显著变化：

- 关键 attention heads 变化
- 因果热点位置变化
- 层级重要性变化
- 所谓“核心 circuit”不再稳定

这说明现有工作找到的可能只是：

- 某个 prompt 的局部实现方式
- 某个模板诱导出的策略
- 某种输入格式下的临时计算路径

而不是该任务真正的、跨语义等价表达都存在的机制。

#### 漏洞 B：表示坐标脆弱性

很多解释依赖于当前隐藏表示的坐标系，例如：

- 某个维度有特定语义
- 某些 neuron 代表某个概念
- 某些 sparse features 是“真实概念”

但内部表示通常存在大量对称性和等价变换自由度。只要模型的外部函数保持不变，不同的内部表示基底可能都能实现同样行为。于是：

- 解释对象可能不是“真实机制”
- 而只是当前表示基底下的一种投影产物

这意味着可解释性研究缺失了一个非常关键的问题：

`什么样的解释对象，在表示等价变换下仍然是稳定可识别的？`

---

## 二、方案一：Idea 1 + Idea 2 Hybrid

## `Invariant Mechanism Discovery under Semantic and Representational Symmetries`

这是最推荐的方案，也是三个方案里最像高水平 oral 论文的方案。

### 2.1 核心问题定义

我们提出以下新的研究问题：

> 一个真正有机制意义的解释，不应只在单个 prompt 和单个表示坐标系下成立；它应该在语义等价输入分布和允许的表示对称变换下，都保持稳定的因果作用。

换句话说，当前大量解释方法试图寻找的是：

- “在这个输入上有效的解释”

而这个 hybrid 方案试图寻找的是：

- “在语义和表示等价类上不变的机制对象”

这相当于把研究目标从 `instance-level explanation` 升级成 `equivalence-class mechanism discovery`。

### 2.2 新的核心洞见

真正的机制应当满足双重稳定性：

1. `语义稳定性`
   - 对同一个任务的不同自然表述，机制不能剧烈漂移
2. `表示稳定性`
   - 对不改变功能的表示等价变换，解释不能完全重写

如果某个解释只在一个 prompt 上强、换一句话就失效，那么它很可能不是任务机制。

如果某个解释只在一个坐标系下强、换一个等价基底就消失，那么它很可能只是坐标幻觉。

因此，真正可信的机制对象应该是某种“不变量”。

### 2.3 方法直觉

给定一个任务语义等价类：

\[
\mathcal{Q} = \{q_1, q_2, \dots, q_n\}
\]

给定一组可解释候选组件：

\[
C \subseteq \text{Heads / Layers / Features / Paths}
\]

给定一组允许的表示对称变换：

\[
g \in \mathcal{G}
\]

我们希望找到一组组件 \( C^\* \)，满足：

- 在不同 \( q \in \mathcal{Q} \) 上具有持续的因果贡献
- 在 \( g \in \mathcal{G} \) 下保持解释效果稳定
- 规模尽量稀疏、便于叙事和分析

可以写成如下目标：

\[
\max_C \ \mathbb{E}_{q \in \mathcal{Q}}[\Delta_q(C)]
- \lambda \cdot \mathrm{Var}_{q \in \mathcal{Q}}[\Delta_q(C)]
- \gamma \cdot \mathbb{E}_{g \sim \mathcal{G}}[\mathrm{Fragility}(g, C)]
- \beta |C|
\]

其中：

- \( \Delta_q(C) \)：组件 \( C \) 在 prompt \( q \) 上的 causal effect
- \( \mathrm{Var}_{q \in \mathcal{Q}} \)：测量跨语义等价实例的机制波动
- \( \mathrm{Fragility}(g, C) \)：测量解释对表示变换的脆弱性
- \( |C| \)：控制解释复杂度

### 2.4 这个方法到底“新”在哪里

它不是：

- 又做一个新的 patching 技巧
- 又发明一个新的 saliency 分数
- 又换一种电路可视化方式

它真正改的是：

`什么样的对象才配被称为机制`

这是问题定义层面的创新，而不是实现层面的微调。

### 2.5 论文故事怎么讲

这个方案最强的地方在于 story 非常完整：

#### 第一幕：前人做了什么

前人通过 activation patching、ablation、circuit analysis 等方法，在具体 prompt 上找到了许多“机制”。

#### 第二幕：为什么现有方法不够

这些机制往往默认：

- 单 prompt 足以代表任务
- 单一表示坐标足以定义解释对象

但这是两个未经证明的强假设。

#### 第三幕：我们发现了什么新问题

我们指出：

- 许多解释在语义变体之间不稳定
- 许多解释在表示等价变换下不可识别

因此，当前方法可能提取的是：

- 局部实现
- 脆弱电路
- 坐标依赖伪机制

#### 第四幕：我们如何解决

我们提出一种新的机制定义与发现框架：

- 只把那些在语义等价类与表示对称变换下都保持因果稳定的组件视为“机制”

#### 第五幕：我们带来了什么结果

我们不仅能识别更稳定的机制，还能：

- 更好恢复 toy task 的 ground-truth mechanism
- 提升跨 prompt 的因果一致性
- 降低解释对 basis choice 的依赖
- 产生更紧凑、更可叙事的机制子图

### 2.6 预期实验设计

#### 数据层

- 合成任务：
  - modular arithmetic
  - bracket matching
  - entity retrieval
  - controlled factual recall
- 自然语言模板任务：
  - country-capital
  - factual association
  - subject-verb agreement
  - simple reasoning templates

每个任务为同一语义目标构造多个等价 prompt。

#### 模型层

- `gpt2`
- `gpt2-small`
- `pythia-160m`
- `pythia-410m`

这些模型适合在 3080 10GB 上进行 patching、激活缓存和中小规模 sweep。

#### Baseline

- 单 prompt activation patching
- 单 prompt head / feature ranking
- 普通平均 causal effect 选择
- 不加 symmetry regularization 的稳定性版本

#### 评估指标

1. `Average Causal Effect`
2. `Cross-Paraphrase Variance`
3. `Mechanism Jaccard / Overlap`
4. `Basis Robustness Score`
5. `Ground-Truth Recovery Accuracy`
6. `Mechanism Sparsity`

### 2.7 为什么这个方案最像 oral

因为它同时具备：

- 上游问题定义创新
- 明确的理论动机
- 自然的算法实现
- 可以量化的指标提升
- 可以生成漂亮图和说服力强的 case study

它兼具“批判现有范式”和“提出新范式”两个层次，这是 oral 级论文很重要的特征。

### 2.8 风险与难点

- 如何定义“合法的表示对称变换”
- 如何避免 reviewer 认为你只是做 robustness analysis
- 如何保证 symmetry regularization 不会把真正的局部机制也误杀

### 2.9 适合的论文定位

- 方法学论文
- 理论驱动的机制可解释性论文
- 有很大机会冲 oral，但必须把“新定义”和“指标收益”都讲扎实

---

## 三、方案二：Idea 1 单做

## `Symmetry-Calibrated Causal Interpretability`

这个方案是三个方案中理论味最强、novelty 最高、但风险也最高的方案。

### 3.1 核心问题定义

我们单独研究一个根本问题：

> 如果一个解释在功能等价的内部表示变换下不稳定，那么它还能被称为“解释”吗？

这个问题试图从第一性原理上挑战当前可解释性研究默认接受的坐标依赖性。

### 3.2 核心洞见

可解释性方法常常在问：

- 哪个 neuron 重要
- 哪个 feature 代表某概念
- 哪个子空间和某语义相关

但这些问题默认这些对象本身是有意义的。

然而在很多神经网络里：

- 单个 neuron 不是自然对象
- 单个维度不是不变量
- 某组 feature 是否“真实存在”取决于表示基底

因此，解释方法必须先回答一个更基础的问题：

`解释对象是否可识别`

### 3.3 方法目标

该方案的目标不是直接发现机制，而是提出一个新的“解释可信度准则”：

> 一个高质量解释，必须在一组不改变模型功能的表示变换下，保持相对稳定的因果性质。

### 3.4 形式化思路

给定模型隐藏表示 \( h \)，以及解释算法输出 \( E(h) \)。

考虑一组变换 \( g \in \mathcal{G} \)，这些变换满足：

- 近似保持下游功能不变
- 但会重写当前表示坐标

定义解释稳定性：

\[
\mathrm{StableFaithfulness}(E) =
\mathbb{E}_{x}[\mathrm{Effect}(E, x)]
- \lambda \cdot \mathbb{E}_{g \sim \mathcal{G}, x}[\mathrm{Drift}(E, g, x)]
\]

其中：

- \( \mathrm{Effect}(E, x) \)：解释所对应组件的因果效应
- \( \mathrm{Drift}(E, g, x) \)：解释结果在变换前后的漂移

### 3.5 可能的方法输出

这个方案可以输出：

- 一个新的解释评估指标
- 一套 symmetry-aware 排序准则
- 一个“稳定解释”筛选器
- 一个判断某类解释是否属于坐标幻觉的 framework

### 3.6 这个方案为什么有理论张力

它不是说：

- “以前方法做得不好，我做得更稳”

而是在说：

- “以前方法可能在对象层面就定义错了”

这使得它天然带有更强的理论冲击力。

### 3.7 论文故事怎么讲

#### 第一幕：现有解释默认坐标有意义

大量方法基于 neuron、head、feature、subspace 等对象定义解释。

#### 第二幕：但这些对象未必可识别

在内部表示等价变换下，行为可保持，但解释可能完全重写。

#### 第三幕：所以 faithfulness 还不够

一个 explanation 即使在当前坐标下“faithful”，也可能不是“identifiable”。

#### 第四幕：我们提出 symmetry-calibrated faithfulness

解释质量不仅取决于因果效应，还取决于对对称变换的稳定性。

#### 第五幕：我们证明传统方法会被坐标幻觉欺骗

通过 toy task 和受控网络，展示：

- 预测能力不变
- 解释重要性排序变化巨大

### 3.8 适合的实验

这个方案最适合：

- 合成任务
- 小型 transformer
- 可控 MLP
- 具有 ground-truth mechanism 的受控环境

例如：

- parity
- modular addition
- linear concept tasks
- finite-state tasks

### 3.9 优点

- novelty 非常强
- 理论锋芒足
- 很容易写出“前人忽略了更根本的问题”的强叙事

### 3.10 缺点

- 容易变成纯 critique paper
- 审稿人会追问：除了告诉别人错了，你提供了什么更好的机制发现能力？
- 如果实验只停留在 toy task，可能被认为工程闭环不足

### 3.11 最适合的定位

- 高风险高收益
- 适合做方法学或理论导向论文
- 如果没有强实证闭环，更像“漂亮的 workshop / strong regular paper”，不一定稳进 oral

---

## 四、方案三：Idea 2 单做

## `Equivalence-Class Causal Tracing`

这个方案是最实用、最好落地、最容易快速形成完整实验闭环的方案。

### 4.1 核心问题定义

当前 activation patching / circuit tracing 通常在单个 prompt 上找机制。但真正任务机制应该对语义等价表达保持稳定。

因此，我们提出：

> 不应解释单个 prompt 的电路，而应解释一个语义等价类的共享机制。

### 4.2 核心洞见

同一任务的多个表达方式可能诱导出不同的局部注意模式，但真正机制应是这些局部实现背后的稳定子结构。

例如，对于同一个国家首都查询：

- “The capital of France is”
- “France's capital city is”
- “In France, the capital is”

如果解释方法在这些 prompt 上给出完全不同的核心组件，那么它更可能提取的是模板依赖策略，而不是任务机制。

### 4.3 方法思路

构造语义等价集合：

\[
\mathcal{Q} = \{q_1, q_2, \dots, q_n\}
\]

对每个候选组件集合 \( C \)，计算：

\[
\text{Score}(C)=
\mathbb{E}_{q \in \mathcal{Q}}[\Delta_q(C)]
- \lambda \cdot \mathrm{Var}_{q \in \mathcal{Q}}[\Delta_q(C)]
- \beta |C|
\]

这意味着理想的机制应当：

- 平均因果效应大
- 跨等价 prompt 波动小
- 复杂度低

### 4.4 与现有方法的区别

现有方法通常优化：

- 在一个样本上最大化 causal effect

而该方案优化：

- 在一个语义类上最大化平均因果效应，同时惩罚机制不稳定性

这相当于把“机制发现”从局部实例问题改写成集合级别问题。

### 4.5 可能的算法实现

可以有多种实现路线：

1. `Head-level search`
   - 枚举或 beam search 找稳定 heads
2. `Path-level search`
   - 找跨 prompt 都高贡献的因果路径
3. `Feature-level search`
   - 在 feature basis 上找共享解释对象
4. `Cluster-level tracing`
   - 先聚合 prompt-specific circuits，再求共享子图

### 4.6 论文故事怎么讲

#### 第一幕：现有工作为什么有问题

现在很多 mechanistic interpretability 工作默认：

- 一个 prompt 上的机制就是该任务机制

但这忽略了任务机制与输入实现之间的区分。

#### 第二幕：我们发现了什么

我们发现，在语义等价 prompt 上：

- 头级归因结果差异显著
- 因果组件不稳定
- 单 prompt circuit 具有明显模板依赖性

#### 第三幕：我们如何重构问题

我们不再解释单个 prompt，而是解释语义等价类。

#### 第四幕：我们的方法优势

相比现有方法，我们发现的机制：

- 更稳定
- 更紧凑
- 更能跨模板迁移
- 更接近任务本身而非模板表面模式

### 4.7 为什么这个方案容易落地

- 不需要特别复杂的新理论结构
- 数据可以用模板和 LLM 重写自动构建
- 在 `gpt2-small` 和 `pythia` 小模型上就能做出系统实验
- patching、ablation、overlap、variance 等指标都容易实现

### 4.8 它最可能带来的指标收益

该方案很容易在以下指标上优于 baseline：

- 跨 prompt causal consistency
- 机制集合 overlap
- 对 unseen template 的泛化稳定性
- 解释复杂度与稳定性的 Pareto 改善

### 4.9 优点

- 工程可行性高
- 数据和模型门槛低
- 很容易出系统图表
- 很适合 poster 和短周期验证

### 4.10 缺点

- novelty 不如 hybrid 和 Idea 1 纯粹
- 容易被 reviewer 视作“稳定性增强版 circuit tracing”
- 如果没有很强的理论 framing，可能更像一篇扎实的 empirical method paper

### 4.11 最适合的定位

- 中低风险
- 非常适合做第一篇可解释性论文
- 很有希望出强 regular paper，但单独冲 oral 需要额外理论提升

---

## 五、三个方案的横向比较

| 维度 | Hybrid | Idea 1 单做 | Idea 2 单做 |
|---|---|---|---|
| 问题是否根本 | 很强 | 最强 | 中等偏强 |
| novelty | 很强 | 最高 | 中等 |
| story 完整度 | 最高 | 强，但偏批判 | 强，偏工程闭环 |
| 指标提升可实现性 | 很高 | 中等 | 很高 |
| 理论深度 | 很强 | 最高 | 中等 |
| 3080 可行性 | 高 | 高 | 很高 |
| oral 潜力 | 最高 | 高风险高上限 | 中等偏高 |
| 实验闭环难度 | 中等 | 中高 | 最低 |

---

## 六、最终建议

### 如果目标是“最有希望冲 oral”

优先选择：

`Idea 1 + Idea 2 Hybrid`

原因：

- 既有更根本的问题定义创新
- 又有清晰的方法实现
- 还能在稳定性和因果指标上做出实证提升
- 最容易讲出“前人没真正解决的问题 -> 新定义 -> 新方法 -> 更强结果”的完整故事

### 如果目标是“追求最高 novelty，不怕高风险”

优先选择：

`Idea 1 单做`

原因：

- 它最有理论冲击力
- 最容易形成“范式漏洞攻击型”论文
- 但必须补足实验闭环，否则容易变成 critique paper

### 如果目标是“尽快形成一篇扎实、可跑通、容易验证的工作”

优先选择：

`Idea 2 单做`

原因：

- 最容易上手
- 最适合当前硬件
- 最容易在短周期内拿到可发表图表和指标

---

## 七、最推荐的落地路线

如果从研究产出和风险收益比来看，最推荐的路径是：

1. 先以 `Idea 2` 作为第一阶段原型
   - 先证明单 prompt 机制在语义等价类上不稳定
   - 建立跨 prompt 的稳定机制发现 pipeline
2. 再引入 `Idea 1` 的表示对称性约束
   - 把方法从“稳定机制搜索”升级为“对称性校准的不变量机制发现”
3. 最终形成 hybrid 论文

这样做的优点是：

- 先拿到实证结果
- 再抬升理论高度
- 风险更可控
- 更适合当前 3080 设备与中短周期科研推进

---

## 八、一句话版本总结

- `Hybrid`：重新定义“什么才算机制”，要求机制同时对语义变体和表示对称变换稳定，这是最有 oral 潜力的方案
- `Idea 1 单做`：从第一性原理攻击当前可解释性范式的坐标依赖性，novelty 最高，但风险也最高
- `Idea 2 单做`：把机制解释从单个 prompt 提升到语义等价类，最容易落地，最适合作为强实证起点
