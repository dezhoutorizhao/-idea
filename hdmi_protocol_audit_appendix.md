# HDMI Protocol Audit and Reproduction Appendix

本文档整理了当前针对 HDMI 论文与补充代码的协议审计、关键任务复现实验、代表性任务 case study，以及当前实验设置的主要 validity threats。本文档的定位不是“证明我们已经严格复现了 HDMI 原文全部结果”，而是：

1. 明确哪些设置已经与论文/补充代码严格对齐；
2. 明确哪些设置只能视为 HDMI-compatible；
3. 指出当前 released materials 无法支撑 exact reproduction 的具体原因；
4. 为后续论文主文和附录提供可 defend 的实验透明性材料。

相关实验文件：

- `D:\神经网络可解释性\compare_hdmi_hybrid_exact.py`
- `D:\神经网络可解释性\compare_hdmi_race_exact.py`
- `D:\神经网络可解释性\compare_hdmi_race_all.json`
- `D:\神经网络可解释性\compare_hdmi_hybrid_exact_seeded.json`
- `D:\神经网络可解释性\hdmi_repro_appendix_probe_epochs.json`
- `D:\神经网络可解释性\race_case_study_diag.json`
- `D:\神经网络可解释性\hdmi_supplementary\hdmi-code\probing.py`
- `D:\神经网络可解释性\hdmi_text.txt`
- `D:\神经网络可解释性\hdmi_table1_excerpt.txt`

## Audit

### A. 审计目标

本节回答三个问题：

1. 当前实验设置是否完全遵循 HDMI 论文描述？
2. 当前实验设置是否完全遵循 HDMI 补充代码？
3. 论文描述、补充代码和当前实现三者之间是否存在系统性不一致？

结论先行：

- 当前实现已经较高程度遵循了 HDMI 已公开、可验证的核心评测与干预逻辑；
- 但当前实现不能被称为“exact author pipeline”；
- 更准确的表述应为：`HDMI-compatible local protocol`。

### B. 协议项逐项审计

| 协议项 | 论文/补充材料表述 | 当前实现 | 结论 |
|---|---|---|---|
| 模型 | `EleutherAI/pythia-70m` | 使用 `EleutherAI/pythia-70m` | exact |
| 干预层 | final layer, `layer_idx=-1` | 使用 `layer_idx=-1` | exact |
| HDMI 目标 | margin objective | 使用 `target logit - source logit` | exact |
| Completeness 定义 | TV 到 counterfactual one-hot | 直接沿用 `probing.py` | exact |
| Selectivity 定义 | pre/post `vZe` 的 TV 归一化 | 直接沿用 `probing.py` | exact |
| Reliability 定义 | 调和平均 | 直接沿用 `probing.py` | exact |
| CausalGym 样本化 | 每个 minimal pair 转成两个样本 | 与 `make_causalgym_samples` 对齐 | exact |
| 评测逻辑 | 比较 gold/alt continuation，替换 first-step logprob | 直接复用了相同思路与函数 | exact |
| HDMI 超参 | 论文 Table/Appendix 写 `alpha=1`, `inner_steps=30` | 手动设为 `1` 和 `30` | compatible to paper, not code default |
| 补充代码默认 HDMI 超参 | `hdmi_alpha=25`, `hdmi_inner_steps=1` | 未采用默认值 | incompatible with code default |
| probe holdout | 论文写 split 内部随机 holdout 20% | 当前实现使用 80/20 | exact |
| validation probe hidden search | 补充代码 `hidden_grid=[0,64,256,512]` | 当前实现一致 | exact to code |
| probe optimizer/batch/lr/wd | 补充代码默认 `AdamW`, `lr=1e-2`, `wd=1e-6`, `bs=256` | 当前实现一致 | exact to code |
| probe epochs | 论文 Appendix 写 `{75,100}`；代码默认 `150` | 当前实现使用 `150` | inconsistent between paper and code |
| CausalGym validation decorrelation | 论文写 validation-probe split 应近似满足 `Zc ⟂ Ze` | 当前实现没有对 CausalGym 做显式 decorrelation | only compatible, not exact to paper text |
| `resplit_3way` / `decorrelate_val` | 代码定义了参数 | 在 `probing.py` 主流程中未真正触发 CausalGym 路径重拆分/去相关 | code-text mismatch |
| 数据版本 | 论文未给固定 revision | 使用 Hugging Face `aryaman/causalgym` 当前版本并转本地 JSON | compatible |
| task-specific baseline stabilization | 论文提到某些 Pythia-70M 任务会调整 interventional split 比例 | 没有作者原始 task-specific config | unknown |

### C. 论文文本与补充代码的关键不一致

#### C.1 HDMI 超参不一致

论文 Appendix/Table 4 给出的 HDMI 超参为：

- `hdmi alpha = 1`
- `hdmi inner steps = 30`

但 `D:\神经网络可解释性\hdmi_supplementary\hdmi-code\probing.py` 中默认参数是：

- `hdmi_alpha = 25.0`
- `hdmi_inner_steps = 1`

这说明：

- 单独运行补充代码默认命令，不会得到论文文字中声明的 HDMI 设置；
- 如果不手动覆盖这两个超参，则不能宣称对齐论文主实验。

#### C.2 probe epochs 不一致

论文 Appendix/Table 4 指出：

- `probe epochs ∈ {75, 100}`

但补充代码默认值是：

- `probe_epochs = 150`
- `valprobe_epochs = 150`

当前实现为了贴近补充代码路径，在大量实验中用了 `150`。这使得当前实验：

- 更接近 released code；
- 但不完全符合论文文字描述。

#### C.3 CausalGym validation decorrelation 不一致

论文正文明确写到：

- validation-probe split 应经过 subsample，使 `Zc` 与 `Ze` 近似独立，以避免 `vZc` 利用 `Ze` 作弊。

但审查 `D:\神经网络可解释性\hdmi_supplementary\hdmi-code\probing.py` 后发现：

- `make_independent_subset` 仅在 LGD 路径中被调用；
- CausalGym 主流程默认直接使用 dataset 自带 train/dev/test；
- `--resplit_3way` 和 `--decorrelate_val` 虽被定义，但没有在主流程中真正落实到 CausalGym 实验。

这意味着：

- 论文对 CausalGym 的协议描述比 released code 更严格；
- 当前代码基础上无法证明作者公开实验确实满足该 decorrelation 条件。

### D. 协议审计结论

当前最可 defend 的说法是：

> 我们实现并审计了一个基于 released supplementary code 和 paper-stated HDMI hyperparameters 的 HDMI-compatible local protocol，但当前 released materials 不足以支撑 exact reproduction of the author pipeline。

同时必须明确：

- 不能把当前实验写成“exactly reproduced HDMI”；
- 也不能仅凭当前复现差距就下结论说 HDMI 存在数据造假。

现阶段更稳妥的判断是：

- released materials 的复现实用性不足；
- 论文文字、补充代码和可执行协议之间存在实质性不一致。

## Reproduction Appendix

### A. 关键任务复现实验设计

为了检查结果偏差是否来自我们自定义脚本，而不是 HDMI released code 本身，我们使用 `probing.py` 的官方路径，在相同模型、相同层、相同 HDMI 超参下，对关键任务进行了附录复现实验。

固定设置：

- model: `EleutherAI/pythia-70m`
- layer: final layer (`layer_idx=-1`)
- HDMI hyperparameters: `hdmi_alpha=1`, `hdmi_inner_steps=30`
- objective: margin objective
- dataset: CausalGym local JSON
- seed: `1337`

变量：

- `probe_epochs ∈ {100, 150}`

其中 `100` 更接近论文 Appendix 的文字描述，`150` 更接近补充代码默认值。

### B. 三个关键任务的 probe epoch 敏感性

结果见 `D:\神经网络可解释性\hdmi_repro_appendix_probe_epochs.json`。

| Task | Probe Epochs | Interventional Zc Probe Acc | vZc Acc | vZe Acc | Completeness | Selectivity | Reliability |
|---|---:|---:|---:|---:|---:|---:|---:|
| `agr_sv_num_pp` | 100 | 0.4500 | 1.0000 | 1.0000 | 0.9997 | 0.9809 | 0.9902 |
| `agr_sv_num_pp` | 150 | 0.4500 | 1.0000 | 1.0000 | 1.0000 | 0.9765 | 0.9881 |
| `gss_subord_pp` | 100 | 0.9750 | 0.7250 | 1.0000 | 0.5609 | 0.9415 | 0.7030 |
| `gss_subord_pp` | 150 | 0.9313 | 0.9500 | 1.0000 | 0.8998 | 0.9670 | 0.9322 |
| `filler_gap_pp` | 100 | 0.7438 | 0.6500 | 1.0000 | 0.5003 | 0.9170 | 0.6474 |
| `filler_gap_pp` | 150 | 0.9750 | 0.4000 | 1.0000 | 0.7149 | 0.8807 | 0.7892 |

### C. 复现实验附录的核心发现

#### C.1 `agr_sv_num_pp` 稳定

`agr_sv_num_pp` 在 `100` 和 `150` epoch 下结果相近，说明这一任务的协议稳定性较好。本地复现与论文数字也最接近。

#### C.2 `gss_subord_pp` 极度敏感

`gss_subord_pp` 的 reliability 从 `0.7030` 跳到 `0.9322`，差值高达 `+0.2292`。这说明：

- 关键结果会被 probe 训练细节大幅改变；
- 当前 released materials 无法保证该任务的单次结果可稳定代表作者原始结果。

#### C.3 `filler_gap_pp` 极度敏感且 probe 结构异常

`filler_gap_pp` 在 `100` epoch 时：

- `vZc acc = 0.65`
- `R = 0.6474`

在 `150` epoch 时：

- `vZc acc = 0.40`
- `R = 0.7892`

也就是说，`vZc` 更差时，HDMI 的 reliability 反而更高。这是一个危险信号，说明：

- 当前 completeness 指标很可能被 probe calibration 污染；
- 该任务不能被当作干净、稳定的主证据。

### D. 本地结果与原文表格的偏差

能直接从原文 excerpt 对齐到的 pythia-70m 任务中，只有少数任务接近复现。

按 `|ΔR| <= 0.02` 视作近似可接受的标准：

- 接近复现的任务：3 个
- 明显超出误差范围的任务：7 个

典型例子：

| Task | Reported R | Local R | Delta |
|---|---:|---:|---:|
| `agr_sv_num_pp` | 0.9688 | 0.9881 | +0.0193 |
| `gss_subord_pp` | 0.8481 | 0.9322 | +0.0841 |
| `filler_gap_pp` | 0.6793 | 0.7892 | +0.1099 |
| `filler_gap_hierarchy` | 1.0000 | 0.7078 | -0.2922 |
| `gss_subord_subj-relc` | 1.0000 | 0.8120 | -0.1880 |

因此，当前最严谨的结论不是“我们复现失败了，因此作者有问题”，而是：

> released materials do not provide enough information for high-fidelity reproduction across key tasks.

## Case Studies

### A. 全量任务总体结构

在 29 个 CausalGym 子任务上，RACE 相比 HDMI 的结果为：

- 优于 HDMI：13 个任务
- 持平：4 个任务
- 差于 HDMI：12 个任务

均值上：

- HDMI mean reliability: `0.8176`
- RACE mean reliability: `0.8354`

但提升结构并不均衡：

- `selectivity` 提升：13 个任务
- `completeness` 提升：6 个任务
- `completeness` 与 `selectivity` 同时提升：仅 2 个任务

这说明当前方法的主要收益来自：

- 更少破坏 non-target factors；
- 而不是系统性地提高 target property flipping 的强度。

### B. 为什么很多增益来自 Selectivity，而不是 Completeness

我们认为原因主要有三点。

#### B.1 HDMI 在若干任务上 completeness 已接近饱和

例如：

- `agr_refl_num_pp`: HDMI completeness `0.9954`
- `filler_gap_hierarchy`: HDMI completeness 约 `1.0000`

在这类任务上，新方法几乎没有 completeness 上升空间。任何 reliability 提升主要只能来自提高 selectivity。

#### B.2 当前 RACE 是“受约束的 HDMI”，本质更保守

在大多数收益任务上，最优配置的 `mix=1.0`，说明：

- 更新几乎完全被限制在 learned subspace 内；
- 这会减少非目标方向漂移；
- 因而自然更容易提高 selectivity。

#### B.3 Reliability 是调和平均

当 completeness 已高时，小幅 selectivity 上升足以显著抬高 reliability。因此不能将当前 RACE 的 reliability 优势直接解释为“更强的因果翻转能力”。

### C. 大幅提升任务的 case study

本节的目标是检查这些大幅提升是否只是评测偶然。

#### C.1 `agr_refl_num_pp`

关键数字：

- HDMI: `C=0.9954`, `S=0.7998`, `R=0.8869`
- RACE: `C=0.9947`, `S=0.9644`, `R=0.9793`
- `ΔC=-0.0007`, `ΔS=+0.1647`, `ΔR=+0.0924`

probe 与诊断：

- validation `vZc acc = 0.975`
- test factual `vZc acc = 0.87`
- test actual-counterfactual `vZc acc = 0.87`
- gradient retention `0.9415`

判断：

- 这是一个相对可信的正例；
- probe 并未失效；
- completeness 几乎不变，selectivity 大幅提升；
- 说明 RACE 在这一任务上确实像是在减少 collateral shift，而不是仅仅依赖评测噪声。

#### C.2 `filler_gap_hierarchy`

关键数字：

- HDMI: `C=1.0000`, `S=0.5478`, `R=0.7078`
- RACE: `C=0.9986`, `S=0.8786`, `R=0.9348`
- `ΔC=-0.0014`, `ΔS=+0.3309`, `ΔR=+0.2269`

probe 与诊断：

- validation `vZc acc = 0.95`
- validation `vZe acc = 0.96`
- test factual/cf `vZc acc = 0.95 / 0.95`
- actual cf completeness ceiling mean `0.9269`
- gradient retention `0.8284`

判断：

- 这不是明显的评测偶然；
- completeness 已近饱和，RACE 的收益主要来自大幅降低非目标扰动；
- 这是目前最强的“selectivity-preserving intervention”证据之一。

#### C.3 `cleft_mod`

关键数字：

- HDMI: `C=0.5000`, `S=0.4765`, `R=0.4879`
- RACE: `C=0.5000`, `S=0.8781`, `R=0.6372`

probe 与诊断：

- validation `vZc acc = 0.65`
- test factual/cf `vZc acc = 0.5 / 0.5`
- actual cf completeness ceiling mean `0.5`

判断：

- 这里的 `vZc` 基本失效；
- completeness 不具备可解释性；
- reliability 增长主要反映 selectivity 部分的变化；
- 这个任务不能作为 strongest evidence，只能作为弱证据或附录材料。

#### C.4 `gss_subord_obj-relc`

关键数字：

- HDMI: `C=0.9560`, `S=0.6433`, `R=0.7691`
- RACE: `C=0.9152`, `S=0.9093`, `R=0.9122`

probe 与诊断：

- validation `vZc acc = 0.8`
- test factual/cf `vZc acc = 0.85 / 0.85`
- actual cf completeness ceiling mean `0.7978`
- gradient retention `0.8099`

判断：

- 这是中等可信的正例；
- probe 尚可，但并不完美；
- RACE 以约 `0.04` 的 completeness 代价换来约 `0.266` 的 selectivity 增长；
- 适合支持“better constrained intervention”，不适合支持“stronger completeness optimizer”。

### D. 失败任务的 case study

#### D.1 `gss_subord_pp`

关键数字：

- HDMI: `C=0.8998`, `S=0.9670`, `R=0.9322`
- RACE: `C=0.8395`, `S=0.9780`, `R=0.9035`

probe 与诊断：

- validation `vZc acc = 0.95`
- test factual/cf `vZc acc = 0.92 / 0.92`
- actual cf completeness ceiling mean `0.8859`
- gradient retention `0.8158`

判断：

- 这是一个较“真”的方法失败；
- probe 质量足够好，不能轻易把失败归因于评测噪声；
- RACE 仍然显著伤害 completeness。

机制解释：

1. continuation 是标点 `,` 与 `.`，属于极强的 readout-specific 目标；
2. 即便引入 gradient alignment，学到的仍然主要是 punctuation readout direction；
3. 该任务真正需要的可能是句法重分析相关的更细粒度因果方向，而不仅是 next-token margin direction；
4. 低秩投影会削弱这类对 completeness 至关重要的细小分量。

因此，`gss_subord_pp` 的失败说明：

> readout-aligned subspace 仍然不等于 causal reanalysis subspace。

#### D.2 `garden_mvrr_mod`

关键数字：

- HDMI: `C=0.0489`, `S=0.9351`, `R=0.0930`
- RACE: `C=0.0147`, `S=0.9517`, `R=0.0289`

probe 与诊断：

- validation `vZc acc = 0.825`
- test factual/cf `vZc acc = 0.9 / 0.9`
- actual cf completeness ceiling mean `0.8905`
- gradient retention `0.6564`

判断：

- 这里 probe 不坏，但 HDMI 自身 completeness 已几乎失效；
- RACE 进一步削弱了完成 target shift 所需的方向；
- 说明 final-layer next-token margin 本身对该任务就不是一个足够强的结构因果代理。

#### D.3 `garden_npz_v-trans_mod`

关键数字：

- HDMI: `C=0.6263`, `S=0.9947`, `R=0.7687`
- RACE: `C=0.5811`, `S=0.9938`, `R=0.7334`

probe 与诊断：

- validation `vZc acc = 0.55`
- test factual/cf `vZc acc = 0.5 / 0.5`
- actual cf completeness ceiling mean `0.5000`

判断：

- 该任务的 probe 已显著退化；
- 当前负结果是真失败还是 probe artifact，不能完全分离；
- 因此这一任务只能作为“风险样本”，不能作为强结论。

#### D.4 `garden_npz_obj_mod`

关键数字：

- HDMI: `C=1.0000`, `S=0.9521`, `R=0.9755`
- RACE: `C=0.9994`, `S=0.9165`, `R=0.9562`

probe 与诊断：

- validation `vZc acc = 0.95`
- gradient retention `0.8805`

判断：

- 该任务上 HDMI 已近乎天花板；
- RACE 反而同时伤害了 selectivity 与 completeness；
- 说明 learned subspace 并不总是比 full gradient 更稳健。

### E. case study 总结

当前证据支持如下更克制、但更稳健的结论：

1. RACE 的主要收益是减少非目标漂移，而不是系统性提高 counterfactual flipping 的完成度；
2. 在 agreement 和部分 hierarchy 任务上，这种“更保守”的更新是有益的；
3. 在 punctuation-sensitive 和 garden-path-sensitive 任务上，readout alignment 仍然不够，甚至会伤 completeness；
4. 因此，当前方法应被解释为“更好的约束式干预”，而不是“普适更强的干预器”。

## Threats to Validity

### A. Protocol Validity

当前最大的 external validity threat 是：

- 论文文字、补充代码和当前实现并不完全一致。

具体表现为：

- HDMI hyperparameters 的 paper/code mismatch；
- probe epochs 的 paper/code mismatch；
- CausalGym decorrelation 在论文中被强调，但 code 中未落实；
- 缺乏作者 run script 与 task-specific tuning 记录。

因此：

- 任何“超过 HDMI 原文”的表述都不安全；
- 任何“严格复现 HDMI”的表述都不成立。

### B. Probe Validity

多个任务表明，validation probe 并不总是稳定。

特别是：

- `filler_gap_pp`
- `cleft_mod`
- `garden_npz_v-trans_mod`

在这些任务上，`vZc` 已接近随机或 calibration 异常。此时：

- completeness 不再是一个可靠的结构因果指标；
- reliability 可能被 selectivity 主导；
- 不能把任务上的单点改善当作强证据。

### C. Metric Interpretation Validity

当前方法在很多任务上的收益主要来自 selectivity，而非 completeness。这意味着：

- “reliability 提升”不应被直接解释为“更好地完成 target property intervention”；
- 更合理的解释是：“在不明显损害目标翻转的前提下，更少改变非目标因素”。

如果主文不明确这一点，读者很容易误解方法能力边界。

### D. Structural Task Validity

`gss_subord_pp`、`garden_mvrr_mod` 等任务说明：

- final-layer next-token margin 不一定能代表结构性因果重分析；
- 即便引入 readout alignment，learned subspace 仍可能偏向局部 token decision，而非全局结构迁移。

因此：

- 当前方法的理论对象仍需进一步升级；
- 更强版本可能需要 nuisance suppression、更早层信息或 multi-step structural objective。

### E. On the Possibility of Data Fabrication

我们不放弃对异常结果保持怀疑，但基于当前证据，不应下“数据造假成立”的结论。

当前能严谨支持的说法只有：

1. released materials 不足以支持高保真复现；
2. 论文描述与补充代码存在实质性不一致；
3. 若干关键任务对训练细节高度敏感；
4. 因而原文结果的复现透明性存在问题。

当前不能严谨支持的说法包括：

- 作者伪造了结果；
- 原文表格必然不真实；
- 当前本地偏差足以证明作者故意操纵指标。

更稳妥的学术表述是：

> We identify substantial reproducibility gaps and protocol ambiguities in the released HDMI materials. These gaps warrant caution when interpreting exact numerical comparisons, but they do not by themselves constitute evidence of fabrication.

## Bottom Line

本附录支持以下最终结论：

1. 当前实验设置在“同一本地协议下比较 HDMI、Hybrid、RACE”这一点上是公平且有研究价值的；
2. 但当前 released materials 不能让我们声称“已严格复现 HDMI 原文”；
3. RACE 在全量任务均值上超过 HDMI，但其主要优势来自 selectivity，而非普遍更强的 completeness；
4. `gss_subord_pp` 和 `garden_*_mod` 的失败说明：仅靠 readout-aligned subspace 仍不足以建模结构性因果重分析；
5. 如果要把这条线推进到 CCF A oral 级，下一步必须补：
   - exact/compatible/unknown 协议审计表；
   - 多 seed 和显著性分析；
   - probe failure audit；
   - RACE v2，其中显式加入 nuisance suppression 或更强的 structural objective。
