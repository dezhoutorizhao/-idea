# MCTR Failed-Task Experiment Log

实验日期：2026-03-11

方法：

- Base: HDMI-compatible local protocol
- Previous method: RACE-delta-gradient
- New direction: MCTR-RACE (Manifold-Constrained Trust Region)

关键脚本与结果：

- `D:\神经网络可解释性\compare_hdmi_mctr_failed_tasks.py`
- `D:\神经网络可解释性\compare_hdmi_mctr_failed_tasks.json`

测试任务：

- `gss_subord_pp`
- `garden_mvrr_mod`
- `garden_npz_v-trans_mod`
- `garden_npz_obj_mod`

## Main Results

### 1. `gss_subord_pp`

- HDMI: `C=0.8998`, `S=0.9670`, `R=0.9322`
- RACE: `C=0.8395`, `S=0.9780`, `R=0.9035`
- MCTR: `C=0.9882`, `S=0.9848`, `R=0.9865`

结论：

- 在不降低 selectivity 的前提下，显著提高了 completeness；
- 同时 reliability 也明显超过 HDMI；
- 这是当前最强的成功任务。

补充现象：

- test 上 `after_task_acc=0.0`，说明干预后 gold continuation 几乎完全被翻转；
- MCTR 到真实 counterfactual hidden 的平均距离也显著小于 HDMI/RACE。

### 2. `garden_mvrr_mod`

- HDMI: `C=0.0489`, `S=0.9351`, `R=0.0930`
- RACE: `C=0.0147`, `S=0.9517`, `R=0.0289`
- MCTR: `C=0.8946`, `S=0.9899`, `R=0.9398`

结论：

- 指标层面看是巨大提升；
- 但 `after_task_acc=0.48`，和 baseline `0.50` 几乎无差别；
- 这说明 completeness 可能再次被 probe 或 off-manifold 机制污染。

判定：

- 该结果目前不能视为可信成功；
- 需要后续用 task-level behavior 或更强评测来再次验证。

### 3. `garden_npz_v-trans_mod`

- HDMI: `C=0.6263`, `S=0.9947`, `R=0.7687`
- RACE: `C=0.5811`, `S=0.9938`, `R=0.7334`
- MCTR: 未找到满足“selectivity 不低于 HDMI 且 completeness 高于 HDMI”的配置

判定：

- 当前方向在该任务上失败；
- 该方向应在此任务上暂时废弃，转尝试下一方向。

### 4. `garden_npz_obj_mod`

- HDMI: `C=1.0000`, `S=0.9521`, `R=0.9755`
- RACE: `C=0.9994`, `S=0.9165`, `R=0.9562`
- MCTR: 未找到满足“selectivity 不低于 HDMI 且 completeness 高于 HDMI”的配置

判定：

- 该任务 HDMI 已接近天花板；
- 当前方向没有实质提升空间；
- 暂时不继续在该任务上投入。

## Current Decision

- 保留 MCTR 方向，因为其在 `gss_subord_pp` 上给出了可信成功信号；
- 不将 `garden_mvrr_mod` 视为已成功，需要额外验证；
- 对 `garden_npz_v-trans_mod` 和 `garden_npz_obj_mod`，当前方向视为失败或无收益；
- 下一阶段如果继续推进，应优先：
  - 对 `gss_subord_pp` 做多 seed 和更严格附录验证；
  - 对 `garden_mvrr_mod` 做 metric sanity check；
  - 对 remaining failures 尝试下一优先方向。
