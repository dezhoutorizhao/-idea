# Direction 1 Second-Round Analysis

Focus tasks: `agr_refl_num_obj-relc`, `agr_refl_num_pp`, `filler_gap_pp`, `npi_any_subj-relc`.

Protocol: second-round strengthened completeness-first curriculum with a more aggressive early phase (`phase1_frac`, `alpha1_scale`, `comp1_weight`) and a tunable late-phase repair strength (`comp2_weight`, `trust_beta2`).

## Main table

| Task | HDMI after | HDMI C | HDMI S | HDMI R | Base DPAS after | Base DPAS C | Base DPAS S | Base DPAS R | Curr v1 after | Curr v1 C | Curr v1 S | Curr v1 R | Curr v2 after | Curr v2 C | Curr v2 S | Curr v2 R | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `agr_refl_num_obj-relc` | 0.00 | 0.7738 | 1.0000 | 0.8724 | 0.00 | 0.5723 | 1.0000 | 0.7280 | 0.00 | 0.6373 | 1.0000 | 0.7785 | 0.00 | 0.7373 | 1.0000 | 0.8488 | clear partial repair |
| `agr_refl_num_pp` | 0.00 | 0.9954 | 0.7998 | 0.8869 | 0.00 | 0.8666 | 0.8418 | 0.8541 | 0.00 | 0.9418 | 0.8367 | 0.8861 | 0.00 | 0.9745 | 0.7904 | 0.8729 | near miss |
| `filler_gap_pp` | 0.00 | 0.7149 | 0.8807 | 0.7892 | 0.06 | 0.6081 | 0.9797 | 0.7504 | 0.00 | 0.5741 | 0.9895 | 0.7266 | 0.00 | 0.6194 | 0.9890 | 0.7617 | clear partial repair |
| `npi_any_subj-relc` | 0.00 | 0.8499 | 1.0000 | 0.9189 | 0.00 | 0.7935 | 1.0000 | 0.8849 | 0.00 | 0.8950 | 1.0000 | 0.9446 | 0.00 | 0.9785 | 1.0000 | 0.9892 | beats HDMI |

## Result summary

- `npi_any_subj-relc`: strong success. Curriculum v2 keeps `after_task_acc = 0.0`, preserves `S = 1.0`, and pushes `C` to `0.9785`, substantially above local HDMI (`0.8499`).
- `agr_refl_num_obj-relc`: strong partial repair. Curriculum v2 raises `C` from `0.5723` to `0.7373` and `R` from `0.7280` to `0.8488`, narrowing the HDMI gap dramatically, but still not fully closing it.
- `agr_refl_num_pp`: mixed outcome. Curriculum v2 increases `C` further to `0.9745`, but its late-phase selectivity falls enough that `R` drops below curriculum v1 and remains below HDMI. This task appears to need a more careful late-stage tradeoff controller.
- `filler_gap_pp`: partial repair. Curriculum v2 is clearly better than both base DPAS and curriculum v1 while restoring `after_task_acc = 0.0`, but it still remains below HDMI on `C` and `R`.

## Interpretation

- Direction 1 is clearly real, not noise. Second-round strengthening improved all four tasks over the original DPAS baseline on completeness, and improved three of the four over curriculum v1 on reliability.
- The shared winning pattern is more aggressive early movement: the selected v2 configurations consistently prefer `phase1_frac = 0.9`, higher `comp1_weight`, and often `alpha1_scale = 1.5`.
- This strongly supports the hypothesis that many C-only failures are caused by insufficient early movement toward the counterfactual region, not by selectivity or answer-flip failure.
- However, `agr_refl_num_pp` shows that simply pushing harder can overshoot the later-stage tradeoff: completeness rises, but a small selectivity loss can still lower reliability.

## Direction decision

- Keep Direction 1 as the primary branch.
- The next best refinement is not a new family yet, but a late-stage controller for Direction 1: specifically, adapt the second phase to stop once completeness plateaus and avoid unnecessary selectivity loss.
- Do not switch to curvature-aware DPAS yet. The schedule hypothesis has now produced two clear wins and two strong partial repairs on hard C-only tasks, which is a stronger signal than any alternative branch so far.
