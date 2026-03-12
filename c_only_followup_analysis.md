# C-only Follow-up Experiments

Focus tasks: `agr_refl_num_obj-relc`, `agr_refl_num_pp`, `filler_gap_pp`, `npi_any_subj-relc`.

Methods tested in priority order:
- Direction 1: completeness-first curriculum
- Direction 2: counterfactual-manifold anchor

## Comparative results

| Task | HDMI after | HDMI C | HDMI S | HDMI R | Base DPAS after | Base DPAS C | Base DPAS S | Base DPAS R | Curriculum after | Curriculum C | Curriculum S | Curriculum R | Manifold after | Manifold C | Manifold S | Manifold R | Best follow-up |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `agr_refl_num_obj-relc` | 0.00 | 0.7738 | 1.0000 | 0.8724 | 0.00 | 0.5723 | 1.0000 | 0.7280 | 0.00 | 0.6373 | 1.0000 | 0.7785 | 0.00 | 0.6132 | 1.0000 | 0.7602 | Curriculum |
| `agr_refl_num_pp` | 0.00 | 0.9954 | 0.7998 | 0.8869 | 0.00 | 0.8666 | 0.8418 | 0.8541 | 0.00 | 0.9418 | 0.8367 | 0.8861 | 0.00 | 0.9001 | 0.8576 | 0.8783 | Curriculum |
| `filler_gap_pp` | 0.00 | 0.7149 | 0.8807 | 0.7892 | 0.06 | 0.6081 | 0.9797 | 0.7504 | 0.00 | 0.5741 | 0.9895 | 0.7266 | 0.00 | 0.5829 | 0.9803 | 0.7311 | Manifold |
| `npi_any_subj-relc` | 0.00 | 0.8499 | 1.0000 | 0.9189 | 0.00 | 0.7935 | 1.0000 | 0.8849 | 0.00 | 0.8950 | 1.0000 | 0.9446 | 0.00 | 0.8130 | 1.0000 | 0.8968 | Curriculum |

## Key findings

- `npi_any_subj-relc`: curriculum clearly succeeds. It keeps `after_task_acc = 0.0`, preserves `S = 1.0`, and raises `C` from `0.7935` to `0.8950`, exceeding local HDMI (`0.8499`).
- `agr_refl_num_pp`: curriculum gives a strong partial repair. It raises `C` from `0.8666` to `0.9418` while keeping `after_task_acc = 0.0`, but still remains slightly below local HDMI `R` by about `0.0008`.
- `agr_refl_num_obj-relc`: curriculum improves substantially over base DPAS (`C +0.0650`, `R +0.0505`) but does not close the full gap to local HDMI.
- `filler_gap_pp`: both follow-up directions fail to improve over the base DPAS reliability-max result while preserving behavior. This remains an open failure task.
- Manifold-anchor is consistently weaker than curriculum on all four tasks in this first implementation.

## Interpretation

- The curriculum result supports the hypothesis that a significant subset of C-only failures comes from over-conservative early regularization: letting the edit move more freely toward the counterfactual state before restoring regularization helps.
- The manifold-anchor result suggests the current bottleneck is not simply that the point anchor is too rigid. A naive local affine counterfactual manifold does not solve the problem and may introduce additional approximation error.
- Therefore, among the tested ideas, direction 1 is currently the only one with clear positive value.

## Direction decision

- Keep direction 1 as the leading completeness-repair branch.
- Drop the current direction-2 implementation in its present form.
- Do not move to curvature-aware DPAS yet before extracting a stronger curriculum-based variant, because direction 1 already shows that optimization schedule matters and direction 2 does not yet justify the extra complexity.
