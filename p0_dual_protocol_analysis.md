# P0 Dual-Protocol Analysis

## Files and checkpoints

- Result JSON: `compare_hdmi_dpas_metric_bcci_dual_protocol.json`
- Git checkpoint before P0 experiments: `bf93015`
- Git checkpoint with dual-protocol implementation: `7a0da58`
- Branch pushed to GitHub: `origin/codex/local`

## Core outcome

- Strict protocol still yields `9/29` wins vs the local HDMI-compatible reproduction: `cleft_mod`, `filler_gap_obj`, `garden_mvrr_mod`, `gss_subord_obj-relc`, `gss_subord_pp`, `gss_subord_subj-relc`, `npi_any_obj-relc`, `npi_ever_obj-relc`, `npi_ever_subj-relc`.
- Reliability-max protocol gives reportable results on all `29/29` tasks, eliminating the previous `N/A` table entries.
- Under reliability-max selection, `14/29` tasks have higher `Reliability` than local HDMI.
- Under reliability-max selection, only `8/29` tasks both improve `Reliability` and preserve `after_task_acc <= HDMI`.

## Main interpretation

- P0 confirms that the previous `N/A` values were not caused only by answer-flip failure.
- The dominant bottleneck is `Completeness`, not `after_task_acc`.
- Reliability-max is useful as a compatibility protocol, but it is not safe as the only protocol for a behavior-faithful paper story.

## Failure reason breakdown for the 20 strict-failure tasks

| Failure mode | Count | Tasks |
| --- | ---: | --- |
| `C` | 12 | `agr_gender`, `agr_refl_num_obj-relc`, `agr_refl_num_pp`, `agr_refl_num_subj-relc`, `agr_sv_num_obj-relc`, `agr_sv_num_pp`, `agr_sv_num_subj-relc`, `filler_gap_hierarchy`, `filler_gap_pp`, `filler_gap_subj`, `garden_npz_obj_mod`, `npi_any_subj-relc` |
| `S` | 3 | `cleft`, `garden_npz_obj`, `gss_subord` |
| `after` | 1 | `garden_mvrr` |
| `after+C` | 2 | `filler_gap_embed_3`, `filler_gap_embed_4` |
| `C+S` | 2 | `garden_npz_v-trans`, `garden_npz_v-trans_mod` |

Key readout:
- `C`-only failures dominate: `12` tasks.
- Pure `after` failure appears on only `1` task: `garden_mvrr`.
- Combined `after+C` failures are limited to `2` tasks: `filler_gap_embed_3`, `filler_gap_embed_4`.

## What reliability-max recovers beyond the strict protocol

- Reliability-max recovers `5` strict-failure tasks whose `Reliability` beats local HDMI.
- Strong salvage tasks (higher `R` with `after_task_acc <= HDMI`): `cleft`, `gss_subord`.
- Weak salvage tasks (higher `R` but worse `after_task_acc`): `filler_gap_hierarchy`, `filler_gap_subj`, `garden_mvrr`.

| Task | after_task_acc | dC | dS | dR | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `cleft` | 0.00 | 0.1152 | -0.0157 | 0.0529 | strong salvage |
| `gss_subord` | 0.00 | 0.0628 | 0.0476 | 0.0542 | strong salvage |
| `filler_gap_hierarchy` | 0.14 | -0.0025 | 0.4336 | 0.2815 | behavior-degrading salvage |
| `filler_gap_subj` | 0.07 | 0.0029 | 0.1002 | 0.0526 | behavior-degrading salvage |
| `garden_mvrr` | 0.44 | 0.7005 | 0.0034 | 0.5400 | behavior-degrading salvage |

## Why dual reporting is necessary

- There are tasks where reliability-max chooses a different configuration than the strict protocol even when a strict-feasible configuration exists.
- This happened on `3` tasks:

| Task | strict after | rmax after | strict R | rmax R | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `cleft_mod` | 0.00 | 0.09 | 0.6377 | 0.6388 | rmax prefers a different tradeoff |
| `filler_gap_obj` | 0.00 | 0.04 | 0.9786 | 0.9893 | rmax prefers a different tradeoff |
| `garden_mvrr_mod` | 0.00 | 0.34 | 0.9693 | 0.9666 | rmax prefers a different tradeoff |

Most important case:
- On `garden_mvrr_mod`, strict selection keeps `after_task_acc = 0.0` and achieves very high test reliability, while reliability-max tuning selects a behavior-weaker configuration (`after_task_acc = 0.34`).
- This is direct evidence that `Reliability` alone is not enough to identify the best behavior-faithful intervention.

## Bottom-line judgment

- P0 is successful.
- It solves the protocol-coverage problem: all 29 tasks now have reportable dual-protocol outputs.
- It sharpens the scientific conclusion: the main barrier is mostly `Completeness`, not answer-flip alone.
- It also justifies the paper design we discussed earlier: keep both a strict behavior-aligned protocol and a reliability-max compatibility protocol.
