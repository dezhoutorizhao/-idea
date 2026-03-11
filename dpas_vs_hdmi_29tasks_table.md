# DPAS + metric-aligned BCCI vs local HDMI-compatible reproduction on 29 CausalGym subtasks

Notes:
- All HDMI numbers below are from the local HDMI-compatible reproduction, not from the HDMI paper tables.
- All Ours numbers below are from the current DPAS + metric-aligned BCCI runs.
- `No feasible config` means the current search grid did not find a config that satisfied the tuning constraints; it does not prove the method is impossible on that task.
- `N/A` does not mean the task was not run. It means the run completed, but no configuration passed the current tuning constraints, so no final test metric was written for the DPAS method in the saved JSON.
- `Strict win` means `after_task_acc <= HDMI after_task_acc` and `Reliability > HDMI Reliability`.

## Overall summary

| Item | Count |
| --- | ---: |
| Total subtasks | 29 |
| Feasible DPAS configs found | 9 |
| Strict wins vs HDMI | 9 |
| C/S/R all strictly higher than HDMI | 3 |
| C/S/R all non-lower than HDMI and at least one strictly higher | 8 |
| At least 2 of C/S/R strictly higher than HDMI | 9 |
| No feasible config under current search | 20 |

| Strict win tasks | Value |
| --- | --- |
| Tasks | `cleft_mod`, `filler_gap_obj`, `garden_mvrr_mod`, `gss_subord_obj-relc`, `gss_subord_pp`, `gss_subord_subj-relc`, `npi_any_obj-relc`, `npi_ever_obj-relc`, `npi_ever_subj-relc` |

## Full comparison table

| Task | HDMI after | HDMI C | HDMI S | HDMI R | Ours after | Ours C | Ours S | Ours R | Status | CSR tag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `agr_gender` | 0.0000 | 0.9734 | 1.0000 | 0.9865 | N/A | N/A | N/A | N/A | No feasible config | N/A |
| `agr_refl_num_obj-relc` | 0.0000 | 0.7738 | 1.0000 | 0.8724 | N/A | N/A | N/A | N/A | No feasible config | N/A |
| `agr_refl_num_pp` | 0.0000 | 0.9954 | 0.7998 | 0.8869 | N/A | N/A | N/A | N/A | No feasible config | N/A |
| `agr_refl_num_subj-relc` | 0.0000 | 1.0000 | 1.0000 | 1.0000 | N/A | N/A | N/A | N/A | No feasible config | N/A |
| `agr_sv_num_obj-relc` | 0.0000 | 0.9998 | 1.0000 | 0.9999 | N/A | N/A | N/A | N/A | No feasible config | N/A |
| `agr_sv_num_pp` | 0.0000 | 1.0000 | 0.9765 | 0.9881 | N/A | N/A | N/A | N/A | No feasible config | N/A |
| `agr_sv_num_subj-relc` | 0.0000 | 0.9994 | 1.0000 | 0.9997 | N/A | N/A | N/A | N/A | No feasible config | N/A |
| `cleft` | 0.0000 | 0.8848 | 0.9961 | 0.9372 | N/A | N/A | N/A | N/A | No feasible config | N/A |
| `cleft_mod` | 0.0000 | 0.5000 | 0.4765 | 0.4879 | 0.0000 | 0.5000 | 0.8803 | 0.6377 | Strict win | C/S/R non-lower |
| `filler_gap_embed_3` | 0.0000 | 0.9290 | 0.8979 | 0.9132 | N/A | N/A | N/A | N/A | No feasible config | N/A |
| `filler_gap_embed_4` | 0.0000 | 0.5174 | 0.9624 | 0.6730 | N/A | N/A | N/A | N/A | No feasible config | N/A |
| `filler_gap_hierarchy` | 0.0000 | 1.0000 | 0.5478 | 0.7078 | N/A | N/A | N/A | N/A | No feasible config | N/A |
| `filler_gap_obj` | 0.0000 | 0.5930 | 1.0000 | 0.7445 | 0.0000 | 0.9581 | 1.0000 | 0.9786 | Strict win | C/S/R non-lower |
| `filler_gap_pp` | 0.0000 | 0.7149 | 0.8807 | 0.7892 | N/A | N/A | N/A | N/A | No feasible config | N/A |
| `filler_gap_subj` | 0.0000 | 0.9521 | 0.8829 | 0.9162 | N/A | N/A | N/A | N/A | No feasible config | N/A |
| `garden_mvrr` | 0.0000 | 0.2868 | 0.9759 | 0.4433 | N/A | N/A | N/A | N/A | No feasible config | N/A |
| `garden_mvrr_mod` | 0.0000 | 0.0489 | 0.9351 | 0.0930 | 0.0000 | 0.9859 | 0.9532 | 0.9693 | Strict win | C/S/R all higher |
| `garden_npz_obj` | 0.0000 | 0.9956 | 0.9587 | 0.9768 | N/A | N/A | N/A | N/A | No feasible config | N/A |
| `garden_npz_obj_mod` | 0.0000 | 1.0000 | 0.9521 | 0.9755 | N/A | N/A | N/A | N/A | No feasible config | N/A |
| `garden_npz_v-trans` | 0.0000 | 0.9975 | 0.9734 | 0.9853 | N/A | N/A | N/A | N/A | No feasible config | N/A |
| `garden_npz_v-trans_mod` | 0.0000 | 0.6263 | 0.9947 | 0.7687 | N/A | N/A | N/A | N/A | No feasible config | N/A |
| `gss_subord` | 0.0000 | 0.8908 | 0.6833 | 0.7734 | N/A | N/A | N/A | N/A | No feasible config | N/A |
| `gss_subord_obj-relc` | 0.0000 | 0.9560 | 0.6433 | 0.7691 | 0.0000 | 0.9452 | 0.9739 | 0.9593 | Strict win | At least 2 CSR higher |
| `gss_subord_pp` | 0.0000 | 0.8998 | 0.9670 | 0.9322 | 0.0000 | 0.9661 | 0.9897 | 0.9778 | Strict win | C/S/R all higher |
| `gss_subord_subj-relc` | 0.0000 | 0.8664 | 0.7641 | 0.8120 | 0.0000 | 0.9002 | 0.9470 | 0.9230 | Strict win | C/S/R all higher |
| `npi_any_obj-relc` | 0.0000 | 0.6727 | 1.0000 | 0.8043 | 0.0000 | 0.7761 | 1.0000 | 0.8739 | Strict win | C/S/R non-lower |
| `npi_any_subj-relc` | 0.0000 | 0.8499 | 1.0000 | 0.9189 | N/A | N/A | N/A | N/A | No feasible config | N/A |
| `npi_ever_obj-relc` | 0.0000 | 0.6278 | 1.0000 | 0.7713 | 0.0000 | 0.9906 | 1.0000 | 0.9953 | Strict win | C/S/R non-lower |
| `npi_ever_subj-relc` | 0.0000 | 0.6452 | 1.0000 | 0.7844 | 0.0000 | 0.9676 | 1.0000 | 0.9835 | Strict win | C/S/R non-lower |
