# Current Issues and Solution Roadmap

## 1. Current state

- The strongest current idea is no longer "another intervention optimizer", but the stronger finding that `probe-effective direction != behavior-effective direction`.
- The strongest current method is `DPAS + metric-aligned BCCI`.
- Under the current strict protocol, the method beats the local HDMI-compatible reproduction on `9/29` CausalGym subtasks.
- The main bottleneck is no longer idea generation, but evidence structure, protocol design, and failure accounting.

## 2. Main problems

### 2.1 Scientific problems

- `Subspace definition is still unstable`
  - We have evidence that counterfactual-shift directions and readout-aligned directions are not reliably behavior-effective.
  - DPAS improves this, but it is not yet demonstrated to be the principled final definition of the effective intervention region.

- `Completeness can be distorted on some tasks`
  - On several tasks, probe calibration is weak, so completeness values cannot always be treated as strong evidence.
  - This means some apparent gains may not reflect true causal control.

- `Reliability gains can come mostly from selectivity`
  - Some tasks improve mainly because selectivity rises while completeness stays flat or becomes less trustworthy.
  - Therefore `R` alone is insufficient as the primary scientific claim.

### 2.2 Protocol problems

- `The current strict protocol is too narrow as the only protocol`
  - A feasible configuration currently requires:
    - `after_task_acc == 0`
    - `completeness >= hdmi_tune.completeness`
    - `selectivity >= hdmi_tune.selectivity`
  - This is a strong behavior-aligned standard, but it suppresses many otherwise informative configurations.

- `Failure information is not saved densely enough`
  - For many tasks, we only know that no feasible configuration was found.
  - We do not yet have a complete task-by-task accounting of whether failure is caused by:
    - behavior flip failure
    - completeness deficit
    - selectivity deficit
    - multiple constraints failing together

- `Current comparisons are HDMI-compatible, not exact HDMI-author replication`
  - This is acceptable if stated rigorously, but it weakens any overly strong "SOTA" claim.

### 2.3 Benchmark and evidence problems

- `Too much evidence is concentrated on one benchmark family`
  - Current main evidence is still centered on CausalGym.
  - This is enough for diagnosis, but not enough for a broad oral-level generality claim.

- `Many tasks are ceiling or saturation tasks`
  - On a large subset of tasks, local HDMI is already near ceiling on completeness and/or selectivity.
  - These tasks are poor places to demand broad improvements and should not dominate the narrative.

### 2.4 Model generalization problems

- `Current strongest evidence is still mainly on pythia-70m`
  - This is enough for idea validation.
  - It is not enough for a strong cross-model claim.

### 2.5 Story and theory problems

- `The work can still be misread as an optimizer tweak`
  - If framed incorrectly, reviewers may see this as local objective engineering.
  - The stronger story is that current causal probing interventions confuse probe alignment with behavior alignment.

- `Theory is promising but not yet fully defensible`
  - We have strong theorem-style intuition and good counterexamples.
  - We do not yet have a sufficiently complete formal account of why the mismatch happens and when DPAS should repair it.

## 3. Most promising solution route

### P0. Immediate priority

- Add a second protocol: `Reliability-max + DeltaTaskAcc`
  - For each task, select the configuration with highest reliability.
  - Report `Completeness / Selectivity / Reliability / DeltaTaskAcc` for every task.
  - This aligns the work more directly with the protocol style of `How Reliable are Causal Probing Interventions?`

- Save dense failure diagnostics
  - For each task, save:
    - best-`R` configuration
    - best-`C` configuration
    - best-`S` configuration
    - lowest-`after_task_acc` configuration
  - This turns current `N/A` results into analyzable failures rather than opaque failures.

- Build a 29-task failure audit
  - Partition tasks into:
    - ceiling tasks
    - diagnostic tasks
    - probe-distorted tasks
    - open-failure tasks

### P1. Evidence strengthening

- Add `LGD`
  - This is the cheapest meaningful benchmark extension.
  - It helps show that the story is not just a CausalGym artifact.

- Add one cross-model validation stage
  - Minimum target: `pythia-160m`
  - Recommended scope:
    - 3-5 strongest diagnostic tasks
    - both strict protocol and reliability-max protocol

### P2. Higher-level strengthening

- Add another benchmark family if possible
  - Preferred next option: `IOI` or another non-CausalGym probing/intervention benchmark.

- Strengthen the formal framing
  - Make the paper define a behavior-probe agreement problem rather than an optimizer variation.
  - Add theorem-style claims that explain:
    - why readout-aligned directions can be behavior-ineffective
    - why reliability can improve mainly through selectivity
    - why behavior-constrained subspaces should help

## 4. Historical runtime anchors

- The current workspace does not contain complete per-run wall-clock logs, so the estimates below are range estimates, not exact timers.
- However, we do have useful empirical anchors from the existing result files:
  - `compare_hdmi_dpas_metric_bcci_failed_tasks.json`
    - written at `2026-03-11 20:43`
    - this corresponds to the 2-task DPAS diagnostic run
  - `compare_hdmi_dpas_metric_bcci_remaining27.json`
    - written at `2026-03-11 23:39`
    - this corresponds to the 27-task remaining DPAS sweep
- From this, a reasonable empirical estimate is:
  - full 27-task DPAS sweep on `pythia-70m`: about `3 hours`
  - full 29-task DPAS-scale sweep on `pythia-70m`: about `3.2-3.8 hours`
  - 2-task diagnostic sweep: about `0.5-1.0 hour`

## 5. Estimated time for the recommended route

### 5.1 Pure experiment runtime estimate

This section estimates runtime only, not coding or paper writing.

| Stage | Main work | Estimated runtime |
| --- | --- | ---: |
| P0-A | Re-run 29 CausalGym tasks with `Reliability-max` selection and dense candidate saving on `pythia-70m` | 4-6 h |
| P0-B | Optional focused re-run of 6-10 open-failure tasks with a slightly expanded search grid | 2-4 h |
| P1-A | LGD on `pythia-70m` under strict + reliability-max protocols | 1-3 h |
| P1-B | 3-5 diagnostic tasks on `pythia-160m` under both protocols | 2-5 h |
| P2-A | 1 additional benchmark family or deeper ablation sweep | 4-8 h |
| Total | Runtime only | 13-26 h |

### 5.2 End-to-end researcher time estimate

This estimate includes coding, debugging, reruns, result collation, and appendix-quality analysis, but not full paper drafting.

| Stage | Estimated total time |
| --- | ---: |
| P0 | 1.0-1.5 days |
| P1 | 1.0-1.5 days |
| P2 | 1.0-2.0 days |
| Total | 3.0-5.0 days |

## 6. Recommended execution order

1. Implement `Reliability-max + DeltaTaskAcc` and dense candidate logging.
2. Re-run all 29 CausalGym subtasks on `pythia-70m`.
3. Produce a failure audit table: `after / C / S / multiple`.
4. Add `LGD`.
5. Validate on `pythia-160m` for 3-5 diagnostic tasks.
6. Only then decide whether to invest in `IOI` or another benchmark family.

## 7. Final judgment

- The idea is already stronger than a routine incremental intervention paper.
- The current weakness is not novelty, but protocol breadth and evidence completeness.
- If P0 and P1 succeed, the work becomes much more defensible.
- If P0 succeeds but P1 fails, the work is still publishable as a strong diagnostic/mechanistic paper, but the oral-level claim remains risky.
