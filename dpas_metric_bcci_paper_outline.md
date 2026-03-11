# DPAS + Metric-Aligned BCCI

Draft paper-style outline for an oral-level interpretability submission.

Status note:
- This document is intentionally written in a paper-facing style, but it distinguishes clearly between:
  - claims already supported by current local evidence;
  - claims that remain hypotheses and require broader validation.
- Current evidence comes from the local HDMI-compatible protocol and the result files:
  - `D:\神经网络可解释性\compare_hdmi_bcci_failed_tasks.json`
  - `D:\神经网络可解释性\compare_hdmi_bcci_metric_aligned_failed_tasks.json`
  - `D:\神经网络可解释性\compare_hdmi_dpas_metric_bcci_failed_tasks.json`
  - `D:\神经网络可解释性\compare_hdmi_mctr_failed_tasks.json`
  - `D:\神经网络可解释性\race_case_study_diag.json`

---

## 1. Working Title

Behavior-Probe Agreement Interventions:
Learning Behavior-Effective and Selective Hidden-State Subspaces for Counterfactual Editing

Alternative shorter title:

DPAS:
Behavior-Probe Agreement Subspaces for Reliable Counterfactual Hidden Interventions

---

## 2. Problem Setting

We study hidden-state interventions for counterfactual behavior editing in language models.

Given:
- a prompt hidden state `h`;
- a factual continuation `y_gold`;
- a counterfactual alternative `y_alt`;
- a target latent factor `Z_c`;
- an auxiliary non-target factor `Z_e`;

the goal is to edit the hidden state into `h'` such that:

1. the model behavior flips toward the counterfactual continuation;
2. the target factor is edited toward the counterfactual value;
3. non-target factors are preserved as much as possible.

In the current HDMI-style evaluation protocol, these desiderata are operationalized by:

- task behavior:
  lower `after_task_acc` is better under the binary gold-vs-alt setting;
- completeness:
  closeness of `vZc(h')` to the counterfactual target distribution;
- selectivity:
  invariance of `vZe(h')` relative to `vZe(h)`;
- reliability:
  harmonic mean of completeness and selectivity.

---

## 3. Core Observation

### 3.1 Empirical anomaly

Current local reproduction and extension experiments reveal a structural mismatch:

- some methods achieve very high completeness/selectivity/reliability, but fail to flip model behavior;
- some methods flip behavior reliably, but destroy completeness;
- therefore, "counterfactual-looking representation" is not equivalent to "behavior-effective intervention."

This anomaly is not a minor implementation issue. It indicates that current hidden intervention methods conflate at least three distinct geometric objects:

- the representation shift associated with factual-to-counterfactual pairs;
- the probe-defined target factor direction;
- the decoder-defined behavior decision boundary.

### 3.2 The underlying hypothesis

The central hypothesis of this work is:

> Effective counterfactual interventions should not be defined purely by representation similarity or probe alignment; they should be defined by agreement between behavior gradients, target-factor gradients, and non-target invariance constraints.

This leads to a new intervention principle:

> A good intervention direction is one that is simultaneously behavior-effective, target-complete, and non-target-selective.

---

## 4. Method Definition

## 4.1 Definitions

Let:

- `h in R^d` be the prompt hidden state at a chosen layer;
- `z_alt(h)` and `z_gold(h)` be decoder scores for the counterfactual and factual candidates;
- `vZc(h)` be a validation probe over the target factor `Z_c`;
- `vZe(h)` be a validation probe over the auxiliary factor `Z_e`.

Define:

### Definition 1: behavior margin

\[
m(h) = z_{alt}(h) - z_{gold}(h).
\]

An intervention is behavior-effective if it achieves `m(h') >= 0`, and margin-stable if it achieves `m(h') >= gamma` for some `gamma > 0`.

### Definition 2: target completeness

Let `e_cf` denote the one-hot counterfactual target distribution for `Z_c`. Then:

\[
\mathrm{Comp}(h') = 1 - \mathrm{TV}\big(vZ_c(h'), e_{cf}\big).
\]

### Definition 3: non-target selectivity

Let `p_e(h) = vZe(h)` and let `M(h)` be the HDMI normalization factor for the auxiliary probe. Then:

\[
\mathrm{Sel}(h, h') = 1 - \frac{\mathrm{TV}(p_e(h'), p_e(h))}{M(h)}.
\]

### Definition 4: behavior-probe agreement direction

Let:

\[
g_b = \nabla_h m(h),
\quad
g_c = \nabla_h \log p_{vZc}(Z_c = cf \mid h),
\quad
g_e = \nabla_h \log p_{vZe}(Z_e = factual \mid h).
\]

A direction is agreement-favoring if it aligns with both `g_b` and `g_c` while avoiding directions of large `g_e` energy.

---

## 4.2 DPAS: Decoder-Probe Agreement Subspace

We define a score matrix:

\[
A = \mathbb{E}[g_b g_c^\top + g_c g_b^\top] - \lambda_e \mathbb{E}[g_e g_e^\top].
\]

The DPAS basis `U in R^{d x r}` is the top-`r` eigenspace of `A`.

Interpretation:

- the first term rewards directions jointly useful for decoder behavior and target-factor editing;
- the second term suppresses directions that disproportionately perturb the auxiliary factor.

This differs from previous subspaces in an essential way:

- representation-shift subspaces privilege factual-vs-counterfactual movement;
- readout-aligned subspaces privilege the target readout;
- DPAS privileges agreement between behavior and target semantics under auxiliary invariance.

---

## 4.3 Metric-Aligned BCCI

Within the learned subspace `U`, we optimize a hidden edit `delta` using a behavior-constrained objective aligned with the evaluation metrics.

The intervention update is computed in the subspace:

\[
\Delta h = P_U g,
\quad
P_U = UU^\top,
\]

or more generally a mixture of full and projected gradients:

\[
\Delta h = (1-\alpha) g + \alpha P_U g.
\]

The optimization objective uses:

### Behavior hinge term

\[
L_{flip}(h') = \max(0, \gamma - m(h')).
\]

This is crucial: once the edit has crossed the decision boundary with reserve `gamma`, the objective stops rewarding further growth of the behavior margin.

### Metric-aligned completeness term

\[
L_{comp}(h') = \mathrm{TV}(vZ_c(h'), e_{cf}).
\]

### Metric-aligned selectivity term

\[
L_{sel}(h,h') = \frac{\mathrm{TV}(vZ_e(h'), vZ_e(h))}{M(h)}.
\]

### Anchor/manifold regularization

Let `a(h)` be a local counterfactual anchor estimated from nearby train counterfactuals. Then:

\[
L_{anchor}(h') = \|h' - a(h)\|_2^2.
\]

The full optimization objective is:

\[
L(h') = L_{flip}(h') + \lambda_c L_{comp}(h') + \lambda_s L_{sel}(h,h') + \lambda_a L_{anchor}(h').
\]

Equivalently, maximizing the negative of this loss yields the practical optimizer used in code.

---

## 4.4 Algorithm Sketch

1. Extract hidden states for train/dev/test splits.
2. Train the interventional and validation probes under the HDMI-compatible protocol.
3. Compute:
   - behavior gradients `g_b`;
   - target probe gradients `g_c`;
   - auxiliary gradients `g_e`.
4. Construct `A` and obtain the top-`r` DPAS basis.
5. For each test sample:
   - compute the hidden state `h`;
   - estimate a local counterfactual anchor;
   - optimize the hidden edit inside the DPAS subspace with the metric-aligned hinge objective;
   - evaluate behavior, completeness, selectivity, and reliability.

---

## 5. Theorem-Style Intuition

This section is not yet a formal theorem proof. It is a theorem-style intuition section that can later be upgraded into propositions/lemmas.

## 5.1 Intuition 1: why probe-only alignment is insufficient

### Informal Proposition 1

If the target probe gradient `g_c` and decoder behavior gradient `g_b` are poorly aligned, then maximizing probe completeness alone can increase `Comp(h')` without guaranteeing a decrease in `after_task_acc`.

### Intuition

Under a first-order approximation:

\[
m(h + \delta) \approx m(h) + g_b^\top \delta,
\]

while the target probe objective changes as:

\[
\log p_{vZc}(cf \mid h+\delta) \approx \log p_{vZc}(cf \mid h) + g_c^\top \delta.
\]

If `g_c^\top delta` is large but `g_b^\top delta` is small or negative, probe completeness can improve without behavioral flipping.

This explains high reliability but weak actual behavior change in probe-dominant interventions.

---

## 5.2 Intuition 2: why behavior-only crossing is insufficient

### Informal Proposition 2

If one only minimizes `after_task_acc` by crossing the decoder decision boundary with minimum action, the resulting edit can lie off the target semantic manifold, causing severe completeness distortion.

### Intuition

The decoder boundary can be locally close to the original hidden state, so a small perturbation may flip behavior:

\[
m(h+\delta) \ge 0.
\]

But this perturbation need not move in a direction consistent with the target-factor geometry:

\[
\mathrm{TV}(vZ_c(h+\delta), e_{cf})
\]

may remain large.

This explains why minimum-crossing methods can achieve low `after_task_acc` while collapsing completeness.

---

## 5.3 Intuition 3: why DPAS should help

### Informal Proposition 3

Under a local linearization, restricting interventions to a subspace that maximizes joint `g_b`-`g_c` alignment and minimizes `g_e` energy increases the chance of finding edits that simultaneously improve behavior effectiveness and probe reliability.

### Intuition

For a unit direction `u`, first-order gains are proportional to:

\[
g_b^\top u, \quad g_c^\top u, \quad g_e^\top u.
\]

A good direction should have:

- large positive `g_b^\top u`;
- large positive `g_c^\top u`;
- small `|g_e^\top u|`.

The DPAS score matrix approximates exactly this desideratum in expectation over samples.

Thus, unlike readout-only or representation-only subspaces, DPAS is explicitly constructed to privilege directions that are both behavior-effective and semantically consistent.

---

## 5.4 Intuition 4: why hinge behavior control matters

### Informal Proposition 4

Replacing a linear reward on behavior margin with a hinge objective reduces over-optimization of decoder margin and therefore decreases unnecessary damage to selectivity once the intervention is already behavior-effective.

### Intuition

A linear reward continues pushing edits deeper into the alternative region even after crossing, while the hinge loss saturates once a safe reserve `gamma` is achieved.

This allows the remaining optimization budget to focus on preserving completeness and selectivity.

---

## 6. Story Spine

This is the paper narrative that should be preserved across abstract, introduction, results, and discussion.

## 6.1 Story in one sentence

Current hidden interventions optimize "looking counterfactual" in probe space, but reliable interventions must optimize agreement between behavior, target semantics, and non-target invariance.

## 6.2 Story in six beats

### Beat 1: accepted assumption

The field implicitly assumes that if an intervention edits hidden states toward counterfactual representations, behavior should follow.

### Beat 2: failure discovery

We show this assumption is false:

- some methods achieve strong probe-based reliability without true answer flipping;
- some methods flip answers but collapse semantic completeness.

### Beat 3: diagnosis

The failure is geometric:

- the behavior decision boundary;
- the target-factor probe geometry;
- the auxiliary-factor invariance geometry

are not the same object.

### Beat 4: new definition

We redefine effective interventions as edits lying in a behavior-probe agreement space, rather than in a purely counterfactual or readout-aligned space.

### Beat 5: new method

We propose:

- DPAS to learn the agreement subspace;
- metric-aligned BCCI to optimize behavior and probe metrics within that subspace.

### Beat 6: consequence

This resolves the previously observed mismatch between behavior and reliability on difficult tasks, while preserving strong performance on tasks where prior methods already work well.

---

## 7. Claims Hierarchy

This section is critical for an oral-level paper because it keeps the paper defendable. The paper must separate what is strongly established from what is only suggestive.

## 7.1 Claim A: strongly defendable with current local evidence

### Claim A1

Probe-based reliability and actual behavior flipping can systematically diverge under hidden interventions.

Support:
- already directly observed in local experiments.

### Claim A2

Optimizing behavior alone or probe alignment alone is insufficient:
- behavior-only edits can destroy completeness;
- probe-dominant edits can fail to flip answers.

Support:
- MACC and prior BCCI/MCTR comparisons already establish this.

### Claim A3

For at least one key failure task, changing the subspace definition can repair the behavior-reliability mismatch.

Support:
- current `garden_mvrr_mod` result under `DPAS + metric-aligned BCCI`.

## 7.2 Claim B: plausible but needs stronger experimental support

### Claim B1

Behavior-probe agreement is a better organizing principle than readout alignment alone for hidden interventions.

Needed:
- more failure tasks;
- more model families;
- ablations against multiple subspace constructions.

### Claim B2

DPAS improves the Pareto frontier between behavior effectiveness and reliability metrics.

Needed:
- frontier plots across tasks/models;
- stronger hyperparameter stability evidence.

## 7.3 Claim C: only make if full evidence is obtained

### Claim C1

DPAS + metric-aligned BCCI is a general intervention framework for reliable counterfactual editing across model scales and architectures.

Needed:
- cross-model validation;
- broad task coverage;
- seed robustness;
- negative case analysis.

### Claim C2

The method establishes a new general theory of causal hidden interventions.

This claim should not be made unless formal analysis becomes substantially stronger.

---

## 8. Oral-Level Experimental Checklist

This checklist is intentionally strict. The method should not be framed as oral-level unless most items below are satisfied.

## 8.1 Protocol fidelity

- [ ] Re-state the full HDMI-compatible protocol precisely.
- [ ] Distinguish exact settings from compatible settings.
- [ ] Report all hyperparameters used in DPAS and metric-aligned BCCI.
- [ ] Use fixed seeds and report variance across at least 3 seeds on key tasks.
- [ ] Release code and result tables.

## 8.2 Core comparison baselines

- [ ] HDMI
- [ ] RACE
- [ ] MCTR
- [ ] MACC
- [ ] raw BCCI
- [ ] metric-aligned BCCI without DPAS
- [ ] DPAS + metric-aligned BCCI

## 8.3 Metrics to report together

Never report reliability alone.

- [ ] `baseline_task_acc`
- [ ] `after_task_acc`
- [ ] `delta_task_acc`
- [ ] `completeness`
- [ ] `selectivity`
- [ ] `reliability`
- [ ] `margin_reserve_rate`
- [ ] actual-counterfactual probe ceiling
- [ ] factual/counterfactual probe accuracies

## 8.4 Task coverage

- [ ] Main failure tasks where prior methods show behavior-reliability mismatch
- [ ] Strong tasks where prior methods already work well
- [ ] Probe-distorted tasks used only as stress tests, not as headline wins
- [ ] Full CausalGym sweep if computationally feasible

Recommended structure:
- primary evidence:
  failure tasks with clear behavior-probe mismatch;
- secondary evidence:
  representative stable tasks;
- stress tests:
  known probe-distorted tasks.

## 8.5 Cross-model validation

At minimum:

- [ ] `EleutherAI/pythia-70m`
- [ ] one larger Pythia model or GPT-2 family model
- [ ] one additional architecture family if feasible

The oral-level version should show the phenomenon is not model-family-specific.

## 8.6 Ablations

Mandatory ablations:

- [ ] remove DPAS, keep metric-aligned BCCI
- [ ] replace DPAS with readout-aligned basis
- [ ] replace DPAS with counterfactual-shift basis
- [ ] remove behavior hinge, use linear margin reward
- [ ] remove selectivity term
- [ ] remove anchor term
- [ ] vary subspace rank
- [ ] vary `lambda_e`
- [ ] vary margin reserve `gamma`

## 8.7 Diagnostics

- [ ] basis diagnostics: gradient norm retention and cosine
- [ ] margin reserve distribution after editing
- [ ] task-wise Pareto plots: `after_task_acc` vs `reliability`
- [ ] `completeness` vs actual-cf ceiling scatter
- [ ] probe validity diagnostics on strongest improvement tasks
- [ ] failure case studies on tasks where DPAS still does not help

## 8.8 Statistical robustness

- [ ] mean and standard deviation across seeds
- [ ] confidence intervals on task averages
- [ ] paired significance tests on key task improvements
- [ ] hyperparameter sensitivity on representative tasks

## 8.9 Validity threats

- [ ] protocol mismatch with author pipeline
- [ ] probe miscalibration
- [ ] off-manifold editing
- [ ] task-specific overfitting
- [ ] architecture dependence
- [ ] metric leakage through validation selection

## 8.10 Release readiness

- [ ] reproducible scripts for every main table
- [ ] one-command run instructions for key experiments
- [ ] clear commit hash table mapping code to results
- [ ] appendix with failed directions and negative results

---

## 9. Recommended Paper Structure

## 9.1 Main paper

1. Introduction
2. Failure diagnosis: why reliability and behavior diverge
3. Agreement-based intervention principle
4. DPAS
5. Metric-aligned BCCI
6. Experiments
7. Analysis and case studies
8. Limitations and broader implications

## 9.2 Appendix

1. Protocol audit
2. Reproduction details
3. Additional task tables
4. Probe validity diagnostics
5. Failure cases
6. Hyperparameter sensitivity
7. Pseudocode

---

## 10. Current Evidence Summary

Based on the current local evidence only, the most defensible summary is:

> The central innovation is not merely a stronger intervention optimizer, but a new definition of effective hidden intervention based on behavior-probe agreement. Preliminary results suggest that this definition resolves a key failure mode of prior methods on at least one hard task, while remaining competitive on tasks where prior methods are already strong.

What can already be said:

- DPAS appears to be the most promising direction discovered so far.
- The strongest current evidence supports the claim that subspace definition is a root cause of the behavior-reliability mismatch.

What cannot yet be said:

- that the method is universally better across models and tasks;
- that it fully replaces prior methods on all metrics;
- that it is already oral-level by evidence standard alone.

---

## 11. Oral-Level Bar

This project reaches a credible oral-level submission only if the following are all true:

1. the behavior-reliability mismatch is shown to be systematic, not anecdotal;
2. DPAS + metric-aligned BCCI fixes that mismatch on multiple hard tasks;
3. the same pattern appears across multiple model scales or families;
4. the paper provides strong failure analysis and probe-validity auditing;
5. the claims remain precise and conservative.

If these conditions are met, the work has a real oral-level profile because it offers:

- a new problem definition;
- a new geometric principle;
- a theory-flavored algorithmic framework;
- and a strong corrective story for the field.

