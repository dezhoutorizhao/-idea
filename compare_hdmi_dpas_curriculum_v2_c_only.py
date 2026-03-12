import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
DPAS_PATH = ROOT / "compare_hdmi_dpas_metric_bcci_failed_tasks.py"
DUAL_RESULTS_PATH = ROOT / "compare_hdmi_dpas_metric_bcci_dual_protocol.json"

TASKS = [
    "agr_refl_num_obj-relc",
    "agr_refl_num_pp",
    "filler_gap_pp",
    "npi_any_subj-relc",
]

DEFAULT_PHASE1_FRAC_GRID = [0.75, 0.9]
DEFAULT_ALPHA1_SCALE_GRID = [1.0, 1.5]
DEFAULT_ALPHA2_SCALE_GRID = [0.5, 1.0]
DEFAULT_COMP1_WEIGHT_GRID = [4.0, 6.0]
DEFAULT_COMP2_WEIGHT_GRID = [1.0, 2.0]
DEFAULT_TRUST2_GRID = [0.0, 0.05]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


dpas = load_module(DPAS_PATH, "dpas_curr_v2_mod")
cmp = dpas.cmp
hdmi = dpas.hdmi
ma_bcci = dpas.ma_bcci
race = dpas.race
mctr = dpas.mctr


def load_dual_results(path: Path) -> Dict[str, Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {task_result["task"]: task_result for task_result in data.get("tasks", [])}


def metrics_to_dict(metrics):
    if metrics is None:
        return None
    if isinstance(metrics, dict):
        result = dict(metrics)
    else:
        result = {
            "baseline_task_acc": float(metrics.baseline_task_acc),
            "after_task_acc": float(metrics.after_task_acc),
            "delta_task_acc": float(metrics.delta_task_acc),
            "completeness": float(metrics.completeness),
            "selectivity": float(metrics.selectivity),
            "reliability": float(metrics.reliability),
            "n": int(metrics.n),
        }
    if "baseline_task_acc" in result and "after_task_acc" in result:
        result["delta_task_acc_positive"] = float(result["baseline_task_acc"] - result["after_task_acc"])
    return result


def curriculum_v2_intervention(
    *,
    model,
    h: torch.Tensor,
    target_token_id: int,
    source_token_id: int,
    cf_zc: int,
    vZc,
    vZe,
    p_ze_before: Optional[torch.Tensor],
    basis: torch.Tensor,
    mix: float,
    alpha: float,
    inner_steps: int,
    anchor: torch.Tensor,
    gamma: float,
    phase1_frac: float,
    alpha1_scale: float,
    alpha2_scale: float,
    comp1_weight: float,
    comp2_weight: float,
    trust_beta2: float,
    sel1_weight: float = 0.0,
    sel2_weight: float = 1.0,
    trust_beta1: float = 0.0,
    anchor_weight1: float = 0.0,
    anchor_weight2: float = 0.05,
) -> torch.Tensor:
    head = hdmi.lm_head_from_model(model)
    head_dtype = next(head.parameters()).dtype
    basis = basis.to(device=h.device, dtype=head_dtype)
    anchor = anchor.to(device=h.device, dtype=head_dtype)
    h_work = h.detach().to(dtype=head_dtype)

    probe_dim = int(vZc(h_work).shape[-1])
    goal = torch.zeros(probe_dim, device=h.device, dtype=head_dtype)
    goal[int(cf_zc)] = 1.0

    total_steps = max(1, int(inner_steps))
    phase1_steps = int(round(total_steps * float(phase1_frac)))
    phase1_steps = min(max(1, phase1_steps), total_steps)

    for step in range(total_steps):
        in_phase1 = step < phase1_steps
        comp_weight = float(comp1_weight if in_phase1 else comp2_weight)
        sel_weight = float(sel1_weight if in_phase1 else sel2_weight)
        trust_beta = float(trust_beta1 if in_phase1 else trust_beta2)
        anchor_weight = float(anchor_weight1 if in_phase1 else anchor_weight2)
        step_alpha = float(alpha * (alpha1_scale if in_phase1 else alpha2_scale))

        with torch.enable_grad():
            h_var = h_work.clone().detach().requires_grad_(True)
            z = head(h_var)
            margin = z[0, int(target_token_id)] - z[0, int(source_token_id)]

            p_c_after = torch.softmax(vZc(h_var), dim=-1)[0]
            comp_loss = ma_bcci.tv_distance_torch(p_c_after, goal)

            sel_loss = torch.zeros((), device=h.device, dtype=head_dtype)
            if vZe is not None and p_ze_before is not None:
                p_ze_after = torch.softmax(vZe(h_var), dim=-1)[0]
                m = max(1.0 - float(torch.min(p_ze_before).item()), float(torch.max(p_ze_before).item()))
                if m > 0:
                    sel_loss = ma_bcci.tv_distance_torch(p_ze_after, p_ze_before) / m

            anchor_penalty = torch.mean((h_var[0] - anchor) ** 2)
            flip_obj = -torch.relu(torch.tensor(float(gamma), device=h.device, dtype=head_dtype) - margin)
            objective = flip_obj - comp_weight * comp_loss - sel_weight * sel_loss - anchor_weight * anchor_penalty
            grad_h = torch.autograd.grad(objective, h_var, retain_graph=False, create_graph=False)[0]

        projected = grad_h @ basis @ basis.T
        update = (1.0 - mix) * grad_h + mix * projected
        with torch.no_grad():
            h_candidate = h_var + step_alpha * update
            if trust_beta > 0.0:
                h_work = (1.0 - trust_beta) * h_candidate + trust_beta * anchor.unsqueeze(0)
            else:
                h_work = h_candidate

    return h_work.to(dtype=h.dtype)


def evaluate_curriculum_v2(
    *,
    model,
    tokenizer,
    samples,
    vZc,
    vZe,
    basis: torch.Tensor,
    mix: float,
    device: torch.device,
    layer_idx: int,
    alpha: float,
    inner_steps: int,
    train_base: np.ndarray,
    train_cf: np.ndarray,
    train_samples,
    k_neighbors: int,
    gamma: float,
    phase1_frac: float,
    alpha1_scale: float,
    alpha2_scale: float,
    comp1_weight: float,
    comp2_weight: float,
    trust_beta2: float,
):
    model.eval()
    vZc.eval()
    if vZe is not None:
        vZe.eval()

    correct_base = 0
    correct_after = 0
    comp_vals = []
    sel_vals = []
    reserve_count = 0

    for sample in samples:
        base_gold = hdmi.sequence_logprob(model, tokenizer, sample.prompt_text, sample.gold_cont, device)
        base_alt = hdmi.sequence_logprob(model, tokenizer, sample.prompt_text, sample.alt_cont, device)
        lp_gold0 = float(base_gold["total_lp"].item())
        lp_alt0 = float(base_alt["total_lp"].item())
        if hdmi.choose_candidate_by_logprob(lp_gold0, lp_alt0) == 0:
            correct_base += 1

        prefix_ids = hdmi.encode(tokenizer, sample.prompt_text, device)
        h0 = hdmi.get_prompt_hidden_at_layer(model, prefix_ids, layer_idx=layer_idx)
        h0_vec = h0[0].detach().float()

        p_ze_before_np = None
        p_ze_before_t = None
        if vZe is not None:
            p_ze_before_t = torch.softmax(vZe(h0), dim=-1)[0].detach()
            p_ze_before_np = p_ze_before_t.cpu().numpy()

        token_gold = int(base_gold["first_token_id"].item())
        token_alt = int(base_alt["first_token_id"].item())
        anchor = mctr.find_local_cf_anchor(
            h_vec=h0_vec,
            sample=sample,
            train_base=train_base,
            train_cf=train_cf,
            train_samples=train_samples,
            k=k_neighbors,
        )

        h1 = curriculum_v2_intervention(
            model=model,
            h=h0,
            target_token_id=token_alt,
            source_token_id=token_gold,
            cf_zc=int(sample.cf_zc),
            vZc=vZc,
            vZe=vZe,
            p_ze_before=p_ze_before_t,
            basis=basis,
            mix=mix,
            alpha=alpha,
            inner_steps=inner_steps,
            anchor=anchor,
            gamma=gamma,
            phase1_frac=phase1_frac,
            alpha1_scale=alpha1_scale,
            alpha2_scale=alpha2_scale,
            comp1_weight=comp1_weight,
            comp2_weight=comp2_weight,
            trust_beta2=trust_beta2,
        )

        gold_lp = hdmi.compute_first_step_logprob_from_h(model, h1, torch.tensor(token_gold, device=device)).item()
        alt_lp = hdmi.compute_first_step_logprob_from_h(model, h1, torch.tensor(token_alt, device=device)).item()
        lp_gold1 = lp_gold0 - float(base_gold["first_lp"].item()) + gold_lp
        lp_alt1 = lp_alt0 - float(base_alt["first_lp"].item()) + alt_lp

        if hdmi.choose_candidate_by_logprob(lp_gold1, lp_alt1) == 0:
            correct_after += 1
        if lp_alt1 - lp_gold1 >= gamma:
            reserve_count += 1

        p_c_after = torch.softmax(vZc(h1), dim=-1).detach().cpu().numpy()[0]
        goal = np.zeros_like(p_c_after)
        goal[int(sample.cf_zc)] = 1.0
        comp_vals.append(1.0 - hdmi.tv_distance(p_c_after, goal))

        if vZe is not None and p_ze_before_np is not None:
            p_ze_after = torch.softmax(vZe(h1), dim=-1).detach().cpu().numpy()[0]
            m = max(1.0 - float(np.min(p_ze_before_np)), float(np.max(p_ze_before_np)))
            sel_vals.append(1.0 - (hdmi.tv_distance(p_ze_after, p_ze_before_np) / m if m > 0 else 0.0))
        else:
            sel_vals.append(1.0)

    n = len(samples)
    baseline_acc = correct_base / n if n else 0.0
    after_acc = correct_after / n if n else 0.0
    completeness = float(np.mean(comp_vals)) if comp_vals else 0.0
    selectivity = float(np.mean(sel_vals)) if sel_vals else 1.0
    return {
        "baseline_task_acc": baseline_acc,
        "after_task_acc": after_acc,
        "delta_task_acc": after_acc - baseline_acc,
        "delta_task_acc_positive": baseline_acc - after_acc,
        "completeness": completeness,
        "selectivity": selectivity,
        "reliability": hdmi.harmonic_mean(completeness, selectivity),
        "n": n,
        "margin_reserve_rate": reserve_count / n if n else 0.0,
    }


def targeted_score(metrics: Dict[str, float]) -> Tuple[float, float, float, float]:
    return (
        float(metrics["completeness"]),
        float(metrics["reliability"]),
        float(metrics["selectivity"]),
        -float(metrics["after_task_acc"]),
    )


def run_task(
    *,
    model,
    tokenizer,
    task: str,
    device: torch.device,
    layer_idx: int,
    hdmi_alpha: float,
    hdmi_inner_steps: int,
    phase1_frac_grid: Sequence[float],
    alpha1_scale_grid: Sequence[float],
    alpha2_scale_grid: Sequence[float],
    comp1_weight_grid: Sequence[float],
    comp2_weight_grid: Sequence[float],
    trust_beta2_grid: Sequence[float],
    dual_results_map: Dict[str, Dict],
):
    dpas.set_seed(1337)
    base_result = dual_results_map[task]
    base_cfg = base_result["completeness_max_protocol"]["selected_config"]

    splits = hdmi._load_causalgym_local_json(str(cmp.LOCAL_JSON_DIR))
    train_records = [record for record in splits["train"] if record["task"] == task]
    dev_records = [record for record in splits["dev"] if record["task"] == task]
    test_records = [record for record in splits["test"] if record["task"] == task]
    train_samples, val_samples, test_samples = hdmi.make_causalgym_samples(
        {"train": train_records, "dev": dev_records, "test": test_records},
        tasks=[task],
    )

    cf_prompts_train = cmp.build_cf_prompts(train_records)
    X_cf_train = cmp.build_features_for_texts(hdmi, model, tokenizer, cf_prompts_train, device, layer_idx)
    probe_c, vZc, vZe, interventional_acc, vzc_acc, vze_acc, X_train, tr_i, te_i = cmp.train_probes_exact(
        hdmi, train_samples, val_samples, model, tokenizer, device, layer_idx
    )
    behavior_grads = race.build_margin_gradients(model, tokenizer, train_samples, device, layer_idx)
    zc_grads = dpas.build_zc_gradients(model, tokenizer, train_samples, device, layer_idx, vZc)
    ze_grads = dpas.build_ze_gradients(model, tokenizer, train_samples, device, layer_idx, vZe) if vZe is not None else np.zeros_like(behavior_grads[:1])

    basis = dpas.learn_dpas_basis(
        behavior_grads=behavior_grads,
        zc_grads=zc_grads,
        ze_grads=ze_grads,
        rank=int(base_cfg["rank"]),
        ge_weight=float(base_cfg["ge_weight"]),
    )

    tune_samples = [train_samples[i] for i in te_i]
    tune_train_samples = [train_samples[i] for i in tr_i]
    hdmi_tune = metrics_to_dict(
        hdmi.evaluate(
            model=model,
            tokenizer=tokenizer,
            samples=tune_samples,
            probe_c_interv=probe_c,
            vZc=vZc,
            vZe=vZe,
            intervention="hdmi",
            hdmi_alpha=hdmi_alpha,
            hdmi_inner_steps=hdmi_inner_steps,
            hdmi_use_margin=True,
            hdmi_normalize_grad=False,
            hdmi_grad_clip_norm=0.0,
            device=device,
            layer_idx=layer_idx,
            verbose=False,
        )
    )

    best_targeted = None
    best_targeted_feasible = None
    for phase1_frac in phase1_frac_grid:
        for alpha1_scale in alpha1_scale_grid:
            for alpha2_scale in alpha2_scale_grid:
                for comp1_weight in comp1_weight_grid:
                    for comp2_weight in comp2_weight_grid:
                        for trust_beta2 in trust_beta2_grid:
                            metrics = evaluate_curriculum_v2(
                                model=model,
                                tokenizer=tokenizer,
                                samples=tune_samples,
                                vZc=vZc,
                                vZe=vZe,
                                basis=basis,
                                mix=1.0,
                                device=device,
                                layer_idx=layer_idx,
                                alpha=hdmi_alpha,
                                inner_steps=hdmi_inner_steps,
                                train_base=X_train[tr_i],
                                train_cf=X_cf_train[tr_i],
                                train_samples=tune_train_samples,
                                k_neighbors=16,
                                gamma=float(base_cfg["gamma"]),
                                phase1_frac=float(phase1_frac),
                                alpha1_scale=float(alpha1_scale),
                                alpha2_scale=float(alpha2_scale),
                                comp1_weight=float(comp1_weight),
                                comp2_weight=float(comp2_weight),
                                trust_beta2=float(trust_beta2),
                            )
                            cfg = {
                                "rank": int(base_cfg["rank"]),
                                "ge_weight": float(base_cfg["ge_weight"]),
                                "gamma": float(base_cfg["gamma"]),
                                "phase1_frac": float(phase1_frac),
                                "alpha1_scale": float(alpha1_scale),
                                "alpha2_scale": float(alpha2_scale),
                                "comp1_weight": float(comp1_weight),
                                "comp2_weight": float(comp2_weight),
                                "trust_beta2": float(trust_beta2),
                            }
                            record = {"config": cfg, "tune_metrics": metrics}
                            if best_targeted is None or targeted_score(metrics) > targeted_score(best_targeted["tune_metrics"]):
                                best_targeted = record

                            feasible = (
                                metrics["after_task_acc"] <= hdmi_tune["after_task_acc"] + 1e-9
                                and metrics["selectivity"] + 1e-6 >= hdmi_tune["selectivity"]
                            )
                            if feasible:
                                if best_targeted_feasible is None or targeted_score(metrics) > targeted_score(best_targeted_feasible["tune_metrics"]):
                                    best_targeted_feasible = record

    selected = best_targeted_feasible if best_targeted_feasible is not None else best_targeted
    test_metrics = evaluate_curriculum_v2(
        model=model,
        tokenizer=tokenizer,
        samples=test_samples,
        vZc=vZc,
        vZe=vZe,
        basis=basis,
        mix=1.0,
        device=device,
        layer_idx=layer_idx,
        alpha=hdmi_alpha,
        inner_steps=hdmi_inner_steps,
        train_base=X_train,
        train_cf=X_cf_train,
        train_samples=train_samples,
        k_neighbors=16,
        gamma=float(base_cfg["gamma"]),
        phase1_frac=float(selected["config"]["phase1_frac"]),
        alpha1_scale=float(selected["config"]["alpha1_scale"]),
        alpha2_scale=float(selected["config"]["alpha2_scale"]),
        comp1_weight=float(selected["config"]["comp1_weight"]),
        comp2_weight=float(selected["config"]["comp2_weight"]),
        trust_beta2=float(selected["config"]["trust_beta2"]),
    )

    return {
        "task": task,
        "probe_metrics": {
            "interventional_zc_probe_acc": interventional_acc,
            "validation_zc_probe_acc": vzc_acc,
            "validation_ze_probe_acc": vze_acc,
        },
        "base_dual_protocol_result": {
            "hdmi_test": base_result["hdmi_test"],
            "reliability_max_test": base_result["reliability_max_protocol"]["test_metrics"],
            "completeness_max_test": base_result["completeness_max_protocol"]["test_metrics"],
            "completeness_max_config": base_result["completeness_max_protocol"]["selected_config"],
        },
        "selection_protocol": {
            "description": "v2 completeness-first curriculum: maximize completeness subject first to after_task_acc <= HDMI_tune.after and selectivity >= HDMI_tune.selectivity; fallback to unconditional completeness-max v2 curriculum",
            "hdmi_tune": hdmi_tune,
            "feasible_found": best_targeted_feasible is not None,
        },
        "selected_curriculum_v2": selected,
        "curriculum_v2_test": test_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--model_name", default="EleutherAI/pythia-70m")
    parser.add_argument("--layer_idx", type=int, default=-1)
    parser.add_argument("--hdmi_alpha", type=float, default=1.0)
    parser.add_argument("--hdmi_inner_steps", type=int, default=30)
    parser.add_argument("--phase1_frac_grid", nargs="+", type=float, default=DEFAULT_PHASE1_FRAC_GRID)
    parser.add_argument("--alpha1_scale_grid", nargs="+", type=float, default=DEFAULT_ALPHA1_SCALE_GRID)
    parser.add_argument("--alpha2_scale_grid", nargs="+", type=float, default=DEFAULT_ALPHA2_SCALE_GRID)
    parser.add_argument("--comp1_weight_grid", nargs="+", type=float, default=DEFAULT_COMP1_WEIGHT_GRID)
    parser.add_argument("--comp2_weight_grid", nargs="+", type=float, default=DEFAULT_COMP2_WEIGHT_GRID)
    parser.add_argument("--trust_beta2_grid", nargs="+", type=float, default=DEFAULT_TRUST2_GRID)
    parser.add_argument("--output", default="compare_hdmi_dpas_curriculum_v2_c_only.json")
    args = parser.parse_args()

    dual_results_map = load_dual_results(DUAL_RESULTS_PATH)
    dpas.set_seed(1337)
    cmp.ensure_local_json(args.tasks)

    device = hdmi.get_device("auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device).eval()
    model.config.output_hidden_states = True

    results = {
        "method": "DPAS-curriculum-completeness-first-v2",
        "tasks": [],
        "model_name": args.model_name,
        "layer_idx": args.layer_idx,
        "hdmi_alpha": args.hdmi_alpha,
        "hdmi_inner_steps": args.hdmi_inner_steps,
    }

    for task in args.tasks:
        print(f"Running curriculum-v2 DPAS on {task}")
        task_result = run_task(
            model=model,
            tokenizer=tokenizer,
            task=task,
            device=device,
            layer_idx=args.layer_idx,
            hdmi_alpha=args.hdmi_alpha,
            hdmi_inner_steps=args.hdmi_inner_steps,
            phase1_frac_grid=args.phase1_frac_grid,
            alpha1_scale_grid=args.alpha1_scale_grid,
            alpha2_scale_grid=args.alpha2_scale_grid,
            comp1_weight_grid=args.comp1_weight_grid,
            comp2_weight_grid=args.comp2_weight_grid,
            trust_beta2_grid=args.trust_beta2_grid,
            dual_results_map=dual_results_map,
        )
        results["tasks"].append(task_result)
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(task_result["selected_curriculum_v2"], indent=2))

    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
