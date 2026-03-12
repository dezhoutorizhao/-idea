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

DEFAULT_K_GRID = [8, 16]
DEFAULT_MANIFOLD_RANK_GRID = [1, 2, 4]
DEFAULT_ANCHOR_WEIGHT_GRID = [0.05, 0.1]
DEFAULT_TRUST_BETA_GRID = [0.0, 0.05]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


dpas = load_module(DPAS_PATH, "dpas_manifold_mod")
cmp = dpas.cmp
hdmi = dpas.hdmi
ma_bcci = dpas.ma_bcci
race = dpas.race


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


def find_local_cf_manifold(
    *,
    h_vec: torch.Tensor,
    sample,
    train_base: np.ndarray,
    train_cf: np.ndarray,
    train_samples,
    k: int,
    manifold_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    candidate_indices = [
        idx for idx, train_sample in enumerate(train_samples)
        if train_sample.zc == sample.zc and train_sample.cf_zc == sample.cf_zc
    ]
    if not candidate_indices:
        candidate_indices = list(range(len(train_samples)))

    candidate_base = torch.from_numpy(train_base[candidate_indices]).to(h_vec.device, dtype=h_vec.dtype)
    candidate_cf = torch.from_numpy(train_cf[candidate_indices]).to(h_vec.device, dtype=h_vec.dtype)
    dists = torch.norm(candidate_base - h_vec.unsqueeze(0), dim=-1)
    topk = min(int(k), dists.numel())
    top_vals, top_idx = torch.topk(dists, k=topk, largest=False)
    weights = 1.0 / top_vals.clamp_min(1e-6)
    weights = weights / weights.sum()
    local_cf = candidate_cf[top_idx]
    center = (local_cf * weights.unsqueeze(-1)).sum(dim=0)

    centered = local_cf - center.unsqueeze(0)
    if centered.shape[0] == 1:
        basis = torch.zeros(centered.shape[1], 1, device=h_vec.device, dtype=h_vec.dtype)
        return center, basis

    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    rank = max(1, min(int(manifold_rank), vh.shape[0]))
    basis = vh[:rank].T.contiguous()
    return center, basis


def affine_subspace_penalty(h_var: torch.Tensor, center: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    delta = h_var[0] - center
    if basis.numel() == 0:
        residual = delta
    else:
        proj = basis @ (basis.T @ delta)
        residual = delta - proj
    return torch.mean(residual ** 2)


def manifold_anchor_intervention(
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
    center: torch.Tensor,
    manifold_basis: torch.Tensor,
    trust_beta: float,
    gamma: float,
    sel_weight: float,
    anchor_weight: float,
) -> torch.Tensor:
    head = hdmi.lm_head_from_model(model)
    head_dtype = next(head.parameters()).dtype
    basis = basis.to(device=h.device, dtype=head_dtype)
    center = center.to(device=h.device, dtype=head_dtype)
    manifold_basis = manifold_basis.to(device=h.device, dtype=head_dtype)
    h_work = h.detach().to(dtype=head_dtype)

    probe_dim = int(vZc(h_work).shape[-1])
    goal = torch.zeros(probe_dim, device=h.device, dtype=head_dtype)
    goal[int(cf_zc)] = 1.0

    for _ in range(max(1, int(inner_steps))):
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

            anchor_penalty = affine_subspace_penalty(h_var, center, manifold_basis)
            flip_obj = -torch.relu(torch.tensor(float(gamma), device=h.device, dtype=head_dtype) - margin)
            objective = flip_obj - comp_loss - float(sel_weight) * sel_loss - float(anchor_weight) * anchor_penalty
            grad_h = torch.autograd.grad(objective, h_var, retain_graph=False, create_graph=False)[0]

        projected = grad_h @ basis @ basis.T
        update = (1.0 - mix) * grad_h + mix * projected
        with torch.no_grad():
            h_candidate = h_var + float(alpha) * update
            if trust_beta > 0.0:
                h_work = (1.0 - float(trust_beta)) * h_candidate + float(trust_beta) * center.unsqueeze(0)
            else:
                h_work = h_candidate

    return h_work.to(dtype=h.dtype)


def evaluate_manifold_anchor(
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
    manifold_rank: int,
    trust_beta: float,
    gamma: float,
    sel_weight: float,
    anchor_weight: float,
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
        center, manifold_basis = find_local_cf_manifold(
            h_vec=h0_vec,
            sample=sample,
            train_base=train_base,
            train_cf=train_cf,
            train_samples=train_samples,
            k=k_neighbors,
            manifold_rank=manifold_rank,
        )

        h1 = manifold_anchor_intervention(
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
            center=center,
            manifold_basis=manifold_basis,
            trust_beta=trust_beta,
            gamma=gamma,
            sel_weight=sel_weight,
            anchor_weight=anchor_weight,
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
    k_grid: Sequence[int],
    manifold_rank_grid: Sequence[int],
    anchor_weight_grid: Sequence[float],
    trust_beta_grid: Sequence[float],
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
    for k in k_grid:
        for manifold_rank in manifold_rank_grid:
            for anchor_weight in anchor_weight_grid:
                for trust_beta in trust_beta_grid:
                    metrics = evaluate_manifold_anchor(
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
                        k_neighbors=int(k),
                        manifold_rank=int(manifold_rank),
                        trust_beta=float(trust_beta),
                        gamma=float(base_cfg["gamma"]),
                        sel_weight=1.0,
                        anchor_weight=float(anchor_weight),
                    )
                    cfg = {
                        "rank": int(base_cfg["rank"]),
                        "ge_weight": float(base_cfg["ge_weight"]),
                        "gamma": float(base_cfg["gamma"]),
                        "k_neighbors": int(k),
                        "manifold_rank": int(manifold_rank),
                        "anchor_weight": float(anchor_weight),
                        "trust_beta": float(trust_beta),
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
    test_metrics = evaluate_manifold_anchor(
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
        k_neighbors=int(selected["config"]["k_neighbors"]),
        manifold_rank=int(selected["config"]["manifold_rank"]),
        trust_beta=float(selected["config"]["trust_beta"]),
        gamma=float(base_cfg["gamma"]),
        sel_weight=1.0,
        anchor_weight=float(selected["config"]["anchor_weight"]),
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
            "description": "maximize completeness subject first to after_task_acc <= HDMI_tune.after and selectivity >= HDMI_tune.selectivity; fallback to unconditional completeness-max manifold-anchor",
            "hdmi_tune": hdmi_tune,
            "feasible_found": best_targeted_feasible is not None,
        },
        "selected_manifold_anchor": selected,
        "manifold_anchor_test": test_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--model_name", default="EleutherAI/pythia-70m")
    parser.add_argument("--layer_idx", type=int, default=-1)
    parser.add_argument("--hdmi_alpha", type=float, default=1.0)
    parser.add_argument("--hdmi_inner_steps", type=int, default=30)
    parser.add_argument("--k_grid", nargs="+", type=int, default=DEFAULT_K_GRID)
    parser.add_argument("--manifold_rank_grid", nargs="+", type=int, default=DEFAULT_MANIFOLD_RANK_GRID)
    parser.add_argument("--anchor_weight_grid", nargs="+", type=float, default=DEFAULT_ANCHOR_WEIGHT_GRID)
    parser.add_argument("--trust_beta_grid", nargs="+", type=float, default=DEFAULT_TRUST_BETA_GRID)
    parser.add_argument("--output", default="compare_hdmi_dpas_manifold_anchor_c_only.json")
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
        "method": "DPAS-manifold-anchor-c-only",
        "tasks": [],
        "model_name": args.model_name,
        "layer_idx": args.layer_idx,
        "hdmi_alpha": args.hdmi_alpha,
        "hdmi_inner_steps": args.hdmi_inner_steps,
    }

    for task in args.tasks:
        print(f"Running manifold-anchor DPAS on {task}")
        task_result = run_task(
            model=model,
            tokenizer=tokenizer,
            task=task,
            device=device,
            layer_idx=args.layer_idx,
            hdmi_alpha=args.hdmi_alpha,
            hdmi_inner_steps=args.hdmi_inner_steps,
            k_grid=args.k_grid,
            manifold_rank_grid=args.manifold_rank_grid,
            anchor_weight_grid=args.anchor_weight_grid,
            trust_beta_grid=args.trust_beta_grid,
            dual_results_map=dual_results_map,
        )
        results["tasks"].append(task_result)
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(task_result["selected_manifold_anchor"], indent=2))

    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
