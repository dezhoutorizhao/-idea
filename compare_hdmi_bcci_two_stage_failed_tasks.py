import argparse
import importlib.util
import json
import random
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
RACE_PATH = ROOT / "compare_hdmi_race_exact.py"
MCTR_PATH = ROOT / "compare_hdmi_mctr_failed_tasks.py"
RACE_RESULTS_PATH = ROOT / "compare_hdmi_race_all.json"
MCTR_RESULTS_PATH = ROOT / "compare_hdmi_mctr_failed_tasks.json"
BCCI_RESULTS_PATH = ROOT / "compare_hdmi_bcci_failed_tasks.json"

DEFAULT_TASKS = ["garden_mvrr_mod", "gss_subord_pp"]
DEFAULT_GAMMA_GRID = [0.1, 0.5]
DEFAULT_TRUST_BETA_GRID = [0.05, 0.1]
DEFAULT_REPAIR_ALPHA_GRID = [0.05, 0.1]
DEFAULT_REPAIR_SEL_GRID = [1.0, 2.0]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


race = load_module(RACE_PATH, "race_mod")
mctr = load_module(MCTR_PATH, "mctr_mod")
cmp = race.cmp
hdmi = race.hdmi


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json_map(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {task_result["task"]: task_result for task_result in data.get("tasks", [])}


def tv_distance_torch(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.abs(p - q).sum()


def project_to_basis(update: torch.Tensor, basis: torch.Tensor, mix: float) -> torch.Tensor:
    projected = update @ basis @ basis.T
    return (1.0 - mix) * update + mix * projected


def orthogonalize_against(update: torch.Tensor, direction: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    denom = torch.sum(direction * direction, dim=-1, keepdim=True).clamp_min(eps)
    coeff = torch.sum(update * direction, dim=-1, keepdim=True) / denom
    return update - coeff * direction


def compute_margin(head, h_var: torch.Tensor, target_token_id: int, source_token_id: int | None) -> torch.Tensor:
    z = head(h_var)
    if source_token_id is not None and source_token_id >= 0:
        return z[0, int(target_token_id)] - z[0, int(source_token_id)]
    return z[0, int(target_token_id)]


def two_stage_bcci_intervention(
    model,
    h: torch.Tensor,
    target_token_id: int,
    source_token_id: int | None,
    cf_zc: int,
    vZc,
    vZe,
    p_ze_before: torch.Tensor | None,
    basis: torch.Tensor,
    mix: float,
    flip_alpha: float,
    flip_steps: int,
    repair_alpha: float,
    repair_steps: int,
    anchor: torch.Tensor,
    trust_beta: float,
    gamma: float,
    repair_sel_weight: float,
) -> torch.Tensor:
    head = hdmi.lm_head_from_model(model)
    head_dtype = next(head.parameters()).dtype
    basis = basis.to(device=h.device, dtype=head_dtype)
    anchor = anchor.to(device=h.device, dtype=head_dtype)
    h_work = h.detach().to(dtype=head_dtype)

    goal = None
    if vZc is not None:
        probe_dim = int(vZc(h_work).shape[-1])
        goal = torch.zeros(probe_dim, device=h.device, dtype=head_dtype)
        goal[int(cf_zc)] = 1.0

    # Stage A: cross the behavior boundary with a modest anchor/probe regularizer.
    for _ in range(max(1, int(flip_steps))):
        with torch.enable_grad():
            h_var = h_work.clone().detach().requires_grad_(True)
            margin = compute_margin(head, h_var, target_token_id, source_token_id)
            if float(margin.item()) >= float(gamma):
                h_work = h_var.detach()
                break

            logprob_cf = F.log_softmax(vZc(h_var), dim=-1)[0, int(cf_zc)]
            anchor_penalty = torch.mean((h_var[0] - anchor) ** 2)
            ze_penalty = torch.zeros((), device=h.device, dtype=head_dtype)
            if vZe is not None and p_ze_before is not None:
                p_ze_after = torch.softmax(vZe(h_var), dim=-1)[0]
                ze_penalty = torch.mean((p_ze_after - p_ze_before) ** 2)

            # Hinge-style flip objective: once the margin reserve is satisfied, stop rewarding larger margins.
            flip_obj = -torch.relu(torch.tensor(float(gamma), device=h.device, dtype=head_dtype) - margin)
            objective = flip_obj + 0.5 * logprob_cf - 0.5 * ze_penalty - 0.05 * anchor_penalty
            grad_h = torch.autograd.grad(objective, h_var, retain_graph=False, create_graph=False)[0]

        update = project_to_basis(grad_h, basis, mix=mix)
        with torch.no_grad():
            h_candidate = h_var + float(flip_alpha) * update
            if trust_beta > 0.0:
                h_work = (1.0 - float(trust_beta)) * h_candidate + float(trust_beta) * anchor.unsqueeze(0)
            else:
                h_work = h_candidate

    # Stage B: repair probe metrics while staying on the flipped side of the boundary.
    for _ in range(max(1, int(repair_steps))):
        with torch.enable_grad():
            h_var = h_work.clone().detach().requires_grad_(True)
            margin = compute_margin(head, h_var, target_token_id, source_token_id)
            if float(margin.item()) < float(gamma):
                break

            p_c_after = torch.softmax(vZc(h_var), dim=-1)[0]
            comp_obj = -tv_distance_torch(p_c_after, goal)

            sel_obj = torch.zeros((), device=h.device, dtype=head_dtype)
            if vZe is not None and p_ze_before is not None:
                p_ze_after = torch.softmax(vZe(h_var), dim=-1)[0]
                m = max(1.0 - float(torch.min(p_ze_before).item()), float(torch.max(p_ze_before).item()))
                if m > 0:
                    sel_obj = -(tv_distance_torch(p_ze_after, p_ze_before) / m)

            anchor_penalty = torch.mean((h_var[0] - anchor) ** 2)
            repair_objective = comp_obj + float(repair_sel_weight) * sel_obj - 0.05 * anchor_penalty
            grad_repair = torch.autograd.grad(repair_objective, h_var, retain_graph=False, create_graph=False)[0]
            grad_margin = torch.autograd.grad(margin, h_var, retain_graph=False, create_graph=False)[0]

        update = project_to_basis(grad_repair, basis, mix=mix)
        update = orthogonalize_against(update, grad_margin)

        accepted = False
        for scale in [1.0, 0.5, 0.25, 0.1]:
            with torch.no_grad():
                h_candidate = h_work + float(repair_alpha) * float(scale) * update
                if trust_beta > 0.0:
                    h_candidate = (1.0 - float(trust_beta)) * h_candidate + float(trust_beta) * anchor.unsqueeze(0)
                candidate_margin = compute_margin(head, h_candidate, target_token_id, source_token_id)
                if float(candidate_margin.item()) >= float(gamma):
                    h_work = h_candidate
                    accepted = True
                    break
        if not accepted:
            break

    return h_work.to(dtype=h.dtype)


def evaluate_two_stage_bcci(
    model,
    tokenizer,
    samples,
    vZc,
    vZe,
    basis: torch.Tensor,
    mix: float,
    device: torch.device,
    layer_idx: int,
    flip_alpha: float,
    flip_steps: int,
    repair_alpha: float,
    repair_steps: int,
    train_base: np.ndarray,
    train_cf: np.ndarray,
    train_samples,
    k_neighbors: int,
    trust_beta: float,
    gamma: float,
    repair_sel_weight: float,
):
    model.eval()
    vZc.eval()
    if vZe is not None:
        vZe.eval()

    correct_base = 0
    correct_after = 0
    comp_vals = []
    sel_vals = []
    feasible_margin_count = 0

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

        h1 = two_stage_bcci_intervention(
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
            flip_alpha=flip_alpha,
            flip_steps=flip_steps,
            repair_alpha=repair_alpha,
            repair_steps=repair_steps,
            anchor=anchor,
            trust_beta=trust_beta,
            gamma=gamma,
            repair_sel_weight=repair_sel_weight,
        )

        if token_gold < 0 or token_alt < 0:
            lp_gold1 = lp_gold0
            lp_alt1 = lp_alt0
        else:
            gold_lp = hdmi.compute_first_step_logprob_from_h(model, h1, torch.tensor(token_gold, device=device)).item()
            alt_lp = hdmi.compute_first_step_logprob_from_h(model, h1, torch.tensor(token_alt, device=device)).item()
            lp_gold1 = lp_gold0 - float(base_gold["first_lp"].item()) + gold_lp
            lp_alt1 = lp_alt0 - float(base_alt["first_lp"].item()) + alt_lp

        if hdmi.choose_candidate_by_logprob(lp_gold1, lp_alt1) == 0:
            correct_after += 1

        if lp_alt1 - lp_gold1 >= gamma:
            feasible_margin_count += 1

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
        "completeness": completeness,
        "selectivity": selectivity,
        "reliability": hdmi.harmonic_mean(completeness, selectivity),
        "n": n,
        "margin_reserve_rate": feasible_margin_count / n if n else 0.0,
    }


def run_task(
    model,
    tokenizer,
    task: str,
    device: torch.device,
    layer_idx: int,
    hdmi_alpha: float,
    hdmi_inner_steps: int,
    base_cfg: Dict[str, float],
    gamma_grid: Sequence[float],
    trust_beta_grid: Sequence[float],
    repair_alpha_grid: Sequence[float],
    repair_sel_grid: Sequence[float],
):
    set_seed(1337)
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
    G_train = race.build_margin_gradients(model, tokenizer, train_samples, device, layer_idx)
    basis = race.learn_race_basis(
        base_features=X_train,
        cf_features=X_cf_train,
        gradients=G_train,
        rank=int(base_cfg["rank"]),
        grad_rank=int(base_cfg["grad_rank"]),
    )

    tune_samples = [train_samples[i] for i in te_i]
    tune_train_samples = [train_samples[i] for i in tr_i]
    hdmi_tune = hdmi.evaluate(
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
    hdmi_test = hdmi.evaluate(
        model=model,
        tokenizer=tokenizer,
        samples=test_samples,
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

    best_cfg = None
    best_metrics = None
    for gamma in gamma_grid:
        for trust_beta in trust_beta_grid:
            for repair_alpha in repair_alpha_grid:
                for repair_sel_weight in repair_sel_grid:
                    metrics = evaluate_two_stage_bcci(
                        model=model,
                        tokenizer=tokenizer,
                        samples=tune_samples,
                        vZc=vZc,
                        vZe=vZe,
                        basis=basis,
                        mix=float(base_cfg["mix"]),
                        device=device,
                        layer_idx=layer_idx,
                        flip_alpha=hdmi_alpha,
                        flip_steps=hdmi_inner_steps,
                        repair_alpha=float(repair_alpha),
                        repair_steps=8,
                        train_base=X_train[tr_i],
                        train_cf=X_cf_train[tr_i],
                        train_samples=tune_train_samples,
                        k_neighbors=16,
                        trust_beta=float(trust_beta),
                        gamma=float(gamma),
                        repair_sel_weight=float(repair_sel_weight),
                    )

                    if metrics["after_task_acc"] > 0.0:
                        continue
                    if metrics["completeness"] + 1e-6 < hdmi_tune.completeness:
                        continue
                    if metrics["selectivity"] + 1e-6 < hdmi_tune.selectivity:
                        continue

                    score = (
                        metrics["reliability"],
                        metrics["selectivity"],
                        metrics["completeness"],
                        metrics["margin_reserve_rate"],
                    )
                    if best_metrics is None:
                        best_cfg = {
                            "gamma": float(gamma),
                            "trust_beta": float(trust_beta),
                            "repair_alpha": float(repair_alpha),
                            "repair_sel_weight": float(repair_sel_weight),
                        }
                        best_metrics = metrics
                    else:
                        current_score = (
                            best_metrics["reliability"],
                            best_metrics["selectivity"],
                            best_metrics["completeness"],
                            best_metrics["margin_reserve_rate"],
                        )
                        if score > current_score:
                            best_cfg = {
                                "gamma": float(gamma),
                                "trust_beta": float(trust_beta),
                                "repair_alpha": float(repair_alpha),
                                "repair_sel_weight": float(repair_sel_weight),
                            }
                            best_metrics = metrics

    improved_test = None
    if best_cfg is not None:
        improved_test = evaluate_two_stage_bcci(
            model=model,
            tokenizer=tokenizer,
            samples=test_samples,
            vZc=vZc,
            vZe=vZe,
            basis=basis,
            mix=float(base_cfg["mix"]),
            device=device,
            layer_idx=layer_idx,
            flip_alpha=hdmi_alpha,
            flip_steps=hdmi_inner_steps,
            repair_alpha=float(best_cfg["repair_alpha"]),
            repair_steps=8,
            train_base=X_train,
            train_cf=X_cf_train,
            train_samples=train_samples,
            k_neighbors=16,
            trust_beta=float(best_cfg["trust_beta"]),
            gamma=float(best_cfg["gamma"]),
            repair_sel_weight=float(best_cfg["repair_sel_weight"]),
        )

    return {
        "task": task,
        "probe_metrics": {
            "interventional_zc_probe_acc": interventional_acc,
            "validation_zc_probe_acc": vzc_acc,
            "validation_ze_probe_acc": vze_acc,
        },
        "base_race_config": base_cfg,
        "search_grid": {
            "gamma": list(gamma_grid),
            "trust_beta": list(trust_beta_grid),
            "repair_alpha": list(repair_alpha_grid),
            "repair_sel_weight": list(repair_sel_grid),
        },
        "hdmi_tune": {
            "completeness": hdmi_tune.completeness,
            "selectivity": hdmi_tune.selectivity,
            "reliability": hdmi_tune.reliability,
        },
        "hdmi_test": {
            "baseline_task_acc": hdmi_test.baseline_task_acc,
            "after_task_acc": hdmi_test.after_task_acc,
            "delta_task_acc": hdmi_test.delta_task_acc,
            "completeness": hdmi_test.completeness,
            "selectivity": hdmi_test.selectivity,
            "reliability": hdmi_test.reliability,
            "n": hdmi_test.n,
        },
        "selected_two_stage_config": best_cfg,
        "selected_two_stage_tune_metrics": best_metrics,
        "two_stage_test": improved_test,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--model_name", default="EleutherAI/pythia-70m")
    parser.add_argument("--layer_idx", type=int, default=-1)
    parser.add_argument("--hdmi_alpha", type=float, default=1.0)
    parser.add_argument("--hdmi_inner_steps", type=int, default=30)
    parser.add_argument("--gamma_grid", nargs="+", type=float, default=DEFAULT_GAMMA_GRID)
    parser.add_argument("--trust_beta_grid", nargs="+", type=float, default=DEFAULT_TRUST_BETA_GRID)
    parser.add_argument("--repair_alpha_grid", nargs="+", type=float, default=DEFAULT_REPAIR_ALPHA_GRID)
    parser.add_argument("--repair_sel_grid", nargs="+", type=float, default=DEFAULT_REPAIR_SEL_GRID)
    parser.add_argument("--output", default="compare_hdmi_bcci_two_stage_failed_tasks.json")
    args = parser.parse_args()

    set_seed(1337)
    cmp.ensure_local_json(args.tasks)

    race_results_map = load_json_map(RACE_RESULTS_PATH)
    mctr_results_map = load_json_map(MCTR_RESULTS_PATH)
    bcci_results_map = load_json_map(BCCI_RESULTS_PATH)

    device = hdmi.get_device("auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device).eval()
    model.config.output_hidden_states = True

    results = {
        "method": "BCCI-two-stage-constrained",
        "tasks": [],
        "model_name": args.model_name,
        "layer_idx": args.layer_idx,
        "hdmi_alpha": args.hdmi_alpha,
        "hdmi_inner_steps": args.hdmi_inner_steps,
    }

    for task in args.tasks:
        print(f"Running two-stage BCCI on {task}")
        task_result = run_task(
            model=model,
            tokenizer=tokenizer,
            task=task,
            device=device,
            layer_idx=args.layer_idx,
            hdmi_alpha=args.hdmi_alpha,
            hdmi_inner_steps=args.hdmi_inner_steps,
            base_cfg=race_results_map[task]["best_tuning_config"],
            gamma_grid=args.gamma_grid,
            trust_beta_grid=args.trust_beta_grid,
            repair_alpha_grid=args.repair_alpha_grid,
            repair_sel_grid=args.repair_sel_grid,
        )
        if task in mctr_results_map:
            task_result["mctr_reference_test"] = mctr_results_map[task].get("mctr_test")
        if task in bcci_results_map:
            task_result["bcci_reference_test"] = bcci_results_map[task].get("bcci_test")
        results["tasks"].append(task_result)
        print(json.dumps(task_result, indent=2))

    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
