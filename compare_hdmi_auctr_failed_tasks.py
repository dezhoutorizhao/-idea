import argparse
import importlib.util
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
MCTR_PATH = ROOT / "compare_hdmi_mctr_failed_tasks.py"

DEFAULT_TASKS = [
    "garden_mvrr_mod",
    "garden_npz_v-trans_mod",
    "garden_npz_obj_mod",
]


def load_mctr_module():
    spec = importlib.util.spec_from_file_location("mctr_mod", MCTR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


mctr = load_mctr_module()
race = mctr.race
cmp = mctr.cmp
hdmi = mctr.hdmi


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auctr_intervention(
    model,
    h: torch.Tensor,
    target_token_id: int,
    source_token_id: int | None,
    basis: torch.Tensor,
    mix: float,
    alpha: float,
    inner_steps: int,
    anchor: torch.Tensor,
    beta_base: float,
    beta_scale: float,
    beta_max: float,
) -> torch.Tensor:
    head = hdmi.lm_head_from_model(model)
    head_dtype = next(head.parameters()).dtype
    basis = basis.to(device=h.device, dtype=head_dtype)
    anchor = anchor.to(device=h.device, dtype=head_dtype)
    h_work = h.detach().to(dtype=head_dtype)

    for _ in range(max(1, int(inner_steps))):
        with torch.enable_grad():
            h_var = h_work.clone().detach().requires_grad_(True)
            z = head(h_var)
            if source_token_id is not None and source_token_id >= 0:
                obj = z[0, int(target_token_id)] - z[0, int(source_token_id)]
            else:
                obj = z[0, int(target_token_id)]
            grad_h = torch.autograd.grad(obj, h_var, retain_graph=False, create_graph=False)[0]
        projected = grad_h @ basis @ basis.T
        update = (1.0 - mix) * grad_h + mix * projected
        with torch.no_grad():
            h_candidate = h_var + alpha * update
            dist = torch.norm(h_candidate[0] - anchor)
            ref = torch.norm(h_var[0] - anchor).clamp_min(1e-6)
            ratio = float((dist / ref).item())
            beta = min(beta_max, max(0.0, beta_base + beta_scale * max(0.0, ratio - 1.0)))
            h_work = (1.0 - beta) * h_candidate + beta * anchor.unsqueeze(0)
    return h_work.to(dtype=h.dtype)


def evaluate_auctr(
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
    beta_base: float,
    beta_scale: float,
    beta_max: float,
):
    model.eval()
    vZc.eval()
    if vZe is not None:
        vZe.eval()

    correct_base = 0
    correct_after = 0
    comp_vals = []
    sel_vals = []

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
        anchor = mctr.find_local_cf_anchor(
            h_vec=h0_vec,
            sample=sample,
            train_base=train_base,
            train_cf=train_cf,
            train_samples=train_samples,
            k=k_neighbors,
        )

        if vZe is not None:
            p_ze_before = torch.softmax(vZe(h0), dim=-1).detach().cpu().numpy()[0]
        else:
            p_ze_before = None

        token_gold = int(base_gold["first_token_id"].item())
        token_alt = int(base_alt["first_token_id"].item())
        h1 = auctr_intervention(
            model=model,
            h=h0,
            target_token_id=token_alt,
            source_token_id=token_gold,
            basis=basis,
            mix=mix,
            alpha=alpha,
            inner_steps=inner_steps,
            anchor=anchor,
            beta_base=beta_base,
            beta_scale=beta_scale,
            beta_max=beta_max,
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

        p_c_after = torch.softmax(vZc(h1), dim=-1).detach().cpu().numpy()[0]
        goal = np.zeros_like(p_c_after)
        goal[int(sample.cf_zc)] = 1.0
        comp_vals.append(1.0 - hdmi.tv_distance(p_c_after, goal))

        if vZe is not None and p_ze_before is not None:
            p_ze_after = torch.softmax(vZe(h1), dim=-1).detach().cpu().numpy()[0]
            m = max(1.0 - float(np.min(p_ze_before)), float(np.max(p_ze_before)))
            sel_vals.append(1.0 - (hdmi.tv_distance(p_ze_after, p_ze_before) / m if m > 0 else 0.0))
        else:
            sel_vals.append(1.0)

    n = len(samples)
    completeness = float(np.mean(comp_vals)) if comp_vals else 0.0
    selectivity = float(np.mean(sel_vals)) if sel_vals else 1.0
    return {
        "baseline_task_acc": correct_base / n if n else 0.0,
        "after_task_acc": correct_after / n if n else 0.0,
        "delta_task_acc": (correct_after - correct_base) / n if n else 0.0,
        "completeness": completeness,
        "selectivity": selectivity,
        "reliability": hdmi.harmonic_mean(completeness, selectivity),
        "n": n,
    }


def run_task(model, tokenizer, task: str, device: torch.device, layer_idx: int, alpha: float, inner_steps: int, base_cfg: Dict[str, float]):
    set_seed(1337)
    splits = hdmi._load_causalgym_local_json(str(cmp.LOCAL_JSON_DIR))
    train_records = [r for r in splits["train"] if r["task"] == task]
    dev_records = [r for r in splits["dev"] if r["task"] == task]
    test_records = [r for r in splits["test"] if r["task"] == task]
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
    basis = race.learn_race_basis(X_train, X_cf_train, G_train, int(base_cfg["rank"]), int(base_cfg["grad_rank"]))

    tune_samples = [train_samples[i] for i in te_i]
    hdmi_tune = hdmi.evaluate(
        model=model,
        tokenizer=tokenizer,
        samples=tune_samples,
        probe_c_interv=probe_c,
        vZc=vZc,
        vZe=vZe,
        intervention="hdmi",
        hdmi_alpha=alpha,
        hdmi_inner_steps=inner_steps,
        hdmi_use_margin=True,
        hdmi_normalize_grad=False,
        hdmi_grad_clip_norm=0.0,
        device=device,
        layer_idx=layer_idx,
        verbose=False,
    )

    best_cfg = None
    best_metrics = None
    for beta_base in [0.0, 0.02, 0.05, 0.1]:
        for beta_scale in [0.05, 0.1, 0.2, 0.4]:
            for beta_max in [0.1, 0.2, 0.35, 0.5]:
                if beta_max < beta_base:
                    continue
                for k_neighbors in [1, 4, 8, 16]:
                    metrics = evaluate_auctr(
                        model=model,
                        tokenizer=tokenizer,
                        samples=tune_samples,
                        vZc=vZc,
                        vZe=vZe,
                        basis=basis,
                        mix=float(base_cfg["mix"]),
                        device=device,
                        layer_idx=layer_idx,
                        alpha=alpha,
                        inner_steps=inner_steps,
                        train_base=X_train[tr_i],
                        train_cf=X_cf_train[tr_i],
                        train_samples=[train_samples[i] for i in tr_i],
                        k_neighbors=k_neighbors,
                        beta_base=beta_base,
                        beta_scale=beta_scale,
                        beta_max=beta_max,
                    )
                    if metrics["selectivity"] + 1e-6 < hdmi_tune.selectivity:
                        continue
                    if metrics["completeness"] <= hdmi_tune.completeness:
                        continue
                    score = (metrics["completeness"] - hdmi_tune.completeness, metrics["reliability"])
                    if best_metrics is None:
                        best_cfg = {
                            "beta_base": beta_base,
                            "beta_scale": beta_scale,
                            "beta_max": beta_max,
                            "k_neighbors": k_neighbors,
                        }
                        best_metrics = metrics
                    else:
                        current_score = (
                            best_metrics["completeness"] - hdmi_tune.completeness,
                            best_metrics["reliability"],
                        )
                        if score > current_score:
                            best_cfg = {
                                "beta_base": beta_base,
                                "beta_scale": beta_scale,
                                "beta_max": beta_max,
                                "k_neighbors": k_neighbors,
                            }
                            best_metrics = metrics

    hdmi_test = hdmi.evaluate(
        model=model,
        tokenizer=tokenizer,
        samples=test_samples,
        probe_c_interv=probe_c,
        vZc=vZc,
        vZe=vZe,
        intervention="hdmi",
        hdmi_alpha=alpha,
        hdmi_inner_steps=inner_steps,
        hdmi_use_margin=True,
        hdmi_normalize_grad=False,
        hdmi_grad_clip_norm=0.0,
        device=device,
        layer_idx=layer_idx,
        verbose=False,
    )

    if best_cfg is None:
        auctr_test = None
    else:
        auctr_test = evaluate_auctr(
            model=model,
            tokenizer=tokenizer,
            samples=test_samples,
            vZc=vZc,
            vZe=vZe,
            basis=basis,
            mix=float(base_cfg["mix"]),
            device=device,
            layer_idx=layer_idx,
            alpha=alpha,
            inner_steps=inner_steps,
            train_base=X_train,
            train_cf=X_cf_train,
            train_samples=train_samples,
            k_neighbors=int(best_cfg["k_neighbors"]),
            beta_base=float(best_cfg["beta_base"]),
            beta_scale=float(best_cfg["beta_scale"]),
            beta_max=float(best_cfg["beta_max"]),
        )

    return {
        "task": task,
        "probe_metrics": {
            "interventional_zc_probe_acc": interventional_acc,
            "validation_zc_probe_acc": vzc_acc,
            "validation_ze_probe_acc": vze_acc,
        },
        "base_race_config": base_cfg,
        "hdmi_tune": {
            "completeness": hdmi_tune.completeness,
            "selectivity": hdmi_tune.selectivity,
            "reliability": hdmi_tune.reliability,
        },
        "selected_auctr_config": best_cfg,
        "selected_auctr_tune_metrics": best_metrics,
        "hdmi_test": {
            "completeness": hdmi_test.completeness,
            "selectivity": hdmi_test.selectivity,
            "reliability": hdmi_test.reliability,
        },
        "auctr_test": auctr_test,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--model_name", default="EleutherAI/pythia-70m")
    parser.add_argument("--layer_idx", type=int, default=-1)
    parser.add_argument("--hdmi_alpha", type=float, default=1.0)
    parser.add_argument("--hdmi_inner_steps", type=int, default=30)
    parser.add_argument("--output", default="compare_hdmi_auctr_failed_tasks.json")
    args = parser.parse_args()

    set_seed(1337)
    cmp.ensure_local_json(args.tasks)
    all_results = json.loads((ROOT / "compare_hdmi_race_all.json").read_text(encoding="utf-8"))
    race_cfg_map = {task_result["task"]: task_result["best_tuning_config"] for task_result in all_results["tasks"]}

    device = hdmi.get_device("auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device).eval()
    model.config.output_hidden_states = True

    results = {
        "method": "AUCTR-RACE",
        "tasks": [],
    }
    for task in args.tasks:
        print(f"Running AUCTR on {task}")
        task_result = run_task(
            model=model,
            tokenizer=tokenizer,
            task=task,
            device=device,
            layer_idx=args.layer_idx,
            alpha=args.hdmi_alpha,
            inner_steps=args.hdmi_inner_steps,
            base_cfg=race_cfg_map[task],
        )
        results["tasks"].append(task_result)
        print(json.dumps(task_result, indent=2))

    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
