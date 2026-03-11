import argparse
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
RACE_PATH = ROOT / "compare_hdmi_race_exact.py"
RACE_RESULTS_PATH = ROOT / "compare_hdmi_race_all.json"
MCTR_RESULTS_PATH = ROOT / "compare_hdmi_mctr_failed_tasks.json"

DEFAULT_TASKS = ["garden_mvrr_mod", "gss_subord_pp"]
DEFAULT_ALPHA_GRID = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0]


def load_race_module():
    spec = importlib.util.spec_from_file_location("race_mod", RACE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


race = load_race_module()
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


def score_candidate_from_hidden(
    model,
    tokenizer,
    sample,
    h1: torch.Tensor,
    base_gold,
    base_alt,
    vZc,
    vZe,
    p_ze_before,
    device: torch.device,
):
    token_gold = int(base_gold["first_token_id"].item())
    token_alt = int(base_alt["first_token_id"].item())
    lp_gold0 = float(base_gold["total_lp"].item())
    lp_alt0 = float(base_alt["total_lp"].item())

    if token_gold < 0 or token_alt < 0:
        lp_gold1 = lp_gold0
        lp_alt1 = lp_alt0
    else:
        gold_lp = hdmi.compute_first_step_logprob_from_h(model, h1, torch.tensor(token_gold, device=device)).item()
        alt_lp = hdmi.compute_first_step_logprob_from_h(model, h1, torch.tensor(token_alt, device=device)).item()
        lp_gold1 = lp_gold0 - float(base_gold["first_lp"].item()) + gold_lp
        lp_alt1 = lp_alt0 - float(base_alt["first_lp"].item()) + alt_lp

    pred_after = hdmi.choose_candidate_by_logprob(lp_gold1, lp_alt1)
    p_c_after = torch.softmax(vZc(h1), dim=-1).detach().cpu().numpy()[0]
    goal = np.zeros_like(p_c_after)
    goal[int(sample.cf_zc)] = 1.0
    completeness = 1.0 - hdmi.tv_distance(p_c_after, goal)

    if vZe is not None and p_ze_before is not None:
        p_ze_after = torch.softmax(vZe(h1), dim=-1).detach().cpu().numpy()[0]
        m = max(1.0 - float(np.min(p_ze_before)), float(np.max(p_ze_before)))
        selectivity = 1.0 - (hdmi.tv_distance(p_ze_after, p_ze_before) / m if m > 0 else 0.0)
    else:
        selectivity = 1.0

    return {
        "pred_after": pred_after,
        "lp_margin_after": lp_alt1 - lp_gold1,
        "completeness": float(completeness),
        "selectivity": float(selectivity),
    }


def choose_macc_candidate(
    model,
    tokenizer,
    sample,
    h0: torch.Tensor,
    base_gold,
    base_alt,
    vZc,
    vZe,
    p_ze_before,
    basis: torch.Tensor,
    mix: float,
    alpha_grid: Sequence[float],
    inner_steps: int,
    device: torch.device,
):
    candidates = []
    for alpha in alpha_grid:
        h1 = cmp.hybrid_intervention(
            hdmi=hdmi,
            model=model,
            h=h0,
            target_token_id=int(base_alt["first_token_id"].item()),
            source_token_id=int(base_gold["first_token_id"].item()),
            basis=basis,
            mix=mix,
            alpha=float(alpha),
            inner_steps=inner_steps,
            use_margin=True,
            normalize_grad=False,
            grad_clip_norm=0.0,
        )
        candidate = score_candidate_from_hidden(
            model=model,
            tokenizer=tokenizer,
            sample=sample,
            h1=h1,
            base_gold=base_gold,
            base_alt=base_alt,
            vZc=vZc,
            vZe=vZe,
            p_ze_before=p_ze_before,
            device=device,
        )
        candidate["alpha"] = float(alpha)
        candidates.append(candidate)

    crossing_candidates = [candidate for candidate in candidates if candidate["pred_after"] == 1]
    if crossing_candidates:
        selected = min(crossing_candidates, key=lambda candidate: (candidate["alpha"], -candidate["selectivity"]))
        crossing_found = True
    else:
        selected = max(candidates, key=lambda candidate: (candidate["lp_margin_after"], candidate["selectivity"], -candidate["alpha"]))
        crossing_found = False

    return selected, crossing_found, candidates


def evaluate_macc(
    model,
    tokenizer,
    samples,
    vZc,
    vZe,
    basis: torch.Tensor,
    mix: float,
    alpha_grid: Sequence[float],
    device: torch.device,
    layer_idx: int,
    inner_steps: int,
):
    model.eval()
    vZc.eval()
    if vZe is not None:
        vZe.eval()

    correct_base = 0
    correct_after = 0
    comp_vals = []
    sel_vals = []
    alpha_counter: Counter[float] = Counter()
    crossing_found = 0
    fallback_used = 0
    sample_diagnostics = []

    for sample in samples:
        base_gold = hdmi.sequence_logprob(model, tokenizer, sample.prompt_text, sample.gold_cont, device)
        base_alt = hdmi.sequence_logprob(model, tokenizer, sample.prompt_text, sample.alt_cont, device)
        lp_gold0 = float(base_gold["total_lp"].item())
        lp_alt0 = float(base_alt["total_lp"].item())
        if hdmi.choose_candidate_by_logprob(lp_gold0, lp_alt0) == 0:
            correct_base += 1

        prefix_ids = hdmi.encode(tokenizer, sample.prompt_text, device)
        h0 = hdmi.get_prompt_hidden_at_layer(model, prefix_ids, layer_idx=layer_idx)

        if vZe is not None:
            p_ze_before = torch.softmax(vZe(h0), dim=-1).detach().cpu().numpy()[0]
        else:
            p_ze_before = None

        selected, found_crossing, candidates = choose_macc_candidate(
            model=model,
            tokenizer=tokenizer,
            sample=sample,
            h0=h0,
            base_gold=base_gold,
            base_alt=base_alt,
            vZc=vZc,
            vZe=vZe,
            p_ze_before=p_ze_before,
            basis=basis,
            mix=mix,
            alpha_grid=alpha_grid,
            inner_steps=inner_steps,
            device=device,
        )

        if selected["pred_after"] == 0:
            correct_after += 1
        comp_vals.append(selected["completeness"])
        sel_vals.append(selected["selectivity"])
        alpha_counter[selected["alpha"]] += 1

        if found_crossing:
            crossing_found += 1
        else:
            fallback_used += 1

        sample_diagnostics.append(
            {
                "selected_alpha": selected["alpha"],
                "crossing_found": found_crossing,
                "selected_lp_margin_after": selected["lp_margin_after"],
                "candidate_lp_margins": [candidate["lp_margin_after"] for candidate in candidates],
            }
        )

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
        "crossing_found_rate": crossing_found / n if n else 0.0,
        "fallback_rate": fallback_used / n if n else 0.0,
        "selected_alpha_hist": {str(alpha): int(count) for alpha, count in sorted(alpha_counter.items())},
        "selected_alpha_mean": float(np.mean([diag["selected_alpha"] for diag in sample_diagnostics])) if sample_diagnostics else 0.0,
    }


def run_task(
    model,
    tokenizer,
    task: str,
    device: torch.device,
    layer_idx: int,
    inner_steps: int,
    alpha_grid: Sequence[float],
    base_cfg: Dict[str, float],
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
    hdmi_tune = hdmi.evaluate(
        model=model,
        tokenizer=tokenizer,
        samples=tune_samples,
        probe_c_interv=probe_c,
        vZc=vZc,
        vZe=vZe,
        intervention="hdmi",
        hdmi_alpha=1.0,
        hdmi_inner_steps=inner_steps,
        hdmi_use_margin=True,
        hdmi_normalize_grad=False,
        hdmi_grad_clip_norm=0.0,
        device=device,
        layer_idx=layer_idx,
        verbose=False,
    )
    race_test = cmp.evaluate_hybrid(
        hdmi=hdmi,
        model=model,
        tokenizer=tokenizer,
        samples=test_samples,
        vZc=vZc,
        vZe=vZe,
        basis=basis,
        mix=float(base_cfg["mix"]),
        device=device,
        layer_idx=layer_idx,
        hdmi_alpha=1.0,
        hdmi_inner_steps=inner_steps,
        hdmi_use_margin=True,
        hdmi_normalize_grad=False,
        hdmi_grad_clip_norm=0.0,
    )
    hdmi_test = hdmi.evaluate(
        model=model,
        tokenizer=tokenizer,
        samples=test_samples,
        probe_c_interv=probe_c,
        vZc=vZc,
        vZe=vZe,
        intervention="hdmi",
        hdmi_alpha=1.0,
        hdmi_inner_steps=inner_steps,
        hdmi_use_margin=True,
        hdmi_normalize_grad=False,
        hdmi_grad_clip_norm=0.0,
        device=device,
        layer_idx=layer_idx,
        verbose=False,
    )
    macc_tune = evaluate_macc(
        model=model,
        tokenizer=tokenizer,
        samples=tune_samples,
        vZc=vZc,
        vZe=vZe,
        basis=basis,
        mix=float(base_cfg["mix"]),
        alpha_grid=alpha_grid,
        device=device,
        layer_idx=layer_idx,
        inner_steps=inner_steps,
    )
    macc_test = evaluate_macc(
        model=model,
        tokenizer=tokenizer,
        samples=test_samples,
        vZc=vZc,
        vZe=vZe,
        basis=basis,
        mix=float(base_cfg["mix"]),
        alpha_grid=alpha_grid,
        device=device,
        layer_idx=layer_idx,
        inner_steps=inner_steps,
    )

    return {
        "task": task,
        "probe_metrics": {
            "interventional_zc_probe_acc": interventional_acc,
            "validation_zc_probe_acc": vzc_acc,
            "validation_ze_probe_acc": vze_acc,
        },
        "base_race_config": base_cfg,
        "alpha_grid": list(alpha_grid),
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
        "race_test": race_test,
        "macc_tune": macc_tune,
        "macc_test": macc_test,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--model_name", default="EleutherAI/pythia-70m")
    parser.add_argument("--layer_idx", type=int, default=-1)
    parser.add_argument("--hdmi_inner_steps", type=int, default=30)
    parser.add_argument("--alpha_grid", nargs="+", type=float, default=DEFAULT_ALPHA_GRID)
    parser.add_argument("--output", default="compare_hdmi_macc_failed_tasks.json")
    args = parser.parse_args()

    set_seed(1337)
    cmp.ensure_local_json(args.tasks)

    race_results_map = load_json_map(RACE_RESULTS_PATH)
    mctr_results_map = load_json_map(MCTR_RESULTS_PATH)

    device = hdmi.get_device("auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device).eval()
    model.config.output_hidden_states = True

    results = {
        "method": "MACC-minimum-action-crossing",
        "tasks": [],
        "model_name": args.model_name,
        "layer_idx": args.layer_idx,
        "hdmi_inner_steps": args.hdmi_inner_steps,
        "alpha_grid": list(args.alpha_grid),
    }

    for task in args.tasks:
        print(f"Running MACC on {task}")
        task_result = run_task(
            model=model,
            tokenizer=tokenizer,
            task=task,
            device=device,
            layer_idx=args.layer_idx,
            inner_steps=args.hdmi_inner_steps,
            alpha_grid=args.alpha_grid,
            base_cfg=race_results_map[task]["best_tuning_config"],
        )
        if task in mctr_results_map:
            task_result["mctr_reference_test"] = mctr_results_map[task].get("mctr_test")
        results["tasks"].append(task_result)
        print(json.dumps(task_result, indent=2))

    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
