import argparse
import importlib.util
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
HDMI_PATH = ROOT / "hdmi_supplementary" / "hdmi-code" / "probing.py"
LOCAL_JSON_DIR = ROOT / "cg_hdmi_three_tasks"
TASKS = ["agr_sv_num_pp", "gss_subord_pp", "filler_gap_pp"]
HDMI_REPORTED = {
    "agr_sv_num_pp": {"completeness": 0.9984, "selectivity": 0.9409, "reliability": 0.9688},
    "gss_subord_pp": {"completeness": 0.9984, "selectivity": 0.7371, "reliability": 0.8481},
    "filler_gap_pp": {"completeness": 0.7404, "selectivity": 0.6275, "reliability": 0.6793},
}


def load_hdmi():
    spec = importlib.util.spec_from_file_location("hdmi_probe", HDMI_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def ensure_local_json(tasks: Sequence[str]) -> None:
    LOCAL_JSON_DIR.mkdir(exist_ok=True)
    ds = load_dataset("aryaman/causalgym")
    split_map = {"train": "train", "validation": "dev", "test": "test"}
    task_set = set(tasks)
    for src_split, dst_split in split_map.items():
        rows = [r for r in ds[src_split] if r["task"] in task_set]
        with open(LOCAL_JSON_DIR / f"{dst_split}.json", "w", encoding="utf-8") as f:
            json.dump(rows, f)


def build_cf_prompts(records: Sequence[dict]) -> List[str]:
    prompts = []
    for row in records:
        base_text = "".join(row["base"])
        src_text = "".join(row["src"])
        prompts.append(src_text)
        prompts.append(base_text)
    return prompts


@torch.no_grad()
def build_features_for_texts(hdmi, model, tokenizer, texts: Sequence[str], device: torch.device, layer_idx: int) -> np.ndarray:
    feats = []
    for text in texts:
        ids = hdmi.encode(tokenizer, text, device)
        h = hdmi.get_prompt_hidden_at_layer(model, ids, layer_idx=layer_idx)
        feats.append(h[0].float().cpu().numpy())
    return np.stack(feats, axis=0)


def learn_basis(base_features: np.ndarray, cf_features: np.ndarray, rank: int) -> torch.Tensor:
    delta = torch.from_numpy(cf_features - base_features).float()
    delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    _, _, vh = torch.linalg.svd(delta, full_matrices=False)
    rank = min(rank, vh.size(0))
    return vh[:rank].T.contiguous()


def hybrid_intervention(
    hdmi,
    model,
    h: torch.Tensor,
    target_token_id: int,
    source_token_id: int | None,
    basis: torch.Tensor,
    mix: float,
    alpha: float,
    inner_steps: int,
    use_margin: bool,
    normalize_grad: bool,
    grad_clip_norm: float,
) -> torch.Tensor:
    head = hdmi.lm_head_from_model(model)
    head_dtype = next(head.parameters()).dtype
    basis = basis.to(device=h.device, dtype=head_dtype)
    h_work = h.detach().to(dtype=head_dtype)
    eps = 1e-9
    for _ in range(max(1, int(inner_steps))):
        with torch.enable_grad():
            h_var = h_work.clone().detach().requires_grad_(True)
            z = head(h_var)
            if use_margin and source_token_id is not None and source_token_id >= 0:
                obj = z[0, int(target_token_id)] - z[0, int(source_token_id)]
            else:
                obj = z[0, int(target_token_id)]
            grad_h = torch.autograd.grad(obj, h_var, retain_graph=False, create_graph=False)[0]
        projected = grad_h @ basis @ basis.T
        grad_h = (1.0 - mix) * grad_h + mix * projected
        if grad_clip_norm and grad_clip_norm > 0.0:
            n = grad_h.norm(p=2, dim=-1, keepdim=True)
            grad_h = torch.where(n > grad_clip_norm, grad_h * (grad_clip_norm / (n + eps)), grad_h)
        if normalize_grad:
            n = grad_h.norm(p=2, dim=-1, keepdim=True)
            grad_h = grad_h / (n + eps)
        with torch.no_grad():
            h_work = (h_var + alpha * grad_h).detach()
    return h_work.to(dtype=h.dtype)


def evaluate_hybrid(
    hdmi,
    model,
    tokenizer,
    samples,
    vZc,
    vZe,
    basis: torch.Tensor,
    mix: float,
    device: torch.device,
    layer_idx: int,
    hdmi_alpha: float,
    hdmi_inner_steps: int,
    hdmi_use_margin: bool,
    hdmi_normalize_grad: bool,
    hdmi_grad_clip_norm: float,
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

        if vZe is not None:
            p_ze_before = torch.softmax(vZe(h0), dim=-1).detach().cpu().numpy()[0]
        else:
            p_ze_before = None

        token_gold = int(base_gold["first_token_id"].item())
        token_alt = int(base_alt["first_token_id"].item())
        h1 = hybrid_intervention(
            hdmi=hdmi,
            model=model,
            h=h0,
            target_token_id=token_alt,
            source_token_id=token_gold,
            basis=basis,
            mix=mix,
            alpha=hdmi_alpha,
            inner_steps=hdmi_inner_steps,
            use_margin=hdmi_use_margin,
            normalize_grad=hdmi_normalize_grad,
            grad_clip_norm=hdmi_grad_clip_norm,
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
    }


def train_probes_exact(hdmi, train_samples, val_samples, model, tokenizer, device: torch.device, layer_idx: int):
    X_train = hdmi.build_features(model, tokenizer, train_samples, device, layer_idx=layer_idx)
    y_train_zc = np.array([s.zc for s in train_samples], dtype=np.int64)
    X_val = hdmi.build_features(model, tokenizer, val_samples, device, layer_idx=layer_idx)
    y_val_zc = np.array([s.zc for s in val_samples], dtype=np.int64)

    rng = np.random.default_rng(1338)
    perm = rng.permutation(len(train_samples))
    cut = int(0.8 * len(perm))
    tr_i, te_i = perm[:cut], perm[cut:]
    probe_c = hdmi.train_probe(
        X_train[tr_i],
        y_train_zc[tr_i],
        epochs=150,
        lr=1e-2,
        wb=1e-6,
        device=device,
        hidden=256,
        batch_size=256,
        verbose=False,
    )
    interventional_acc = hdmi.eval_probe_acc(probe_c, X_train[te_i], y_train_zc[te_i], device=device)

    rng2 = np.random.default_rng(1340)
    perm2 = rng2.permutation(len(val_samples))
    cut2 = int(0.8 * len(perm2))
    tr_v_idx, va_v_idx = perm2[:cut2], perm2[cut2:]
    vZc = hdmi.train_validation_probe_with_selection(
        X_val[tr_v_idx],
        y_val_zc[tr_v_idx],
        X_val[va_v_idx],
        y_val_zc[va_v_idx],
        hidden_grid=(0, 64, 256, 512),
        device=device,
        epochs=150,
        lr=1e-2,
        wd=1e-6,
        batch_size=256,
    )
    vzc_acc = hdmi.eval_probe_acc(vZc, X_val[va_v_idx], y_val_zc[va_v_idx], device=device)

    y_train_ze = np.array([s.ze for s in train_samples], dtype=np.int64)
    y_val_ze = np.array([s.ze for s in val_samples], dtype=np.int64)
    train_mask = y_train_ze >= 0
    val_mask = y_val_ze >= 0
    vZe = hdmi.train_validation_probe_with_selection(
        X_train[train_mask],
        y_train_ze[train_mask],
        X_val[val_mask],
        y_val_ze[val_mask],
        hidden_grid=(0, 64, 256, 512),
        device=device,
        epochs=150,
        lr=1e-2,
        wd=1e-6,
        batch_size=256,
    )
    vze_acc = hdmi.eval_probe_acc(vZe, X_val[val_mask], y_val_ze[val_mask], device=device)
    return probe_c, vZc, vZe, interventional_acc, vzc_acc, vze_acc, X_train, tr_i, te_i


def run_task(hdmi, model, tokenizer, task: str, device: torch.device, layer_idx: int, hdmi_alpha: float, hdmi_inner_steps: int):
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1337)

    splits = hdmi._load_causalgym_local_json(str(LOCAL_JSON_DIR))
    train_records = [r for r in splits["train"] if r["task"] == task]
    dev_records = [r for r in splits["dev"] if r["task"] == task]
    test_records = [r for r in splits["test"] if r["task"] == task]

    train_samples, val_samples, test_samples = hdmi.make_causalgym_samples(
        {"train": train_records, "dev": dev_records, "test": test_records},
        tasks=[task],
    )
    cf_prompts_train = build_cf_prompts(train_records)
    X_cf_train = build_features_for_texts(hdmi, model, tokenizer, cf_prompts_train, device, layer_idx)

    probe_c, vZc, vZe, interventional_acc, vzc_acc, vze_acc, X_train, tr_i, te_i = train_probes_exact(
        hdmi, train_samples, val_samples, model, tokenizer, device, layer_idx
    )

    hdmi_metrics = hdmi.evaluate(
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

    tune_samples = [train_samples[i] for i in te_i]
    best_cfg = None
    best_metrics = None
    for rank in [1, 2, 4, 8, 16, 32]:
        basis = learn_basis(X_train[tr_i], X_cf_train[tr_i], rank)
        for mix in [0.25, 0.5, 0.75, 1.0]:
            metrics = evaluate_hybrid(
                hdmi=hdmi,
                model=model,
                tokenizer=tokenizer,
                samples=tune_samples,
                vZc=vZc,
                vZe=vZe,
                basis=basis,
                mix=mix,
                device=device,
                layer_idx=layer_idx,
                hdmi_alpha=hdmi_alpha,
                hdmi_inner_steps=hdmi_inner_steps,
                hdmi_use_margin=True,
                hdmi_normalize_grad=False,
                hdmi_grad_clip_norm=0.0,
            )
            if best_metrics is None or metrics["reliability"] > best_metrics["reliability"]:
                best_metrics = metrics
                best_cfg = {"rank": rank, "mix": mix}

    assert best_cfg is not None and best_metrics is not None
    basis = learn_basis(X_train, X_cf_train, best_cfg["rank"])
    hybrid_metrics = evaluate_hybrid(
        hdmi=hdmi,
        model=model,
        tokenizer=tokenizer,
        samples=test_samples,
        vZc=vZc,
        vZe=vZe,
        basis=basis,
        mix=best_cfg["mix"],
        device=device,
        layer_idx=layer_idx,
        hdmi_alpha=hdmi_alpha,
        hdmi_inner_steps=hdmi_inner_steps,
        hdmi_use_margin=True,
        hdmi_normalize_grad=False,
        hdmi_grad_clip_norm=0.0,
    )

    return {
        "task": task,
        "sample_sizes": {"train": len(train_samples), "validation": len(val_samples), "test": len(test_samples)},
        "probe_metrics": {
            "interventional_zc_probe_acc": interventional_acc,
            "validation_zc_probe_acc": vzc_acc,
            "validation_ze_probe_acc": vze_acc,
        },
        "best_tuning_config": best_cfg,
        "best_tuning_metrics": best_metrics,
        "hdmi_local": asdict(hdmi_metrics),
        "hybrid_local": hybrid_metrics,
        "hdmi_reported": HDMI_REPORTED[task],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="EleutherAI/pythia-70m")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--layer_idx", type=int, default=-1)
    parser.add_argument("--hdmi_alpha", type=float, default=1.0)
    parser.add_argument("--hdmi_inner_steps", type=int, default=30)
    parser.add_argument("--output", default="compare_hdmi_hybrid_exact.json")
    args = parser.parse_args()

    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)

    ensure_local_json(args.tasks)
    device = hdmi.get_device("auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device).eval()
    model.config.output_hidden_states = True

    results = {
        "model_name": args.model_name,
        "layer_idx": args.layer_idx,
        "hdmi_alpha": args.hdmi_alpha,
        "hdmi_inner_steps": args.hdmi_inner_steps,
        "tasks": [],
    }
    for task in args.tasks:
        print(f"Running exact comparison for {task}")
        task_result = run_task(
            hdmi=hdmi,
            model=model,
            tokenizer=tokenizer,
            task=task,
            device=device,
            layer_idx=args.layer_idx,
            hdmi_alpha=args.hdmi_alpha,
            hdmi_inner_steps=args.hdmi_inner_steps,
        )
        results["tasks"].append(task_result)
        print(json.dumps(task_result, indent=2))

    def mean_for(key: str) -> Dict[str, float]:
        return {
            metric: float(np.mean([task_result[key][metric] for task_result in results["tasks"]]))
            for metric in ["completeness", "selectivity", "reliability"]
        }

    results["means"] = {
        "hdmi_reported": mean_for("hdmi_reported"),
        "hdmi_local": mean_for("hdmi_local"),
        "hybrid_local": mean_for("hybrid_local"),
    }
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["means"], indent=2))


hdmi = load_hdmi()


if __name__ == "__main__":
    main()
