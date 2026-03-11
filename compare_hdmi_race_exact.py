import argparse
import importlib.util
import json
import random
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
BASE_COMPARE_PATH = ROOT / "compare_hdmi_hybrid_exact.py"
TASKS = ["agr_sv_num_pp", "gss_subord_pp", "filler_gap_pp"]


def load_compare_module():
    spec = importlib.util.spec_from_file_location("hdmi_hybrid_compare", BASE_COMPARE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cmp = load_compare_module()
hdmi = cmp.hdmi


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_margin_gradients(
    model,
    tokenizer,
    samples,
    device: torch.device,
    layer_idx: int,
) -> np.ndarray:
    head = hdmi.lm_head_from_model(model)
    head_dtype = next(head.parameters()).dtype
    grads = []

    for sample in samples:
        base_gold = hdmi.sequence_logprob(model, tokenizer, sample.prompt_text, sample.gold_cont, device)
        base_alt = hdmi.sequence_logprob(model, tokenizer, sample.prompt_text, sample.alt_cont, device)
        token_gold = int(base_gold["first_token_id"].item())
        token_alt = int(base_alt["first_token_id"].item())

        prefix_ids = hdmi.encode(tokenizer, sample.prompt_text, device)
        h0 = hdmi.get_prompt_hidden_at_layer(model, prefix_ids, layer_idx=layer_idx)

        if token_alt < 0:
            grads.append(np.zeros(h0.shape[-1], dtype=np.float32))
            continue

        with torch.enable_grad():
            h_var = h0.detach().to(dtype=head_dtype).clone().requires_grad_(True)
            z = head(h_var)
            if token_gold >= 0:
                objective = z[0, token_alt] - z[0, token_gold]
            else:
                objective = z[0, token_alt]
            grad_h = torch.autograd.grad(objective, h_var, retain_graph=False, create_graph=False)[0]
        grads.append(grad_h[0].float().detach().cpu().numpy())

    return np.stack(grads, axis=0)


def compute_basis_diagnostics(basis: torch.Tensor, gradients: np.ndarray) -> Dict[str, float]:
    grad_t = torch.from_numpy(gradients).float()
    proj = grad_t @ basis @ basis.T
    grad_norm = grad_t.norm(dim=-1).clamp_min(1e-9)
    proj_norm = proj.norm(dim=-1)
    cosine = (grad_t * proj).sum(dim=-1) / (grad_norm * proj_norm.clamp_min(1e-9))
    return {
        "mean_grad_norm_retention": float((proj_norm / grad_norm).mean().item()),
        "median_grad_norm_retention": float((proj_norm / grad_norm).median().item()),
        "mean_grad_cosine": float(cosine.mean().item()),
        "median_grad_cosine": float(cosine.median().item()),
    }


def learn_race_basis(
    base_features: np.ndarray,
    cf_features: np.ndarray,
    gradients: np.ndarray,
    rank: int,
    grad_rank: int,
) -> torch.Tensor:
    delta = torch.from_numpy(cf_features - base_features).float()
    delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    grad_t = torch.from_numpy(gradients).float()
    grad_t = grad_t / grad_t.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    _, _, grad_vh = torch.linalg.svd(grad_t, full_matrices=False)
    grad_rank = min(grad_rank, grad_vh.size(0))
    grad_basis = grad_vh[:grad_rank].T.contiguous()

    aligned_delta = delta @ grad_basis @ grad_basis.T
    aligned_norm = aligned_delta.norm(dim=-1, keepdim=True)
    keep_mask = aligned_norm.squeeze(-1) > 1e-6

    if not bool(keep_mask.any()):
        return grad_basis[:, : min(rank, grad_basis.size(1))].contiguous()

    aligned_delta = aligned_delta[keep_mask] / aligned_norm[keep_mask].clamp_min(1e-6)
    _, _, vh = torch.linalg.svd(aligned_delta, full_matrices=False)
    rank = min(rank, vh.size(0))
    return vh[:rank].T.contiguous()


def run_task(
    model,
    tokenizer,
    task: str,
    device: torch.device,
    layer_idx: int,
    hdmi_alpha: float,
    hdmi_inner_steps: int,
):
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
    G_train = build_margin_gradients(model, tokenizer, train_samples, device, layer_idx)

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
    best_diag = None
    for rank in [2, 4, 8, 16, 32]:
        for grad_rank in [4, 8, 16, 32, 64]:
            basis = learn_race_basis(X_train[tr_i], X_cf_train[tr_i], G_train[tr_i], rank, grad_rank)
            diag = compute_basis_diagnostics(basis, G_train[te_i])
            for mix in [0.5, 0.75, 1.0]:
                metrics = cmp.evaluate_hybrid(
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
                    best_cfg = {"rank": rank, "grad_rank": grad_rank, "mix": mix}
                    best_metrics = metrics
                    best_diag = diag

    assert best_cfg is not None and best_metrics is not None and best_diag is not None

    basis = learn_race_basis(
        base_features=X_train,
        cf_features=X_cf_train,
        gradients=G_train,
        rank=best_cfg["rank"],
        grad_rank=best_cfg["grad_rank"],
    )
    race_metrics = cmp.evaluate_hybrid(
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
    race_diag = compute_basis_diagnostics(basis, build_margin_gradients(model, tokenizer, test_samples, device, layer_idx))

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
        "tuning_basis_diagnostics": best_diag,
        "hdmi_local": {
            "baseline_task_acc": hdmi_metrics.baseline_task_acc,
            "after_task_acc": hdmi_metrics.after_task_acc,
            "delta_task_acc": hdmi_metrics.delta_task_acc,
            "completeness": hdmi_metrics.completeness,
            "selectivity": hdmi_metrics.selectivity,
            "reliability": hdmi_metrics.reliability,
            "n": hdmi_metrics.n,
        },
        "race_local": race_metrics,
        "race_test_basis_diagnostics": race_diag,
        "hdmi_reported": cmp.HDMI_REPORTED.get(task),
    }


def mean_for(results, key: str) -> Dict[str, float]:
    valid_tasks = [task_result for task_result in results["tasks"] if task_result.get(key) is not None]
    if not valid_tasks:
        return {}
    return {
        metric: float(np.mean([task_result[key][metric] for task_result in valid_tasks]))
        for metric in ["completeness", "selectivity", "reliability"]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="EleutherAI/pythia-70m")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--layer_idx", type=int, default=-1)
    parser.add_argument("--hdmi_alpha", type=float, default=1.0)
    parser.add_argument("--hdmi_inner_steps", type=int, default=30)
    parser.add_argument("--output", default="compare_hdmi_race_exact.json")
    args = parser.parse_args()

    set_seed(1337)
    cmp.ensure_local_json(args.tasks)

    device = hdmi.get_device("auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device).eval()
    model.config.output_hidden_states = True

    results = {
        "method": "RACE-delta-gradient",
        "model_name": args.model_name,
        "layer_idx": args.layer_idx,
        "hdmi_alpha": args.hdmi_alpha,
        "hdmi_inner_steps": args.hdmi_inner_steps,
        "tasks": [],
    }

    for task in args.tasks:
        print(f"Running RACE comparison for {task}")
        task_result = run_task(
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

    results["means"] = {
        "hdmi_reported": mean_for(results, "hdmi_reported"),
        "hdmi_local": mean_for(results, "hdmi_local"),
        "race_local": mean_for(results, "race_local"),
    }
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["means"], indent=2))


if __name__ == "__main__":
    main()
