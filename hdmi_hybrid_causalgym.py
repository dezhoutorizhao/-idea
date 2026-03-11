import argparse
import importlib.util
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
HDMI_PROBING = ROOT / "hdmi_supplementary" / "hdmi-code" / "probing.py"
TASKS = ["agr_sv_num_pp", "gss_subord_pp", "filler_gap_pp"]
HDMI_REPORTED = {
    "agr_sv_num_pp": {"completeness": 0.9984, "selectivity": 0.9409, "reliability": 0.9688},
    "gss_subord_pp": {"completeness": 0.9984, "selectivity": 0.7371, "reliability": 0.8481},
    "filler_gap_pp": {"completeness": 0.7404, "selectivity": 0.6275, "reliability": 0.6793},
}


def load_hdmi_module():
    spec = importlib.util.spec_from_file_location("hdmi_probe", HDMI_PROBING)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@dataclass
class HybridSample:
    prompt_text: str
    cf_prompt_text: str
    gold_cont: str
    alt_cont: str
    zc: int
    cf_zc: int
    ze: Optional[int]
    task: str


@dataclass
class Metrics:
    baseline_task_acc: float
    after_task_acc: float
    delta_task_acc: float
    completeness: float
    selectivity: float
    reliability: float
    n: int


def build_text(parts: Sequence[str]) -> str:
    return "".join(parts)


def make_samples_for_task(dataset, task: str) -> Tuple[List[HybridSample], List[HybridSample], List[HybridSample]]:
    train_rows = [row for row in dataset["train"] if row["task"] == task]
    val_rows = [row for row in dataset["validation"] if row["task"] == task]
    test_rows = [row for row in dataset["test"] if row["task"] == task]

    types = sorted({row["base_type"] for row in train_rows + val_rows + test_rows} | {row["src_type"] for row in train_rows + val_rows + test_rows})
    zc_map = {label: i for i, label in enumerate(types)}

    def convert(rows: List[dict]) -> List[HybridSample]:
        samples = []
        for row in rows:
            text = build_text(row["base"])
            cf_text = build_text(row["src"])
            samples.append(
                HybridSample(
                    prompt_text=text,
                    cf_prompt_text=cf_text,
                    gold_cont=str(row["base_label"]),
                    alt_cont=str(row["src_label"]),
                    zc=zc_map[row["base_type"]],
                    cf_zc=zc_map[row["src_type"]],
                    ze=hdmi.ze_prep_family(text, window=12),
                    task=task,
                )
            )
        return samples

    return convert(train_rows), convert(val_rows), convert(test_rows)


@torch.no_grad()
def build_features_for_prompts(hdmi, model, tokenizer, texts: Sequence[str], device: torch.device, layer_idx: int) -> np.ndarray:
    feats = []
    for text in texts:
        ids = hdmi.encode(tokenizer, text, device)
        h = hdmi.get_prompt_hidden_at_layer(model, ids, layer_idx=layer_idx)
        feats.append(h[0].float().cpu().numpy())
    return np.stack(feats, axis=0)


def learn_semantic_subspace(base_features: np.ndarray, cf_features: np.ndarray, rank: int) -> torch.Tensor:
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
    source_token_id: Optional[int],
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
    h_work = h.detach().to(dtype=head_dtype)
    basis = basis.to(device=h.device, dtype=h_work.dtype)
    eps = 1e-9
    for _ in range(max(1, int(inner_steps))):
        with torch.enable_grad():
            h_var = h_work.clone().detach().requires_grad_(True)
            z = head(h_var)
            if use_margin and source_token_id is not None and source_token_id >= 0:
                objective = z[0, int(target_token_id)] - z[0, int(source_token_id)]
            else:
                objective = z[0, int(target_token_id)]
            grad_h = torch.autograd.grad(objective, h_var, retain_graph=False, create_graph=False)[0]
        projected = grad_h @ basis @ basis.T
        grad_h = (1.0 - mix) * grad_h + mix * projected
        if grad_clip_norm and grad_clip_norm > 0.0:
            n = grad_h.norm(p=2, dim=-1, keepdim=True)
            grad_h = torch.where(n > grad_clip_norm, grad_h * (grad_clip_norm / (n + eps)), grad_h)
        if normalize_grad:
            grad_h = grad_h / grad_h.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        with torch.no_grad():
            h_work = (h_var + alpha * grad_h).detach()
    return h_work.to(dtype=h.dtype)


def evaluate_method(
    hdmi,
    model,
    tokenizer,
    samples: List[HybridSample],
    probe_c_interv,
    vZc,
    vZe,
    method: str,
    device: torch.device,
    layer_idx: int,
    hdmi_alpha: float,
    hdmi_inner_steps: int,
    hdmi_use_margin: bool,
    hdmi_normalize_grad: bool,
    hdmi_grad_clip_norm: float,
    basis: Optional[torch.Tensor] = None,
    mix: float = 1.0,
) -> Metrics:
    model.eval()
    probe_c_interv.eval()
    vZc.eval()
    if vZe is not None:
        vZe.eval()

    correct_base = 0
    correct_after = 0
    comp_vals: List[float] = []
    sel_vals: List[float] = []

    with torch.no_grad():
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
                p_ze_before = F.softmax(vZe(h0), dim=-1).cpu().numpy()[0]
            else:
                p_ze_before = None

            token_gold = int(base_gold["first_token_id"].item())
            token_alt = int(base_alt["first_token_id"].item())
            if method == "hdmi":
                h1 = hdmi.hdmi_intervention(
                    model=model,
                    h=h0,
                    target_token_id=token_alt,
                    source_token_id=token_gold,
                    alpha=hdmi_alpha,
                    inner_steps=hdmi_inner_steps,
                    use_margin=hdmi_use_margin,
                    normalize_grad=hdmi_normalize_grad,
                    grad_clip_norm=hdmi_grad_clip_norm,
                )
            elif method == "hybrid":
                assert basis is not None
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
            else:
                raise ValueError(method)

            if token_gold < 0 or token_alt < 0:
                lp_gold1 = lp_gold0
                lp_alt1 = lp_alt0
            else:
                lp1_gold_first = hdmi.compute_first_step_logprob_from_h(model, h1, torch.tensor(token_gold, device=device)).item()
                lp1_alt_first = hdmi.compute_first_step_logprob_from_h(model, h1, torch.tensor(token_alt, device=device)).item()
                lp_gold1 = lp_gold0 - float(base_gold["first_lp"].item()) + lp1_gold_first
                lp_alt1 = lp_alt0 - float(base_alt["first_lp"].item()) + lp1_alt_first
            if hdmi.choose_candidate_by_logprob(lp_gold1, lp_alt1) == 0:
                correct_after += 1

            p_c_after = F.softmax(vZc(h1), dim=-1).cpu().numpy()[0]
            goal = np.zeros_like(p_c_after)
            goal[int(sample.cf_zc)] = 1.0
            comp_vals.append(1.0 - hdmi.tv_distance(p_c_after, goal))

            if vZe is not None and p_ze_before is not None:
                p_ze_after = F.softmax(vZe(h1), dim=-1).cpu().numpy()[0]
                m = max(1.0 - float(np.min(p_ze_before)), float(np.max(p_ze_before)))
                sel_vals.append(1.0 - (hdmi.tv_distance(p_ze_after, p_ze_before) / m if m > 0 else 0.0))
            else:
                sel_vals.append(1.0)

    n = len(samples)
    baseline_acc = correct_base / n
    after_acc = correct_after / n
    completeness = float(np.mean(comp_vals))
    selectivity = float(np.mean(sel_vals))
    return Metrics(
        baseline_task_acc=baseline_acc,
        after_task_acc=after_acc,
        delta_task_acc=after_acc - baseline_acc,
        completeness=completeness,
        selectivity=selectivity,
        reliability=hdmi.harmonic_mean(completeness, selectivity),
        n=n,
    )


def run_task(
    hdmi,
    model,
    tokenizer,
    dataset,
    task: str,
    device: torch.device,
    layer_idx: int,
    hdmi_alpha: float,
    hdmi_inner_steps: int,
    hdmi_use_margin: bool,
    hdmi_normalize_grad: bool,
    hdmi_grad_clip_norm: float,
) -> Dict[str, object]:
    train_samples, val_samples, test_samples = make_samples_for_task(dataset, task)
    X_train = hdmi.build_features(model, tokenizer, train_samples, device, layer_idx=layer_idx)
    X_train_cf = build_features_for_prompts(hdmi, model, tokenizer, [s.cf_prompt_text for s in train_samples], device, layer_idx)
    X_val = hdmi.build_features(model, tokenizer, val_samples, device, layer_idx=layer_idx)

    y_train_zc = np.array([s.zc for s in train_samples], dtype=np.int64)
    y_val_zc = np.array([s.zc for s in val_samples], dtype=np.int64)
    y_train_ze = np.array([s.ze for s in train_samples], dtype=np.int64)
    y_val_ze = np.array([s.ze for s in val_samples], dtype=np.int64)

    rng = np.random.default_rng(43)
    perm = rng.permutation(len(train_samples))
    cut = int(0.8 * len(perm))
    tr_i, tune_i = perm[:cut], perm[cut:]

    probe_c_interv = hdmi.train_probe(
        X_train[tr_i],
        y_train_zc[tr_i],
        epochs=100,
        lr=1e-2,
        wb=1e-6,
        device=device,
        hidden=None,
        batch_size=256,
        verbose=False,
    )
    interventional_probe_acc = hdmi.eval_probe_acc(probe_c_interv, X_train[tune_i], y_train_zc[tune_i], device=device)

    rng2 = np.random.default_rng(45)
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
        epochs=75,
        lr=1e-2,
        wd=1e-6,
        batch_size=256,
    )
    vZc_acc = hdmi.eval_probe_acc(vZc, X_val[va_v_idx], y_val_zc[va_v_idx], device=device)

    vZe = hdmi.train_validation_probe_with_selection(
        X_train,
        y_train_ze,
        X_val,
        y_val_ze,
        hidden_grid=(0, 64, 256, 512),
        device=device,
        epochs=75,
        lr=1e-2,
        wd=1e-6,
        batch_size=256,
    )
    vZe_acc = hdmi.eval_probe_acc(vZe, X_val, y_val_ze, device=device)

    hdmi_metrics = evaluate_method(
        hdmi=hdmi,
        model=model,
        tokenizer=tokenizer,
        samples=test_samples,
        probe_c_interv=probe_c_interv,
        vZc=vZc,
        vZe=vZe,
        method="hdmi",
        device=device,
        layer_idx=layer_idx,
        hdmi_alpha=hdmi_alpha,
        hdmi_inner_steps=hdmi_inner_steps,
        hdmi_use_margin=hdmi_use_margin,
        hdmi_normalize_grad=hdmi_normalize_grad,
        hdmi_grad_clip_norm=hdmi_grad_clip_norm,
    )

    tune_samples = [train_samples[i] for i in tune_i]
    best_cfg = None
    best_metric = None
    for rank in [1, 2, 4, 8, 16, 32]:
        basis = learn_semantic_subspace(X_train[tr_i], X_train_cf[tr_i], rank)
        for mix in [0.25, 0.5, 0.75, 1.0]:
            metrics = evaluate_method(
                hdmi=hdmi,
                model=model,
                tokenizer=tokenizer,
                samples=tune_samples,
                probe_c_interv=probe_c_interv,
                vZc=vZc,
                vZe=vZe,
                method="hybrid",
                device=device,
                layer_idx=layer_idx,
                hdmi_alpha=hdmi_alpha,
                hdmi_inner_steps=hdmi_inner_steps,
                hdmi_use_margin=hdmi_use_margin,
                hdmi_normalize_grad=hdmi_normalize_grad,
                hdmi_grad_clip_norm=hdmi_grad_clip_norm,
                basis=basis,
                mix=mix,
            )
            if best_metric is None or metrics.reliability > best_metric.reliability:
                best_metric = metrics
                best_cfg = {"rank": rank, "mix": mix}

    assert best_cfg is not None and best_metric is not None
    basis = learn_semantic_subspace(X_train, X_train_cf, best_cfg["rank"])
    hybrid_metrics = evaluate_method(
        hdmi=hdmi,
        model=model,
        tokenizer=tokenizer,
        samples=test_samples,
        probe_c_interv=probe_c_interv,
        vZc=vZc,
        vZe=vZe,
        method="hybrid",
        device=device,
        layer_idx=layer_idx,
        hdmi_alpha=hdmi_alpha,
        hdmi_inner_steps=hdmi_inner_steps,
        hdmi_use_margin=hdmi_use_margin,
        hdmi_normalize_grad=hdmi_normalize_grad,
        hdmi_grad_clip_norm=hdmi_grad_clip_norm,
        basis=basis,
        mix=best_cfg["mix"],
    )

    return {
        "task": task,
        "split_sizes": {"train": len(train_samples), "validation": len(val_samples), "test": len(test_samples)},
        "probe_metrics": {
            "interventional_zc_probe_acc": interventional_probe_acc,
            "validation_zc_probe_acc": vZc_acc,
            "validation_ze_probe_acc": vZe_acc,
        },
        "best_tuning_config": best_cfg,
        "best_tuning_metrics": best_metric.__dict__,
        "hdmi_local": hdmi_metrics.__dict__,
        "hybrid_local": hybrid_metrics.__dict__,
        "hdmi_reported": HDMI_REPORTED[task],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="EleutherAI/pythia-70m")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--layer_idx", type=int, default=-1)
    parser.add_argument("--hdmi_alpha", type=float, default=1.0)
    parser.add_argument("--hdmi_inner_steps", type=int, default=30)
    parser.add_argument("--output", default="hdmi_hybrid_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device).eval()
    dataset = load_dataset("aryaman/causalgym")

    results = {
        "model_name": args.model_name,
        "layer_idx": args.layer_idx,
        "hdmi_alpha": args.hdmi_alpha,
        "hdmi_inner_steps": args.hdmi_inner_steps,
        "tasks": [],
    }
    for task in args.tasks:
        print(f"Running {task} ...")
        task_result = run_task(
            hdmi=hdmi,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            task=task,
            device=device,
            layer_idx=args.layer_idx,
            hdmi_alpha=args.hdmi_alpha,
            hdmi_inner_steps=args.hdmi_inner_steps,
            hdmi_use_margin=True,
            hdmi_normalize_grad=False,
            hdmi_grad_clip_norm=0.0,
        )
        results["tasks"].append(task_result)
        print(json.dumps(task_result, indent=2))

    def mean_for(key: str):
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


hdmi = load_hdmi_module()


if __name__ == "__main__":
    main()
