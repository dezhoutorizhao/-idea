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
RACE_PATH = ROOT / "compare_hdmi_race_exact.py"
MCTR_PATH = ROOT / "compare_hdmi_mctr_failed_tasks.py"
MA_BCCI_PATH = ROOT / "compare_hdmi_bcci_metric_aligned_failed_tasks.py"
RACE_RESULTS_PATH = ROOT / "compare_hdmi_race_all.json"
MCTR_RESULTS_PATH = ROOT / "compare_hdmi_mctr_failed_tasks.json"
BCCI_RESULTS_PATH = ROOT / "compare_hdmi_bcci_failed_tasks.json"
MA_BCCI_RESULTS_PATH = ROOT / "compare_hdmi_bcci_metric_aligned_failed_tasks.json"

DEFAULT_TASKS = ["garden_mvrr_mod", "gss_subord_pp"]
DEFAULT_GAMMA_GRID = [0.1, 0.5]
DEFAULT_TRUST_BETA_GRID = [0.05, 0.1]
DEFAULT_RANK_GRID = [2, 4, 8]
DEFAULT_GE_WEIGHT_GRID = [0.1, 1.0]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


race = load_module(RACE_PATH, "race_mod")
mctr = load_module(MCTR_PATH, "mctr_mod")
ma_bcci = load_module(MA_BCCI_PATH, "ma_bcci_mod")
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


def build_zc_gradients(model, tokenizer, samples, device: torch.device, layer_idx: int, vZc) -> np.ndarray:
    grads = []
    for sample in samples:
        prefix_ids = hdmi.encode(tokenizer, sample.prompt_text, device)
        h0 = hdmi.get_prompt_hidden_at_layer(model, prefix_ids, layer_idx=layer_idx)
        with torch.enable_grad():
            h_var = h0.detach().clone().requires_grad_(True)
            logits = vZc(h_var)
            objective = torch.log_softmax(logits, dim=-1)[0, int(sample.cf_zc)]
            grad_h = torch.autograd.grad(objective, h_var, retain_graph=False, create_graph=False)[0]
        grads.append(grad_h[0].float().detach().cpu().numpy())
    return np.stack(grads, axis=0)


def build_ze_gradients(model, tokenizer, samples, device: torch.device, layer_idx: int, vZe) -> np.ndarray:
    grads = []
    for sample in samples:
        if getattr(sample, "ze", -1) is None or int(sample.ze) < 0:
            continue
        prefix_ids = hdmi.encode(tokenizer, sample.prompt_text, device)
        h0 = hdmi.get_prompt_hidden_at_layer(model, prefix_ids, layer_idx=layer_idx)
        with torch.enable_grad():
            h_var = h0.detach().clone().requires_grad_(True)
            logits = vZe(h_var)
            objective = torch.log_softmax(logits, dim=-1)[0, int(sample.ze)]
            grad_h = torch.autograd.grad(objective, h_var, retain_graph=False, create_graph=False)[0]
        grads.append(grad_h[0].float().detach().cpu().numpy())
    if not grads:
        hidden = hdmi.get_prompt_hidden_at_layer(model, hdmi.encode(tokenizer, samples[0].prompt_text, device), layer_idx=layer_idx).shape[-1]
        return np.zeros((1, int(hidden)), dtype=np.float32)
    return np.stack(grads, axis=0)


def learn_dpas_basis(
    behavior_grads: np.ndarray,
    zc_grads: np.ndarray,
    ze_grads: np.ndarray,
    rank: int,
    ge_weight: float,
) -> torch.Tensor:
    gb = torch.from_numpy(behavior_grads).float()
    gc = torch.from_numpy(zc_grads).float()
    ge = torch.from_numpy(ze_grads).float()

    gb = gb / gb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    gc = gc / gc.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    ge = ge / ge.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    cross_cov = (gb.T @ gc + gc.T @ gb) / max(1, gb.shape[0])
    ze_cov = (ge.T @ ge) / max(1, ge.shape[0])
    score_mat = cross_cov - float(ge_weight) * ze_cov
    eigvals, eigvecs = torch.linalg.eigh(score_mat)
    order = torch.argsort(eigvals, descending=True)
    eigvecs = eigvecs[:, order]
    rank = min(rank, eigvecs.shape[1])
    return eigvecs[:, :rank].contiguous()


def run_task(
    model,
    tokenizer,
    task: str,
    device: torch.device,
    layer_idx: int,
    hdmi_alpha: float,
    hdmi_inner_steps: int,
    gamma_grid: Sequence[float],
    trust_beta_grid: Sequence[float],
    rank_grid: Sequence[int],
    ge_weight_grid: Sequence[float],
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
    behavior_grads = race.build_margin_gradients(model, tokenizer, train_samples, device, layer_idx)
    zc_grads = build_zc_gradients(model, tokenizer, train_samples, device, layer_idx, vZc)
    ze_grads = build_ze_gradients(model, tokenizer, train_samples, device, layer_idx, vZe) if vZe is not None else np.zeros_like(behavior_grads[:1])

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
    best_basis_diag = None
    for rank in rank_grid:
        for ge_weight in ge_weight_grid:
            basis = learn_dpas_basis(
                behavior_grads=behavior_grads[tr_i],
                zc_grads=zc_grads[tr_i],
                ze_grads=ze_grads,
                rank=int(rank),
                ge_weight=float(ge_weight),
            )
            for gamma in gamma_grid:
                for trust_beta in trust_beta_grid:
                    metrics = ma_bcci.evaluate_metric_aligned_bcci(
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
                        trust_beta=float(trust_beta),
                        gamma=float(gamma),
                        sel_weight=1.0,
                        anchor_weight=0.05,
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
                            "rank": int(rank),
                            "ge_weight": float(ge_weight),
                            "gamma": float(gamma),
                            "trust_beta": float(trust_beta),
                        }
                        best_metrics = metrics
                        best_basis_diag = race.compute_basis_diagnostics(basis, behavior_grads[te_i])
                    else:
                        current_score = (
                            best_metrics["reliability"],
                            best_metrics["selectivity"],
                            best_metrics["completeness"],
                            best_metrics["margin_reserve_rate"],
                        )
                        if score > current_score:
                            best_cfg = {
                                "rank": int(rank),
                                "ge_weight": float(ge_weight),
                                "gamma": float(gamma),
                                "trust_beta": float(trust_beta),
                            }
                            best_metrics = metrics
                            best_basis_diag = race.compute_basis_diagnostics(basis, behavior_grads[te_i])

    improved_test = None
    if best_cfg is not None:
        final_basis = learn_dpas_basis(
            behavior_grads=behavior_grads,
            zc_grads=zc_grads,
            ze_grads=ze_grads,
            rank=int(best_cfg["rank"]),
            ge_weight=float(best_cfg["ge_weight"]),
        )
        improved_test = ma_bcci.evaluate_metric_aligned_bcci(
            model=model,
            tokenizer=tokenizer,
            samples=test_samples,
            vZc=vZc,
            vZe=vZe,
            basis=final_basis,
            mix=1.0,
            device=device,
            layer_idx=layer_idx,
            alpha=hdmi_alpha,
            inner_steps=hdmi_inner_steps,
            train_base=X_train,
            train_cf=X_cf_train,
            train_samples=train_samples,
            k_neighbors=16,
            trust_beta=float(best_cfg["trust_beta"]),
            gamma=float(best_cfg["gamma"]),
            sel_weight=1.0,
            anchor_weight=0.05,
        )

    return {
        "task": task,
        "probe_metrics": {
            "interventional_zc_probe_acc": interventional_acc,
            "validation_zc_probe_acc": vzc_acc,
            "validation_ze_probe_acc": vze_acc,
        },
        "search_grid": {
            "rank": list(rank_grid),
            "ge_weight": list(ge_weight_grid),
            "gamma": list(gamma_grid),
            "trust_beta": list(trust_beta_grid),
            "sel_weight": 1.0,
            "anchor_weight": 0.05,
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
        "selected_dpas_config": best_cfg,
        "selected_dpas_tune_metrics": best_metrics,
        "selected_dpas_basis_diag": best_basis_diag,
        "dpas_metric_bcci_test": improved_test,
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
    parser.add_argument("--rank_grid", nargs="+", type=int, default=DEFAULT_RANK_GRID)
    parser.add_argument("--ge_weight_grid", nargs="+", type=float, default=DEFAULT_GE_WEIGHT_GRID)
    parser.add_argument("--output", default="compare_hdmi_dpas_metric_bcci_failed_tasks.json")
    args = parser.parse_args()

    set_seed(1337)
    cmp.ensure_local_json(args.tasks)

    mctr_results_map = load_json_map(MCTR_RESULTS_PATH)
    bcci_results_map = load_json_map(BCCI_RESULTS_PATH)
    ma_bcci_results_map = load_json_map(MA_BCCI_RESULTS_PATH)

    device = hdmi.get_device("auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device).eval()
    model.config.output_hidden_states = True

    results = {
        "method": "DPAS-plus-metric-aligned-BCCI",
        "tasks": [],
        "model_name": args.model_name,
        "layer_idx": args.layer_idx,
        "hdmi_alpha": args.hdmi_alpha,
        "hdmi_inner_steps": args.hdmi_inner_steps,
    }

    for task in args.tasks:
        print(f"Running DPAS+metric-BCCI on {task}")
        task_result = run_task(
            model=model,
            tokenizer=tokenizer,
            task=task,
            device=device,
            layer_idx=args.layer_idx,
            hdmi_alpha=args.hdmi_alpha,
            hdmi_inner_steps=args.hdmi_inner_steps,
            gamma_grid=args.gamma_grid,
            trust_beta_grid=args.trust_beta_grid,
            rank_grid=args.rank_grid,
            ge_weight_grid=args.ge_weight_grid,
        )
        if task in mctr_results_map:
            task_result["mctr_reference_test"] = mctr_results_map[task].get("mctr_test")
        if task in bcci_results_map:
            task_result["bcci_reference_test"] = bcci_results_map[task].get("bcci_test")
        if task in ma_bcci_results_map:
            task_result["metric_bcci_reference_test"] = ma_bcci_results_map[task].get("metric_aligned_test")
        results["tasks"].append(task_result)
        print(json.dumps(task_result, indent=2))

    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
