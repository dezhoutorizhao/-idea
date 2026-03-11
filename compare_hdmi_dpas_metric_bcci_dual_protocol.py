import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
BASE_DPAS_PATH = ROOT / "compare_hdmi_dpas_metric_bcci_failed_tasks.py"

ALL_TASKS = [
    "agr_gender",
    "agr_refl_num_obj-relc",
    "agr_refl_num_pp",
    "agr_refl_num_subj-relc",
    "agr_sv_num_obj-relc",
    "agr_sv_num_pp",
    "agr_sv_num_subj-relc",
    "cleft",
    "cleft_mod",
    "filler_gap_embed_3",
    "filler_gap_embed_4",
    "filler_gap_hierarchy",
    "filler_gap_obj",
    "filler_gap_pp",
    "filler_gap_subj",
    "garden_mvrr",
    "garden_mvrr_mod",
    "garden_npz_obj",
    "garden_npz_obj_mod",
    "garden_npz_v-trans",
    "garden_npz_v-trans_mod",
    "gss_subord",
    "gss_subord_obj-relc",
    "gss_subord_pp",
    "gss_subord_subj-relc",
    "npi_any_obj-relc",
    "npi_any_subj-relc",
    "npi_ever_obj-relc",
    "npi_ever_subj-relc",
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


dpas = load_module(BASE_DPAS_PATH, "dpas_base_mod")
cmp = dpas.cmp
hdmi = dpas.hdmi
ma_bcci = dpas.ma_bcci
race = dpas.race


def metrics_to_dict(metrics) -> Dict[str, float]:
    if metrics is None:
        return None
    if isinstance(metrics, dict):
        result = {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v) for k, v in metrics.items()}
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


def config_key(cfg: Optional[Dict[str, float]]) -> Optional[Tuple[Tuple[str, float], ...]]:
    if cfg is None:
        return None
    return tuple(sorted((str(k), float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v) for k, v in cfg.items()))


def score_tuple(mode: str, metrics: Dict[str, float]) -> Tuple[float, ...]:
    reserve = float(metrics.get("margin_reserve_rate", -1.0))
    after = float(metrics["after_task_acc"])
    if mode == "strict":
        return (
            float(metrics["reliability"]),
            float(metrics["selectivity"]),
            float(metrics["completeness"]),
            reserve,
            -after,
        )
    if mode == "reliability":
        return (
            float(metrics["reliability"]),
            -after,
            float(metrics["selectivity"]),
            float(metrics["completeness"]),
            reserve,
        )
    if mode == "completeness":
        return (
            float(metrics["completeness"]),
            float(metrics["reliability"]),
            float(metrics["selectivity"]),
            -after,
            reserve,
        )
    if mode == "selectivity":
        return (
            float(metrics["selectivity"]),
            float(metrics["reliability"]),
            float(metrics["completeness"]),
            -after,
            reserve,
        )
    if mode == "after":
        return (
            -after,
            float(metrics["reliability"]),
            float(metrics["selectivity"]),
            float(metrics["completeness"]),
            reserve,
        )
    raise ValueError(f"Unknown mode: {mode}")


def evaluate_cfg_on_test(
    *,
    model,
    tokenizer,
    test_samples,
    vZc,
    vZe,
    behavior_grads: np.ndarray,
    zc_grads: np.ndarray,
    ze_grads: np.ndarray,
    X_train: np.ndarray,
    X_cf_train: np.ndarray,
    train_samples,
    layer_idx: int,
    hdmi_alpha: float,
    hdmi_inner_steps: int,
    cfg: Dict[str, float],
) -> Dict[str, float]:
    basis = dpas.learn_dpas_basis(
        behavior_grads=behavior_grads,
        zc_grads=zc_grads,
        ze_grads=ze_grads,
        rank=int(cfg["rank"]),
        ge_weight=float(cfg["ge_weight"]),
    )
    metrics = ma_bcci.evaluate_metric_aligned_bcci(
        model=model,
        tokenizer=tokenizer,
        samples=test_samples,
        vZc=vZc,
        vZe=vZe,
        basis=basis,
        mix=1.0,
        device=next(model.parameters()).device,
        layer_idx=layer_idx,
        alpha=hdmi_alpha,
        inner_steps=hdmi_inner_steps,
        train_base=X_train,
        train_cf=X_cf_train,
        train_samples=train_samples,
        k_neighbors=16,
        trust_beta=float(cfg["trust_beta"]),
        gamma=float(cfg["gamma"]),
        sel_weight=1.0,
        anchor_weight=0.05,
    )
    metrics = metrics_to_dict(metrics)
    basis_diag = race.compute_basis_diagnostics(basis, behavior_grads)
    return {"config": cfg, "test_metrics": metrics, "basis_diag": basis_diag}


def run_task(
    *,
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
    dpas.set_seed(1337)
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
    hdmi_test = metrics_to_dict(
        hdmi.evaluate(
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
    )

    candidate_records: List[Dict] = []
    best_strict = None
    best_r = None
    best_c = None
    best_s = None
    best_after = None

    for rank in rank_grid:
        for ge_weight in ge_weight_grid:
            basis = dpas.learn_dpas_basis(
                behavior_grads=behavior_grads[tr_i],
                zc_grads=zc_grads[tr_i],
                ze_grads=ze_grads,
                rank=int(rank),
                ge_weight=float(ge_weight),
            )
            basis_diag = race.compute_basis_diagnostics(basis, behavior_grads[te_i])
            for gamma in gamma_grid:
                for trust_beta in trust_beta_grid:
                    cfg = {
                        "rank": int(rank),
                        "ge_weight": float(ge_weight),
                        "gamma": float(gamma),
                        "trust_beta": float(trust_beta),
                    }
                    metrics = metrics_to_dict(
                        ma_bcci.evaluate_metric_aligned_bcci(
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
                    )
                    passes_after = metrics["after_task_acc"] <= 0.0
                    passes_c = metrics["completeness"] + 1e-6 >= hdmi_tune["completeness"]
                    passes_s = metrics["selectivity"] + 1e-6 >= hdmi_tune["selectivity"]
                    strict_feasible = passes_after and passes_c and passes_s

                    record = {
                        "config": cfg,
                        "tune_metrics": metrics,
                        "basis_diag": basis_diag,
                        "passes_after_constraint": passes_after,
                        "passes_completeness_constraint": passes_c,
                        "passes_selectivity_constraint": passes_s,
                        "strict_feasible": strict_feasible,
                    }
                    candidate_records.append(record)

                    if strict_feasible:
                        if best_strict is None or score_tuple("strict", metrics) > score_tuple("strict", best_strict["tune_metrics"]):
                            best_strict = record
                    if best_r is None or score_tuple("reliability", metrics) > score_tuple("reliability", best_r["tune_metrics"]):
                        best_r = record
                    if best_c is None or score_tuple("completeness", metrics) > score_tuple("completeness", best_c["tune_metrics"]):
                        best_c = record
                    if best_s is None or score_tuple("selectivity", metrics) > score_tuple("selectivity", best_s["tune_metrics"]):
                        best_s = record
                    if best_after is None or score_tuple("after", metrics) > score_tuple("after", best_after["tune_metrics"]):
                        best_after = record

    chosen = {
        "strict_feasible_best": best_strict,
        "reliability_max": best_r,
        "completeness_max": best_c,
        "selectivity_max": best_s,
        "lowest_after_task_acc": best_after,
    }

    evaluated = {}
    for name, record in chosen.items():
        if record is None:
            evaluated[name] = None
            continue
        key = config_key(record["config"])
        if key not in evaluated:
            evaluated[key] = evaluate_cfg_on_test(
                model=model,
                tokenizer=tokenizer,
                test_samples=test_samples,
                vZc=vZc,
                vZe=vZe,
                behavior_grads=behavior_grads,
                zc_grads=zc_grads,
                ze_grads=ze_grads,
                X_train=X_train,
                X_cf_train=X_cf_train,
                train_samples=train_samples,
                layer_idx=layer_idx,
                hdmi_alpha=hdmi_alpha,
                hdmi_inner_steps=hdmi_inner_steps,
                cfg=record["config"],
            )
        evaluated[name] = evaluated[key]

    candidate_summary = {
        "total_candidates": len(candidate_records),
        "passes_after_count": int(sum(record["passes_after_constraint"] for record in candidate_records)),
        "passes_completeness_count": int(sum(record["passes_completeness_constraint"] for record in candidate_records)),
        "passes_selectivity_count": int(sum(record["passes_selectivity_constraint"] for record in candidate_records)),
        "passes_cs_count": int(
            sum(
                record["passes_completeness_constraint"] and record["passes_selectivity_constraint"]
                for record in candidate_records
            )
        ),
        "strict_feasible_count": int(sum(record["strict_feasible"] for record in candidate_records)),
        "best_tune_reliability": None if best_r is None else float(best_r["tune_metrics"]["reliability"]),
        "best_tune_completeness": None if best_c is None else float(best_c["tune_metrics"]["completeness"]),
        "best_tune_selectivity": None if best_s is None else float(best_s["tune_metrics"]["selectivity"]),
        "lowest_tune_after_task_acc": None if best_after is None else float(best_after["tune_metrics"]["after_task_acc"]),
    }

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
        "hdmi_tune": hdmi_tune,
        "hdmi_test": hdmi_test,
        "candidate_summary": candidate_summary,
        "candidate_records": candidate_records,
        "strict_protocol": None
        if best_strict is None
        else {
            "selected_config": best_strict["config"],
            "tune_metrics": best_strict["tune_metrics"],
            "basis_diag": best_strict["basis_diag"],
            "test_metrics": evaluated["strict_feasible_best"]["test_metrics"],
            "test_basis_diag": evaluated["strict_feasible_best"]["basis_diag"],
        },
        "reliability_max_protocol": None
        if best_r is None
        else {
            "selected_config": best_r["config"],
            "tune_metrics": best_r["tune_metrics"],
            "basis_diag": best_r["basis_diag"],
            "test_metrics": evaluated["reliability_max"]["test_metrics"],
            "test_basis_diag": evaluated["reliability_max"]["basis_diag"],
        },
        "completeness_max_protocol": None
        if best_c is None
        else {
            "selected_config": best_c["config"],
            "tune_metrics": best_c["tune_metrics"],
            "basis_diag": best_c["basis_diag"],
            "test_metrics": evaluated["completeness_max"]["test_metrics"],
            "test_basis_diag": evaluated["completeness_max"]["basis_diag"],
        },
        "selectivity_max_protocol": None
        if best_s is None
        else {
            "selected_config": best_s["config"],
            "tune_metrics": best_s["tune_metrics"],
            "basis_diag": best_s["basis_diag"],
            "test_metrics": evaluated["selectivity_max"]["test_metrics"],
            "test_basis_diag": evaluated["selectivity_max"]["basis_diag"],
        },
        "lowest_after_protocol": None
        if best_after is None
        else {
            "selected_config": best_after["config"],
            "tune_metrics": best_after["tune_metrics"],
            "basis_diag": best_after["basis_diag"],
            "test_metrics": evaluated["lowest_after_task_acc"]["test_metrics"],
            "test_basis_diag": evaluated["lowest_after_task_acc"]["basis_diag"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS)
    parser.add_argument("--model_name", default="EleutherAI/pythia-70m")
    parser.add_argument("--layer_idx", type=int, default=-1)
    parser.add_argument("--hdmi_alpha", type=float, default=1.0)
    parser.add_argument("--hdmi_inner_steps", type=int, default=30)
    parser.add_argument("--gamma_grid", nargs="+", type=float, default=dpas.DEFAULT_GAMMA_GRID)
    parser.add_argument("--trust_beta_grid", nargs="+", type=float, default=dpas.DEFAULT_TRUST_BETA_GRID)
    parser.add_argument("--rank_grid", nargs="+", type=int, default=dpas.DEFAULT_RANK_GRID)
    parser.add_argument("--ge_weight_grid", nargs="+", type=float, default=dpas.DEFAULT_GE_WEIGHT_GRID)
    parser.add_argument("--output", default="compare_hdmi_dpas_metric_bcci_dual_protocol.json")
    args = parser.parse_args()

    dpas.set_seed(1337)
    cmp.ensure_local_json(args.tasks)

    device = hdmi.get_device("auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device).eval()
    model.config.output_hidden_states = True

    results = {
        "method": "DPAS-plus-metric-aligned-BCCI-dual-protocol",
        "tasks": [],
        "model_name": args.model_name,
        "layer_idx": args.layer_idx,
        "hdmi_alpha": args.hdmi_alpha,
        "hdmi_inner_steps": args.hdmi_inner_steps,
        "protocols": {
            "strict_protocol": {
                "description": "after_task_acc == 0 and completeness/selectivity not below local HDMI tune values",
            },
            "reliability_max_protocol": {
                "description": "select the highest-reliability config, regardless of strict feasibility",
            },
        },
    }

    for task in args.tasks:
        print(f"Running DPAS dual-protocol on {task}")
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
        results["tasks"].append(task_result)
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(task_result["candidate_summary"], indent=2))

    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
