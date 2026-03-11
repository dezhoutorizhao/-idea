import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AUCTR_PATH = ROOT / "compare_hdmi_auctr_failed_tasks.py"


def load_auctr():
    spec = importlib.util.spec_from_file_location("auctr_mod", AUCTR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


auctr = load_auctr()
hdmi = auctr.hdmi
cmp = auctr.cmp
race = auctr.race


def run_task_fast(model, tokenizer, task, device, layer_idx, alpha, inner_steps, base_cfg):
    auctr.set_seed(1337)
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

    fast_grid = [
        {"beta_base": 0.0, "beta_scale": 0.1, "beta_max": 0.2, "k_neighbors": 4},
        {"beta_base": 0.02, "beta_scale": 0.2, "beta_max": 0.35, "k_neighbors": 4},
        {"beta_base": 0.05, "beta_scale": 0.2, "beta_max": 0.35, "k_neighbors": 8},
        {"beta_base": 0.1, "beta_scale": 0.4, "beta_max": 0.5, "k_neighbors": 8},
    ]
    best_cfg = None
    best_metrics = None
    for cfg in fast_grid:
        metrics = auctr.evaluate_auctr(
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
            k_neighbors=int(cfg["k_neighbors"]),
            beta_base=float(cfg["beta_base"]),
            beta_scale=float(cfg["beta_scale"]),
            beta_max=float(cfg["beta_max"]),
        )
        if metrics["selectivity"] + 1e-6 < hdmi_tune.selectivity:
            continue
        if metrics["completeness"] <= hdmi_tune.completeness:
            continue
        score = (metrics["completeness"] - hdmi_tune.completeness, metrics["reliability"])
        if best_metrics is None:
            best_cfg = cfg
            best_metrics = metrics
        else:
            current_score = (
                best_metrics["completeness"] - hdmi_tune.completeness,
                best_metrics["reliability"],
            )
            if score > current_score:
                best_cfg = cfg
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

    test_metrics = None
    if best_cfg is not None:
        test_metrics = auctr.evaluate_auctr(
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
        "selected_fast_auctr_config": best_cfg,
        "selected_fast_auctr_tune_metrics": best_metrics,
        "hdmi_test": {
            "completeness": hdmi_test.completeness,
            "selectivity": hdmi_test.selectivity,
            "reliability": hdmi_test.reliability,
        },
        "fast_auctr_test": test_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["garden_mvrr_mod", "garden_npz_v-trans_mod", "garden_npz_obj_mod"])
    parser.add_argument("--model_name", default="EleutherAI/pythia-70m")
    parser.add_argument("--layer_idx", type=int, default=-1)
    parser.add_argument("--hdmi_alpha", type=float, default=1.0)
    parser.add_argument("--hdmi_inner_steps", type=int, default=30)
    parser.add_argument("--output", default="compare_hdmi_auctr_fast_failed_tasks.json")
    args = parser.parse_args()

    auctr.set_seed(1337)
    cmp.ensure_local_json(args.tasks)
    all_results = json.loads((ROOT / "compare_hdmi_race_all.json").read_text(encoding="utf-8"))
    race_cfg_map = {task_result["task"]: task_result["best_tuning_config"] for task_result in all_results["tasks"]}

    device = hdmi.get_device("auto")
    tokenizer = auctr.AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    model = auctr.AutoModelForCausalLM.from_pretrained(args.model_name).to(device).eval()
    model.config.output_hidden_states = True

    results = {"method": "AUCTR-fast", "tasks": []}
    for task in args.tasks:
        print(f"Running fast AUCTR on {task}")
        task_result = run_task_fast(
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
