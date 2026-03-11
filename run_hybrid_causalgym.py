import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


SEED = 42
TASKS = ["agr_sv_num_pp", "gss_subord_pp", "filler_gap_pp"]
HDMI_REPORTED = {
    "agr_sv_num_pp": {"completeness": 0.9984, "selectivity": 0.9409, "reliability": 0.9688},
    "gss_subord_pp": {"completeness": 0.9984, "selectivity": 0.7371, "reliability": 0.8481},
    "filler_gap_pp": {"completeness": 0.7404, "selectivity": 0.6275, "reliability": 0.6793},
}
PREP_FAMILY = {
    "of": "OF",
    "in": "IN",
    "inside": "IN",
    "within": "IN",
    "into": "IN",
    "by": "WITH_BY",
    "with": "WITH_BY",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Sample:
    text: str
    cf_text: str
    task: str
    base_type: str
    target_type: str
    base_label: str
    target_label: str
    zc_before: int
    zc_after: int
    ze: int


class Probe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int | None):
        super().__init__()
        if hidden_dim is None:
            self.net = nn.Linear(input_dim, output_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_text(parts: Sequence[str]) -> str:
    return "".join(parts)


def last_preposition_family(text: str) -> int:
    stripped = text.replace("<|endoftext|>", " ").lower()
    tokens = stripped.split()
    window = tokens[-12:]
    for token in reversed(window):
        token = token.strip(".,;:!?\"'()[]{}")
        if token in PREP_FAMILY:
            family = PREP_FAMILY[token]
            return {"OF": 1, "IN": 2, "WITH_BY": 3}[family]
        if token in {"to", "from", "on", "at", "under", "over", "behind", "before", "after", "near", "around", "through", "across", "for", "about"}:
            return 4
    return 0


def train_val_split(items: Sequence, ratio: float = 0.8, seed: int = SEED) -> Tuple[List, List]:
    idx = list(range(len(items)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    cut = max(1, int(len(idx) * ratio))
    train_idx = idx[:cut]
    val_idx = idx[cut:]
    if not val_idx:
        val_idx = train_idx[-1:]
        train_idx = train_idx[:-1]
    return [items[i] for i in train_idx], [items[i] for i in val_idx]


def harmonic_mean(a: float, b: float) -> float:
    if a + b == 0:
        return 0.0
    return 2 * a * b / (a + b)


def tv_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return 0.5 * (p - q).abs().sum(dim=-1)


def max_possible_tv_shift(p: torch.Tensor) -> torch.Tensor:
    p_min = p.min(dim=-1).values
    p_max = p.max(dim=-1).values
    return torch.maximum(1 - p_min, p_max).clamp_min(1e-6)


def encode_samples(task_rows: Sequence[dict]) -> List[Sample]:
    types = sorted({row["base_type"] for row in task_rows} | {row["src_type"] for row in task_rows})
    zc_map = {label: i for i, label in enumerate(types)}
    samples = []
    for row in task_rows:
        text = build_text(row["base"])
        cf_text = build_text(row["src"])
        samples.append(
            Sample(
                text=text,
                cf_text=cf_text,
                task=row["task"],
                base_type=row["base_type"],
                target_type=row["src_type"],
                base_label=row["base_label"],
                target_label=row["src_label"],
                zc_before=zc_map[row["base_type"]],
                zc_after=zc_map[row["src_type"]],
                ze=last_preposition_family(text),
            )
        )
    return samples


def filter_task_rows(dataset_split, task: str) -> List[dict]:
    return [row for row in dataset_split if row["task"] == task]


@torch.inference_mode()
def collect_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    reps = []
    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start:start + batch_size])
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[-1]
        last_idx = enc["attention_mask"].sum(dim=-1) - 1
        reps.append(hidden[torch.arange(hidden.size(0), device=device), last_idx].float().cpu())
    return torch.cat(reps, dim=0)


def build_probe_dataset(samples: Sequence[Sample], hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zc = torch.tensor([sample.zc_before for sample in samples], dtype=torch.long)
    ze = torch.tensor([sample.ze for sample in samples], dtype=torch.long)
    return hidden, zc, ze


def approximately_independent_indices(zc: torch.Tensor, ze: torch.Tensor, seed: int = SEED) -> List[int]:
    rng = random.Random(seed)
    classes = sorted(zc.unique().tolist())
    groups: Dict[int, Dict[int, List[int]]] = {}
    for idx, (zc_i, ze_i) in enumerate(zip(zc.tolist(), ze.tolist())):
        groups.setdefault(ze_i, {}).setdefault(zc_i, []).append(idx)

    keep = []
    for ze_value, class_to_indices in groups.items():
        if any(c not in class_to_indices for c in classes):
            continue
        quota = min(len(class_to_indices[c]) for c in classes)
        for c in classes:
            choices = class_to_indices[c][:]
            rng.shuffle(choices)
            keep.extend(choices[:quota])
    keep = sorted(set(keep))
    if not keep:
        return list(range(len(zc)))
    return keep


def fit_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    out_dim: int,
    device: torch.device,
) -> Tuple[Probe, Dict[str, float]]:
    candidates = [None, 64, 256, 512]
    best_probe = None
    best_state = None
    best_metrics = {"val_accuracy": -1.0, "hidden_dim": None}

    for hidden_dim in candidates:
        model = Probe(x_train.size(-1), out_dim, hidden_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-6)
        loss_fn = nn.CrossEntropyLoss()
        batch_size = min(256, len(x_train))
        best_local_acc = -1.0
        best_local_state = None
        for _ in range(100):
            perm = torch.randperm(len(x_train), device=device)
            for start in range(0, len(x_train), batch_size):
                idx = perm[start:start + batch_size]
                logits = model(x_train[idx])
                loss = loss_fn(logits, y_train[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                val_logits = model(x_val)
                val_acc = (val_logits.argmax(dim=-1) == y_val).float().mean().item()
            if val_acc > best_local_acc:
                best_local_acc = val_acc
                best_local_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if best_local_acc > best_metrics["val_accuracy"]:
            best_metrics = {"val_accuracy": best_local_acc, "hidden_dim": hidden_dim}
            best_probe = Probe(x_train.size(-1), out_dim, hidden_dim).to(device)
            best_state = best_local_state

    assert best_probe is not None and best_state is not None
    best_probe.load_state_dict(best_state)
    best_probe.eval()
    return best_probe, best_metrics


def token_ids(tokenizer: AutoTokenizer, labels: Sequence[str]) -> torch.Tensor:
    ids = []
    for label in labels:
        toks = tokenizer.encode(label, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"Expected single-token label, got {label!r} -> {toks}")
        ids.append(toks[0])
    return torch.tensor(ids, dtype=torch.long)


def hdmi_direction(embed_out: torch.Tensor, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
    return embed_out[tgt_ids] - embed_out[src_ids]


def learn_subspace(train_hidden: torch.Tensor, cf_hidden: torch.Tensor, rank: int) -> torch.Tensor:
    deltas = cf_hidden - train_hidden
    deltas = deltas / deltas.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    _, _, vh = torch.linalg.svd(deltas, full_matrices=False)
    rank = min(rank, vh.size(0))
    return vh[:rank].T


def project_direction(direction: torch.Tensor, basis: torch.Tensor, mix: float) -> torch.Tensor:
    projected = direction @ basis @ basis.T
    blended = (1.0 - mix) * direction + mix * projected
    proj_norm = blended.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    dir_norm = direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return blended / proj_norm * dir_norm


@torch.inference_mode()
def evaluate_method(
    probe_zc: Probe,
    probe_ze: Probe,
    hidden: torch.Tensor,
    direction: torch.Tensor,
    zc_after: torch.Tensor,
    alpha: float,
    inner_steps: int,
    device: torch.device,
) -> Dict[str, float]:
    hidden = hidden.to(device)
    direction = direction.to(device)
    zc_after = zc_after.to(device)
    before_zc = probe_zc(hidden).softmax(dim=-1)
    before_ze = probe_ze(hidden).softmax(dim=-1)
    hidden_after = hidden + alpha * inner_steps * direction
    after_zc = probe_zc(hidden_after).softmax(dim=-1)
    after_ze = probe_ze(hidden_after).softmax(dim=-1)

    target_onehot = torch.zeros_like(after_zc)
    target_onehot.scatter_(1, zc_after.unsqueeze(-1), 1.0)
    completeness = 1.0 - tv_distance(after_zc, target_onehot)
    selectivity = 1.0 - tv_distance(after_ze, before_ze) / max_possible_tv_shift(before_ze)
    comp = completeness.mean().item()
    sel = selectivity.mean().item()
    return {
        "completeness": comp,
        "selectivity": sel,
        "reliability": harmonic_mean(comp, sel),
    }


def prepare_split(
    samples: Sequence[Sample],
    hidden: torch.Tensor,
    cf_hidden: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    data = {
        "hidden": hidden,
        "zc_before": torch.tensor([s.zc_before for s in samples], dtype=torch.long),
        "zc_after": torch.tensor([s.zc_after for s in samples], dtype=torch.long),
        "ze": torch.tensor([s.ze for s in samples], dtype=torch.long),
        "src_ids": None,
        "tgt_ids": None,
    }
    if cf_hidden is not None:
        data["cf_hidden"] = cf_hidden
    return data


def run_task(
    task: str,
    dataset,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
    alpha: float,
    inner_steps: int,
) -> Dict[str, object]:
    train_samples = encode_samples(filter_task_rows(dataset["train"], task))
    val_samples = encode_samples(filter_task_rows(dataset["validation"], task))
    test_samples = encode_samples(filter_task_rows(dataset["test"], task))

    train_text = [s.text for s in train_samples]
    train_cf_text = [s.cf_text for s in train_samples]
    val_text = [s.text for s in val_samples]
    test_text = [s.text for s in test_samples]

    train_hidden = collect_hidden_states(model, tokenizer, train_text, device)
    train_cf_hidden = collect_hidden_states(model, tokenizer, train_cf_text, device)
    val_hidden = collect_hidden_states(model, tokenizer, val_text, device)
    test_hidden = collect_hidden_states(model, tokenizer, test_text, device)

    probe_x, probe_zc, probe_ze = build_probe_dataset(val_samples, val_hidden)
    keep = approximately_independent_indices(probe_zc, probe_ze)
    probe_x = probe_x[keep]
    probe_zc = probe_zc[keep]
    probe_ze = probe_ze[keep]

    probe_indices = list(range(len(probe_x)))
    probe_train_idx, probe_val_idx = train_val_split(probe_indices, ratio=0.8, seed=SEED)
    probe_train_x = probe_x[probe_train_idx].to(device)
    probe_val_x = probe_x[probe_val_idx].to(device)
    probe_zc_train = probe_zc[probe_train_idx].to(device)
    probe_zc_val = probe_zc[probe_val_idx].to(device)
    probe_ze_train = probe_ze[probe_train_idx].to(device)
    probe_ze_val = probe_ze[probe_val_idx].to(device)

    probe_zc_model, zc_probe_metrics = fit_probe(
        probe_train_x,
        probe_zc_train,
        probe_val_x,
        probe_zc_val,
        out_dim=len(set([s.zc_before for s in val_samples])),
        device=device,
    )
    probe_ze_model, ze_probe_metrics = fit_probe(
        probe_train_x,
        probe_ze_train,
        probe_val_x,
        probe_ze_val,
        out_dim=max(s.ze for s in val_samples) + 1,
        device=device,
    )

    embed_out = model.embed_out.weight.detach().float().cpu()
    train_src_ids = token_ids(tokenizer, [s.base_label for s in train_samples])
    train_tgt_ids = token_ids(tokenizer, [s.target_label for s in train_samples])
    test_src_ids = token_ids(tokenizer, [s.base_label for s in test_samples])
    test_tgt_ids = token_ids(tokenizer, [s.target_label for s in test_samples])

    hdmi_test_dir = hdmi_direction(embed_out, test_src_ids, test_tgt_ids)
    hdmi_metrics = evaluate_method(
        probe_zc_model,
        probe_ze_model,
        test_hidden,
        hdmi_test_dir,
        torch.tensor([s.zc_after for s in test_samples], dtype=torch.long),
        alpha=alpha,
        inner_steps=inner_steps,
        device=device,
    )

    train_idx, tune_idx = train_val_split(list(range(len(train_samples))), ratio=0.8, seed=SEED)
    tune_hidden = train_hidden[tune_idx]
    tune_src_ids = train_src_ids[tune_idx]
    tune_tgt_ids = train_tgt_ids[tune_idx]
    tune_zc_after = torch.tensor([train_samples[i].zc_after for i in tune_idx], dtype=torch.long)

    best_cfg = None
    best_metrics = None
    for rank in [1, 2, 4, 8, 16, 32]:
        basis = learn_subspace(train_hidden[train_idx], train_cf_hidden[train_idx], rank)
        for mix in [0.25, 0.5, 0.75, 1.0]:
            tune_dir = hdmi_direction(embed_out, tune_src_ids, tune_tgt_ids)
            hybrid_tune_dir = project_direction(tune_dir, basis, mix)
            metrics = evaluate_method(
                probe_zc_model,
                probe_ze_model,
                tune_hidden,
                hybrid_tune_dir,
                tune_zc_after,
                alpha=alpha,
                inner_steps=inner_steps,
                device=device,
            )
            if best_metrics is None or metrics["reliability"] > best_metrics["reliability"]:
                best_metrics = metrics
                best_cfg = {"rank": rank, "mix": mix}

    assert best_cfg is not None and best_metrics is not None
    basis = learn_subspace(train_hidden, train_cf_hidden, best_cfg["rank"])
    hybrid_test_dir = project_direction(hdmi_test_dir, basis, best_cfg["mix"])
    hybrid_metrics = evaluate_method(
        probe_zc_model,
        probe_ze_model,
        test_hidden,
        hybrid_test_dir,
        torch.tensor([s.zc_after for s in test_samples], dtype=torch.long),
        alpha=alpha,
        inner_steps=inner_steps,
        device=device,
    )

    return {
        "task": task,
        "probe_metrics": {
            "zc_probe_val_accuracy": zc_probe_metrics["val_accuracy"],
            "ze_probe_val_accuracy": ze_probe_metrics["val_accuracy"],
        },
        "best_tuning_config": best_cfg,
        "best_tuning_metrics": best_metrics,
        "hdmi_reproduced": hdmi_metrics,
        "hybrid": hybrid_metrics,
        "hdmi_reported": HDMI_REPORTED[task],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--inner_steps", type=int, default=30)
    parser.add_argument("--output", default="hybrid_results.json")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    dataset = load_dataset("aryaman/causalgym")
    results = {
        "model": args.model,
        "alpha": args.alpha,
        "inner_steps": args.inner_steps,
        "tasks": [],
    }
    for task in args.tasks:
        print(f"Running task: {task}")
        task_result = run_task(task, dataset, tokenizer, model, device, args.alpha, args.inner_steps)
        results["tasks"].append(task_result)
        print(json.dumps(task_result, indent=2))

    reported_mean = {
        metric: float(np.mean([t["hdmi_reported"][metric] for t in results["tasks"]]))
        for metric in ["completeness", "selectivity", "reliability"]
    }
    reproduced_mean = {
        metric: float(np.mean([t["hdmi_reproduced"][metric] for t in results["tasks"]]))
        for metric in ["completeness", "selectivity", "reliability"]
    }
    hybrid_mean = {
        metric: float(np.mean([t["hybrid"][metric] for t in results["tasks"]]))
        for metric in ["completeness", "selectivity", "reliability"]
    }
    results["means"] = {
        "hdmi_reported": reported_mean,
        "hdmi_reproduced": reproduced_mean,
        "hybrid": hybrid_mean,
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {output_path}")
    print(json.dumps(results["means"], indent=2))


if __name__ == "__main__":
    main()
