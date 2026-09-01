"""Train the BIO tagger (Stage C).

Runs identically on a laptop CPU and on a Modal A10G -- ``modal_app.py`` only supplies the
GPU and the volume. Everything model-specific is a flag, because the ablation (Phase 9)
sweeps four encoders and a checkpoint that only trains under one config is not a result.

Two choices here answer P2.5 "recall asymmetry" directly:

* **Class-weighted loss.** ``O`` is ~90% of tokens. Unweighted, the cheapest way to a good
  loss is to predict ``O`` everywhere -- which is a breach generator. PHI classes get
  ``--phi-weight`` (default 5x).
* **Per-class thresholds are *not* set here.** Training emits calibrated probabilities; the
  recall floor is hit at inference by tuning thresholds on dev (Phase 4). Baking a threshold
  into the loss would make it untunable without retraining.

Long notes are windowed with stride, so a 22-page record does not get truncated at 512
tokens -- silent truncation is the single easiest way to ship a leak.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

#: The identifier categories the tagger predicts. Deliberately *not* every
#: ``types.Category``: ZIP, SSN, URL, IP and the rest are validated patterns where a rule
#: with a checksum beats a probability, and merge.py already ranks RULES above NEURAL for
#: them. The model handles what rules cannot: names, geography, organisations, and the
#: unbounded category-18 tail.
LABELS = (
    "NAME_PATIENT", "NAME_PROVIDER", "NAME_OTHER", "ORG", "GEO_STREET", "GEO_CITY",
    "DATE", "AGE", "AGE_OVER_89", "PHONE", "FAX", "EMAIL", "MRN", "HEALTH_PLAN_ID",
    "ACCOUNT", "LICENCE", "DEVICE", "ID_GENERIC", "PROFESSION",
)
BIO = ["O"] + [f"{p}-{l}" for l in LABELS for p in ("B", "I")]
LABEL2ID = {t: i for i, t in enumerate(BIO)}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def encode(rows: list[dict], tok, max_len: int, stride: int) -> list[dict]:
    """Char spans -> windowed BIO token labels.

    Special tokens and padding get ``-100`` so they are ignored by the loss. A token is
    ``B-`` only when it is the first token overlapping the span, which is what makes the
    entity-level (not token-level) metric in ``eval/`` meaningful.
    """
    out: list[dict] = []
    for row in rows:
        enc = tok(
            row["text"],
            truncation=True,
            max_length=max_len,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        spans = sorted(row["spans"], key=lambda s: s["start"])
        for ids, offsets in zip(enc["input_ids"], enc["offset_mapping"]):
            labels = []
            for (s, e) in offsets:
                if s == e:                       # special token or empty piece
                    labels.append(-100)
                    continue
                tag = "O"
                for sp in spans:
                    if s < sp["end"] and sp["start"] < e:
                        prefix = "B" if s <= sp["start"] else "I"
                        tag = f"{prefix}-{sp['label']}"
                        break
                labels.append(LABEL2ID.get(tag, 0))
            out.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
    return out


class WeightedTrainer(Trainer):
    """Cross-entropy with PHI classes upweighted. See the module docstring."""

    def __init__(self, *a, class_weights: torch.Tensor, **kw) -> None:
        super().__init__(*a, **kw)
        self._w = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        logits = model(**inputs).logits
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            weight=self._w.to(logits.device),
            ignore_index=-100,
        )
        return (loss, {"logits": logits}) if return_outputs else loss


def _spans_from_tags(tags: list[str]) -> set[tuple[int, int, str]]:
    """Strict BIO decode to (start, end, label) token spans, for entity-level scoring."""
    out, cur = set(), None
    for i, t in enumerate(tags + ["O"]):
        if t.startswith("B-") or (t.startswith("I-") and (cur is None or cur[2] != t[2:])):
            if cur:
                out.add((cur[0], i, cur[2]))
            cur = (i, i, t[2:])
        elif t == "O":
            if cur:
                out.add((cur[0], i, cur[2]))
            cur = None
    return out


def metrics_fn(eval_pred) -> dict:
    """Entity-level P/R/F1 plus the two numbers the brief actually cares about.

    ``recall`` is the headline. ``leak_rate`` is the fraction of *windows* with at least one
    missed identifier -- the compliance number, and the one that must go to zero.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    tp = fp = fn = 0
    leaky = total = 0
    for p_row, l_row in zip(preds, labels):
        keep = l_row != -100
        gold = _spans_from_tags([BIO[i] for i in l_row[keep]])
        pred = _spans_from_tags([BIO[i] for i in p_row[keep]])
        tp += len(gold & pred)
        fp += len(pred - gold)
        missed = len(gold - pred)
        fn += missed
        total += 1
        leaky += bool(missed)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "leak_rate": leaky / total if total else 0.0,
    }


def train(
    data_dir: Path,
    out_dir: Path,
    model_name: str = "answerdotai/ModernBERT-base",
    epochs: float = 3.0,
    lr: float = 3e-5,
    batch: int = 16,
    max_len: int = 512,
    stride: int = 128,
    phi_weight: float = 5.0,
    seed: int = 0,
) -> dict:
    tok = AutoTokenizer.from_pretrained(model_name)
    train_rows = encode(load_jsonl(data_dir / "train.jsonl"), tok, max_len, stride)
    dev_rows = encode(load_jsonl(data_dir / "dev.jsonl"), tok, max_len, stride)

    weights = torch.full((len(BIO),), phi_weight)
    weights[LABEL2ID["O"]] = 1.0

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(BIO),
        id2label={i: t for t, i in LABEL2ID.items()},
        label2id=LABEL2ID,
    )
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch * 2,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        seed=seed,
        bf16=torch.cuda.is_available(),
        report_to=[],
    )
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_rows,
        eval_dataset=dev_rows,
        data_collator=DataCollatorForTokenClassification(tok),
        compute_metrics=metrics_fn,
        class_weights=weights,
    )
    trainer.train()
    dev = trainer.evaluate()

    # P2.3 wants the exact parameter count, and it must be under 1e9. Recorded next to the
    # checkpoint so the claim in the README is checkable rather than asserted.
    n_params = sum(p.numel() for p in model.parameters())
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    result = {"model": model_name, "seed": seed, "n_params": n_params, **dev}
    test_path = data_dir / "test.jsonl"
    if test_path.exists():
        result["test"] = trainer.evaluate(encode(load_jsonl(test_path), tok, max_len, stride))
    (out_dir / "train_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--data", type=Path, default=root / "data" / "bio")
    ap.add_argument("--out", type=Path, default=root / "artifacts" / "tagger")
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--phi-weight", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train(a.data, a.out, a.model, a.epochs, a.lr, a.batch,
          phi_weight=a.phi_weight, seed=a.seed)


if __name__ == "__main__":
    main()
