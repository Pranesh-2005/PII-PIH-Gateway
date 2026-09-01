"""Modal app: train the tagger on a GPU, and run the ablation sweep in parallel.

Three entry points:

* ``modal run training/modal_app.py::build_data``  -- generate the BIO datasets into the volume.
* ``modal run training/modal_app.py::train``       -- one A10G training run, checkpoint to the volume.
* ``modal run training/modal_app.py::ablate``      -- 4 encoders x 2 seeds, fanned out with ``.map``.

Pull the checkpoint down with the CLI rather than a function -- a 600MB return value is a bad
idea and ``modal volume get`` already exists::

    modal volume get phi-gateway-data artifacts/ModernBERT-base-s0 artifacts/tagger

**Serving stays CPU-local.** A ~150M encoder is ~50ms/note on CPU, which is the stronger
demo story: no network, no GPU, full data residency -- the entire premise of the project.
Modal is here for training and for the sweep, not for inference.
"""

from __future__ import annotations

import modal

app = modal.App("phi-gateway-tagger")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.48.0",
        "accelerate==1.2.1",
        # deberta-v3 ships a SentencePiece tokenizer and nothing else can convert it; without
        # these two the whole sweep dies on one config. protobuf is sentencepiece's silent
        # runtime requirement for the slow->fast conversion.
        "sentencepiece==0.2.0",
        "protobuf==5.29.2",
        "tiktoken==0.8.0",
        "numpy<2.2",
    )
    .add_local_dir(
        __file__.rsplit("modal_app.py", 1)[0], remote_path="/root/training", copy=True
    )
)

vol = modal.Volume.from_name("phi-gateway-data", create_if_missing=True)
VOL = "/vol"

#: The ablation grid. Four encoders spanning the plausible choices -- a modern long-context
#: encoder, a strong general encoder, a clinical-domain encoder, and the ordinary baseline --
#: times two seeds, because a single-seed ranking of four close models is noise.
#: Every one is well under the 1B parameter ceiling P2.3 imposes.
ABLATION = [
    {"model": m, "seed": s}
    for m in (
        "answerdotai/ModernBERT-base",
        "microsoft/deberta-v3-base",
        "emilyalsentzer/Bio_ClinicalBERT",
        "roberta-base",
    )
    for s in (0, 1)
]


@app.function(image=image, volumes={VOL: vol}, timeout=900)
def build_data(train: int = 2400, dev: int = 300, test: int = 300, seed: int = 17) -> dict:
    import subprocess
    import sys

    out = f"{VOL}/bio"
    subprocess.run(
        [sys.executable, "build_datasets.py", "--train", str(train), "--dev", str(dev),
         "--test", str(test), "--seed", str(seed), "--out", out],
        cwd="/root/training/data", check=True,
    )
    vol.commit()
    import os

    return {f: os.path.getsize(f"{out}/{f}") for f in sorted(os.listdir(out))}


@app.function(image=image, gpu="A10G", volumes={VOL: vol}, timeout=3600)
def train(
    model: str = "answerdotai/ModernBERT-base",
    seed: int = 0,
    epochs: float = 3.0,
    lr: float = 3e-5,
    batch: int = 16,
    phi_weight: float = 5.0,
) -> dict:
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root/training")
    from train import train as run          # noqa: PLC0415  (import needs the sys.path above)

    tag = f"{model.split('/')[-1]}-s{seed}"
    result = run(
        Path(f"{VOL}/bio"), Path(f"{VOL}/artifacts/{tag}"),
        model_name=model, epochs=epochs, lr=lr, batch=batch,
        phi_weight=phi_weight, seed=seed,
    )
    vol.commit()
    return result


@app.function(image=image, volumes={VOL: vol}, timeout=7200)
def ablate(epochs: float = 3.0) -> list[dict]:
    """Fan the grid out across containers, then rank on recall.

    Recall, not F1: a false negative is a breach and a false positive is an inconvenience.
    Every run is reported in ``eval/REPORT.md``, not just the winner -- a sweep where only
    the best number survives is not evidence.
    """
    import json

    # ``return_exceptions`` because one unloadable encoder must not take the sweep with it --
    # the first attempt died entirely on a deberta-v3 tokenizer conversion, losing seven
    # healthy runs. A failed config is reported as a failed config.
    raw = list(
        train.map(
            [c["model"] for c in ABLATION],
            [c["seed"] for c in ABLATION],
            kwargs={"epochs": epochs},
            return_exceptions=True,
        )
    )
    results, failed = [], []
    for cfg, r in zip(ABLATION, raw):
        if isinstance(r, Exception):
            failed.append({**cfg, "error": f"{type(r).__name__}: {r}"[:300]})
        else:
            results.append(r)
    results.sort(key=lambda r: r.get("eval_recall", 0.0), reverse=True)
    with open(f"{VOL}/ablation.json", "w", encoding="utf-8") as fh:
        json.dump({"runs": results, "failed": failed}, fh, indent=2)
    vol.commit()
    for r in results:
        print(f"{r['model']:42s} seed={r['seed']}  "
              f"R={r.get('eval_recall', 0):.4f}  P={r.get('eval_precision', 0):.4f}  "
              f"leak={r.get('eval_leak_rate', 1):.4f}  params={r['n_params']:,}")
    for f in failed:
        print(f"{f['model']:42s} seed={f['seed']}  FAILED  {f['error']}")
    return results


@app.local_entrypoint()
def main(step: str = "all", model: str = "answerdotai/ModernBERT-base", epochs: float = 3.0):
    """``modal run training/modal_app.py --step data|train|ablate|all``."""
    if step in ("data", "all"):
        print("datasets:", build_data.remote())
    if step in ("train", "all"):
        print("train:", train.remote(model=model, epochs=epochs))
    if step == "ablate":
        ablate.remote(epochs=epochs)
