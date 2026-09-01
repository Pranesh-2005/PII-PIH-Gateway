"""Build the BIO datasets the tagger trains on.

Writes JSONL with character spans, not token labels: tokenisation belongs to the model, and
keeping char offsets means the same file trains ModernBERT, DeBERTa or anything else in the
ablation without regeneration.

Splits:

* ``train`` / ``dev``  -- from the train surface pool, colon-heavy field layout.
* ``test``             -- from the *disjoint* test pool, colonless layout, adversarial half
  built from templates that appear in no training note.

Scored separately in ``eval/REPORT.md`` on purpose. A well-diagnosed honest result beats an
unverifiable good one, and reporting one blended number would hide generator memorisation.

Deliberately not wired in: ``ai4privacy/pii-masking-300k``. It is permissively licensed and
would add non-clinical breadth, but it is a 300k-row download and the clinical carrier here
is what the eval measures. Add it as a stage-one pretrain if recall on names plateaus.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from gen_notes import generate

OUT = Path(__file__).resolve().parents[2] / "data" / "bio"


def _write(path: Path, notes: list) -> Counter:
    counts: Counter = Counter()
    with path.open("w", encoding="utf-8") as fh:
        for n in notes:
            counts.update(s["label"] for s in n.spans)
            fh.write(json.dumps({
                "text": n.text, "spans": n.spans, "source": n.source, "split": n.split,
            }) + "\n")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=2400)
    ap.add_argument("--dev", type=int, default=300)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    # 30% adversarial. Enough that the hard cases shape the decision boundary, not so much
    # that ordinary clinical prose becomes the minority the model gets wrong.
    plan = {
        "train": (args.train, int(args.train * 0.3), "train", args.seed),
        "dev": (args.dev, int(args.dev * 0.3), "train", args.seed + 1),
        "test": (args.test, int(args.test * 0.5), "test", args.seed + 2),
    }
    for name, (n_syn, n_adv, pool, seed) in plan.items():
        notes = generate(n_syn, n_adv, pool, seed)
        counts = _write(args.out / f"{name}.jsonl", notes)
        total = sum(counts.values())
        print(f"{name:5s} {len(notes):5d} notes  {total:6d} spans  "
              f"{len(counts):2d} labels  top={counts.most_common(5)}")


if __name__ == "__main__":
    main()
