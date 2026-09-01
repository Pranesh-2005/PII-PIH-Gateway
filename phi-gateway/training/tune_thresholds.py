"""Phase 4: pick per-category thresholds that hit a recall floor on dev.

The brief's asymmetry stated as a procedure rather than a sentiment: for each category, take
the **highest** threshold that still meets ``--floor`` recall, and report the precision you paid
for it. Recall is the constraint; precision is the observation, so among the thresholds that
clear the floor the strictest one is the right pick.

Writes a YAML fragment ready to paste into ``configs/policy.yaml``, and prints the
precision/recall pair at each chosen point so the trade is visible rather than buried.

**Support is printed, and categories with no gold spans are excluded from the output.** The
first version of this script printed ``1.0000`` recall whenever the denominator was zero, which
made a category the dev set never mentions indistinguishable from a solved one -- and every
category came back ``thr=0.90 R=1.0000 P=1.0000``. Two different facts were hiding behind the
same number: real saturation (the tagger emits ~0.999 confidences, so no threshold in the grid
discriminates and the whole sweep is uninformative) and empty gold (PROFESSION, which reported
recall 1.0 alongside precision 0.0 -- arithmetically impossible unless tp+fn was 0). A tuned
threshold for a category with no evidence is a guess with a decimal point on it, so it is not
written at all.

Run after training:  ``python training/tune_thresholds.py --floor 0.99``
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GRID = [round(0.05 * i, 2) for i in range(1, 19)]     # 0.05 .. 0.90


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "bio" / "dev.jsonl")
    ap.add_argument("--floor", type=float, default=0.99)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--out", type=Path, default=ROOT / "configs" / "thresholds.tuned.yaml")
    a = ap.parse_args()

    from phi_gateway.config import load_policy
    from phi_gateway.detectors import fields as field_detector
    from phi_gateway.detectors import neural, rules, structural

    policy = load_policy()
    rows = [json.loads(l) for l in a.data.open(encoding="utf-8")][: a.limit]

    # (category, threshold) -> [tp, fp, fn]. Scored once over all thresholds in one pass:
    # every candidate span carries its score, so thresholding is a filter, not a re-run.
    tally: dict[tuple[str, float], list[int]] = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        segs = structural.segment(row["text"])
        cands = (rules.detect(row["text"], segs)
                 + field_detector.detect(row["text"], segs)
                 + neural.detect(row["text"], segs, policy))
        for t in GRID:
            kept = [s for s in cands if s.score >= t]
            for g in row["spans"]:
                if any(s.start < g["end"] and g["start"] < s.end
                       and s.category.value == g["label"] for s in kept):
                    tally[(g["label"], t)][0] += 1
                else:
                    tally[(g["label"], t)][2] += 1
            for s in kept:
                if not any(s.start < g["end"] and g["start"] < s.end
                           and s.category.value == g["label"] for g in row["spans"]):
                    tally[(s.category.value, t)][1] += 1

    categories = sorted({c for c, _ in tally})
    chosen: dict[str, float] = {}
    skipped: list[str] = []
    print(f"{'category':16s} {'thr':>5s} {'recall':>7s} {'prec':>7s} "
          f"{'tp':>5s} {'fp':>5s} {'fn':>5s}   (floor={a.floor})")
    for cat in categories:
        # Support at the most permissive point, which is the largest the gold count can be:
        # thresholding only removes candidates, never adds gold. Zero here means the dev slice
        # carries no annotation for this category at all, so there is nothing to tune against.
        base_tp, _, base_fn = tally[(cat, GRID[0])]
        if base_tp + base_fn == 0:
            print(f"{cat:16s} {'--':>5s} {'--':>7s} {'--':>7s} "
                  f"{0:5d} {tally[(cat, GRID[0])][1]:5d} {0:5d}   NO GOLD SPANS -- not tuned")
            skipped.append(cat)
            continue

        best = None
        for t in GRID:
            tp, fp, fn = tally[(cat, t)]
            rec = tp / (tp + fn)
            prec = tp / (tp + fp) if tp + fp else 0.0
            if rec >= a.floor:
                # Among thresholds clearing the recall floor, the strictest wins: same recall,
                # fewer false positives. See the module docstring.
                best = (t, rec, prec, tp, fp, fn)
        if best is None:
            # Floor unreachable at any threshold. Take the most permissive point and say so --
            # silently lowering the floor is how a compliance number becomes fiction.
            tp, fp, fn = tally[(cat, GRID[0])]
            rec = tp / (tp + fn)
            prec = tp / (tp + fp) if tp + fp else 0.0
            best = (GRID[0], rec, prec, tp, fp, fn)
            note = "   FLOOR UNREACHABLE"
        else:
            note = ""
        print(f"{cat:16s} {best[0]:5.2f} {best[1]:7.4f} {best[2]:7.4f} "
              f"{best[3]:5d} {best[4]:5d} {best[5]:5d}{note}")
        chosen[cat] = best[0]

    # Saturation is a property of the sweep, not of a category, so it is reported once. A model
    # trained to R=0.999 emits ~0.999 on nearly every span, so every grid point keeps the same
    # set and every row reads P=R=1.0000. That is not a tuning result -- it is the grid failing
    # to discriminate, and pasting it into policy.yaml would raise thresholds to 0.90 on the
    # strength of no evidence.
    flat = [c for c in chosen if len({tally[(c, t)][0] for t in GRID}) == 1]
    if flat:
        print(f"\nSATURATED (identical tp at every threshold in {GRID[0]}..{GRID[-1]}): "
              f"{len(flat)}/{len(chosen)} categories -- {', '.join(sorted(flat))}")
        print("The sweep cannot discriminate here. Do NOT paste these thresholds into "
              "policy.yaml; they are the grid maximum, not a measured optimum.")
    if skipped:
        print(f"\nNO GOLD SPANS, omitted from the YAML: {', '.join(skipped)}")

    a.out.write_text(
        "# Generated by training/tune_thresholds.py. Paste under `categories:` in policy.yaml.\n"
        "# Categories with no gold spans on dev are absent by design -- see the script docstring.\n"
        + "".join(f"  {c}:\n    threshold: {t}\n" for c, t in sorted(chosen.items())),
        encoding="utf-8",
    )
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
