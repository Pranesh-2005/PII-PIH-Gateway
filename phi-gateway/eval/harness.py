"""Evaluation harness: entity-level P/R/F1, recall, and leak rate.

Three systems, same data, same scorer:

* ``gateway``      -- the full pipeline (rules + fields + tagger + arbitration).
* ``regex_only``   -- the identical pipeline with Stage C disabled. This is the ablation that
  answers "did the model actually buy anything", and it is also the honest floor: if the
  tagger adds nothing, the right call is to ship without it and say so.
* ``presidio``     -- Microsoft Presidio with default recognizers. Optional; skipped with a
  printed note if it is not installed, because a missing baseline should not block a run.

**Leak rate is the headline compliance number**: the fraction of *notes* with at least one
missed identifier. Recall is reported next to it, and precision *at* that recall rather than
instead of it -- a false negative is a breach and a false positive is an inconvenience, so a
system tuned to look good on precision is the wrong trade.

Two leak numbers are printed, because the strict one turned out to be measuring two things.
``leak_rate`` counts any unmatched gold span, and a *category disagreement* is unmatched: the
generator's adversarial half is full of ``<surname> Memorial Hospital``, and a tagger that calls
that a name rather than an organisation scores identically to one that missed it entirely --
though the first produced ``[NAME_PATIENT_3]`` and left nothing readable. ``uncovered_rate`` is
the label-blind version and is the actual breach rate. The gap between them is the taxonomy
error rate. Reporting only the strict number overstates exposure; reporting only the loose one
hides that placeholder types are wrong, which is a utility problem downstream.

Matching is span overlap on the *character* level with a category check, not exact boundary
equality. A mask covering "Marcus D. Whitfield" when the gold span is "Whitfield" has leaked
nothing; scoring it as a miss would reward tighter masks over safer ones. Boundary quality is
reported separately as ``exact``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phi_gateway.config import load_policy                    # noqa: E402
from phi_gateway.pipeline import deidentify_full               # noqa: E402

#: Gold labels the tagger predicts but the rule layer cannot be expected to, and vice versa.
#: Scoring is over the union; nothing is excluded, because an identifier no system was asked
#: to find is still a leak.
_ALIASES = {
    # A patient/provider distinction the gold data makes and Presidio does not. Collapsed
    # only for the Presidio comparison, and only for it -- see ``_normalise``.
    "NAME_PATIENT": "NAME", "NAME_PROVIDER": "NAME", "NAME_OTHER": "NAME",
    "GEO_STREET": "GEO", "GEO_CITY": "GEO", "GEO_ZIP": "GEO",
    "MRN": "ID", "ACCOUNT": "ID", "HEALTH_PLAN_ID": "ID", "LICENCE": "ID",
    "DEVICE": "ID", "ID_GENERIC": "ID", "SSN": "ID",
    "AGE": "AGE", "AGE_OVER_89": "AGE", "FAX": "PHONE",
}


def _normalise(label: str, coarse: bool) -> str:
    return _ALIASES.get(label, label) if coarse else label


@dataclass
class Score:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    exact: int = 0
    notes: int = 0
    leaky_notes: int = 0
    #: Notes with at least one gold span that **no** prediction covers, whatever its label.
    #: This, not ``leaky_notes``, is the compliance number. See ``uncovered_rate``.
    exposed_notes: int = 0
    uncovered: int = 0
    ms: list[float] = field(default_factory=list)
    per_cat: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0, 0]))
    #: Surface text of every missed gold span, ``(category, surface)``. A leak rate says how
    #: many notes leaked; this says *what* leaked, which is the only form of the number that
    #: can be acted on. Without it the 0.60 leak rate was traceable to a category but not to a
    #: shape, and "ORG recall is 0.87" is not a bug report.
    missed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def leak_rate(self) -> float:
        return self.leaky_notes / self.notes if self.notes else 0.0

    @property
    def uncovered_rate(self) -> float:
        """Fraction of notes where an identifier is still **readable** in the masked text.

        ``leak_rate`` counts a note as leaking whenever a gold span went unmatched, and a
        category disagreement counts as unmatched. That conflates two failures with very
        different consequences: "Szymanski Memorial Hospital" masked as ``[NAME_PATIENT_3]``
        instead of ``[ORG_3]`` is a *taxonomy* error -- the identifier is gone from the
        document -- while an unmatched span that no prediction covers at all is a breach.

        Both are reported. This one is the compliance number; ``leak_rate`` is the stricter
        upper bound and stays visible so the gap between them is the taxonomy error rate
        rather than something to be quietly discovered later.
        """
        return self.exposed_notes / self.notes if self.notes else 0.0

    def row(self) -> dict:
        ms = sorted(self.ms)
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "leak_rate": round(self.leak_rate, 4),
            "uncovered_rate": round(self.uncovered_rate, 4),
            "uncovered_spans": self.uncovered,
            "exact_boundary": round(self.exact / self.tp, 4) if self.tp else 0.0,
            "notes": self.notes,
            "p50_ms": round(ms[len(ms) // 2], 1) if ms else 0.0,
            "p95_ms": round(ms[int(len(ms) * 0.95)], 1) if ms else 0.0,
        }


def _score_note(sc: Score, text: str, gold: list[dict], pred: list[tuple[int, int, str]],
                coarse: bool) -> None:
    """Greedy overlap match, one gold span to at most one prediction."""
    unused = list(pred)
    missed = 0
    exposed = 0
    for g in gold:
        gl = _normalise(g["label"], coarse)
        hit = None
        for i, (s, e, label) in enumerate(unused):
            if s < g["end"] and g["start"] < e and _normalise(label, coarse) == gl:
                hit = i
                break
        if hit is None:
            sc.fn += 1
            sc.per_cat[gl][2] += 1
            sc.missed.append((gl, text[g["start"]:g["end"]]))
            missed += 1
            # Label-blind second look. If *any* prediction overlaps this gold span, the text
            # was masked and the miss is a category disagreement, not an exposure.
            if not any(s < g["end"] and g["start"] < e for s, e, _ in pred):
                sc.uncovered += 1
                exposed += 1
        else:
            s, e, _ = unused.pop(hit)
            sc.tp += 1
            sc.per_cat[gl][0] += 1
            if (s, e) == (g["start"], g["end"]):
                sc.exact += 1
    for _, _, label in unused:
        sc.fp += 1
        sc.per_cat[_normalise(label, coarse)][1] += 1
    sc.notes += 1
    sc.leaky_notes += bool(missed)
    sc.exposed_notes += bool(exposed)


# --------------------------------------------------------------------------------------
# Systems under test
# --------------------------------------------------------------------------------------


def _gateway_runner(neural: bool):
    policy = load_policy()
    policy.neural.enabled = neural

    def run(text: str) -> tuple[list[tuple[int, int, str]], float]:
        r = deidentify_full(text, policy=policy)
        return ([(s.start, s.end, s.category.value) for s in r.spans],
                r.stats["ms_total"])

    return run


def _presidio_runner():
    try:
        from presidio_analyzer import AnalyzerEngine
    except ImportError:
        return None
    engine = AnalyzerEngine()
    # Presidio's taxonomy is its own. Mapped to ours only where the meaning matches; an
    # unmapped entity is still counted as a prediction, so the baseline is not flattered by
    # silently dropping its false positives.
    m = {
        "PERSON": "NAME_PATIENT", "ORGANIZATION": "ORG", "LOCATION": "GEO_CITY",
        "DATE_TIME": "DATE", "PHONE_NUMBER": "PHONE", "EMAIL_ADDRESS": "EMAIL",
        "US_SSN": "SSN", "US_DRIVER_LICENSE": "LICENCE", "URL": "URL", "IP_ADDRESS": "IP",
        "MEDICAL_LICENSE": "LICENCE", "AGE": "AGE", "NRP": "PROFESSION",
    }

    def run(text: str) -> tuple[list[tuple[int, int, str]], float]:
        import time
        t0 = time.perf_counter()
        res = engine.analyze(text=text, language="en")
        ms = (time.perf_counter() - t0) * 1000
        return [(r.start, r.end, m.get(r.entity_type, "ID_GENERIC")) for r in res], ms

    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "bio" / "test.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = all notes")
    ap.add_argument("--coarse", action="store_true",
                    help="collapse NAME_*/GEO_*/ID_* before matching (fair to Presidio)")
    ap.add_argument("--systems", default="gateway,regex_only,presidio")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results.json")
    a = ap.parse_args()

    rows = [json.loads(l) for l in a.data.open(encoding="utf-8")]
    if a.limit:
        rows = rows[: a.limit]

    runners: dict = {}
    for name in a.systems.split(","):
        if name == "gateway":
            runners[name] = _gateway_runner(neural=True)
        elif name == "regex_only":
            runners[name] = _gateway_runner(neural=False)
        elif name == "presidio":
            r = _presidio_runner()
            if r is None:
                print("presidio not installed -- baseline skipped "
                      "(pip install 'phi-gateway[eval]')")
                continue
            runners[name] = r

    # Split-aware: the adversarial half is where leak rate must be zero, so it is scored on
    # its own as well as blended. A blended number hides the case the panel will probe.
    results: dict = {}
    for name, run in runners.items():
        overall, by_source = Score(), defaultdict(Score)
        for row in rows:
            pred, ms = run(row["text"])
            for sc in (overall, by_source[row["source"]]):
                _score_note(sc, row["text"], row["spans"], pred, a.coarse)
                sc.ms.append(ms)
        results[name] = {
            "overall": overall.row(),
            "by_source": {k: v.row() for k, v in by_source.items()},
            "per_category": {
                cat: {"tp": v[0], "fp": v[1], "fn": v[2],
                      "recall": round(v[0] / (v[0] + v[2]), 4) if v[0] + v[2] else 0.0}
                for cat, v in sorted(overall.per_cat.items())
            },
            # Distinct missed surfaces with a count each, worst first. Deduplicated because a
            # generator repeats the same organisation across notes and fifty copies of one
            # miss is one bug, not fifty.
            "missed_surfaces": {
                cat: [{"text": t, "n": n} for (t, n) in
                      sorted(Counter(s for c, s in overall.missed if c == cat).items(),
                             key=lambda kv: -kv[1])[:40]]
                for cat in sorted({c for c, _ in overall.missed})
            },
        }
        o = results[name]["overall"]
        print(f"{name:12s} R={o['recall']:.4f}  P={o['precision']:.4f}  F1={o['f1']:.4f}  "
              f"leak={o['leak_rate']:.4f}  uncovered={o['uncovered_rate']:.4f}  "
              f"p50={o['p50_ms']}ms  p95={o['p95_ms']}ms")
        for cat, items in results[name]["missed_surfaces"].items():
            print(f"  MISSED {cat:12s} " + ", ".join(f"{i['text']!r}x{i['n']}" for i in items[:12]))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
