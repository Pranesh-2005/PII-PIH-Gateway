"""Phase 8: does the masked note still answer clinical questions?

The other half of the brief's tension. Leak rate measures the first failure ("redact too
little and you leak patient data"); this measures the second ("redact too much and the
downstream LLM becomes useless"). A gateway reporting only leak rate has answered half the
question and can trivially score perfectly on it by masking everything.

Two checks, deliberately unequal in kind:

**1. Interval preservation -- arithmetic, no LLM, no API key, cannot flake.**
   ``interval_preserving`` mode emits ``[DATE_7 = DATE_3 + 12d]``. The mapping still holds the
   original surface for each placeholder, so the *claimed* interval can be checked against the
   *true* one with subtraction. This is the plan's "interval question" reduced to its actual
   content: either every annotated delta equals the real delta or the date policy is broken.
   An LLM in the loop here would only add a way for a correct answer to be graded wrong.

**2. Fixed-question QA -- same questions, original vs masked, answers compared.**
   Questions are generic across clinical notes on purpose. Ten hand-written per note across a
   50-note set is real annotation labour with no obvious payoff: what matters is whether
   masking *changed the answer*, and that is measurable with questions that apply to any note.
   Agreement is scored with ``rapidfuzz`` (already a dependency for entity clustering) on
   normalised answers.

   Requires a provider key. Without one the check is skipped with a printed note rather than
   failing the run -- same policy as the Presidio baseline in ``harness.py``.

Run::

    python eval/utility/qa_eval.py                 # interval check only, offline
    python eval/utility/qa_eval.py --llm           # adds the QA half, needs GROQ/NVIDIA key
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phi_gateway.config import load_policy                          # noqa: E402
from phi_gateway.pipeline import deidentify_full                     # noqa: E402
from phi_gateway.policy import dates as date_policy                  # noqa: E402

#: ``[DATE_7 = DATE_3 + 12d]`` / ``[DATE_2 = DATE_1 - 4d]``.
_ANNOTATED = re.compile(r"\[(DATE_\d+)\s*=\s*(DATE_\d+)\s*([+-])\s*(\d+)d\]")

#: Answerable from clinical content alone, in any note that has that content. A question whose
#: answer *is* an identifier ("what is the patient's name?") is excluded by design -- masking is
#: supposed to destroy that, and counting it as utility loss would score the gateway for working.
QUESTIONS = (
    "What is the primary diagnosis or presenting complaint?",
    "What medications are mentioned, with doses if stated?",
    "What was the pain score, and on what scale?",
    "What did the physical examination find?",
    "What imaging or laboratory studies were performed, and what did they show?",
    "What is the treatment plan?",
    "How many days elapsed between the earliest and the latest date in this note?",
    "Was any surgical procedure performed or planned? If so, what?",
    "What were the vital signs?",
    "What follow-up interval was specified?",
)

_UNKNOWN = ("unknown", "not stated", "not mentioned", "not specified", "n/a", "none",
            "not provided", "cannot determine", "not documented")


def check_intervals(text: str, policy=None) -> dict:
    """Verify every annotated interval in the masked text against the true interval.

    Returns counts rather than a bool: a partial failure is a different diagnosis from a total
    one, and "0 annotations found" is a third thing again (the note has fewer than two parseable
    full dates, so there was no interval to preserve).
    """
    result = deidentify_full(text, policy=policy)
    masked, mapping = result.masked_text, result.mapping

    originals: dict[str, object] = {}
    for name, entry in mapping.entries.items():
        # Mapping keys are the *rendered* placeholder, annotation included:
        # ``'[DATE_1 = DATE_3 - 151d]'``, not ``'DATE_1'``. Pull the bare name back out.
        bare = re.match(r"\[(DATE_\d+)", name)
        if bare is None:
            continue
        for surface in (entry.canonical, *entry.surfaces):
            parsed = date_policy.parse(surface)
            if parsed is not None and parsed.full:
                originals[bare.group(1)] = parsed.as_date()
                break

    checked = correct = unresolvable = 0
    failures: list[dict] = []
    for m in _ANNOTATED.finditer(masked):
        target, anchor, sign, days = m.group(1), m.group(2), m.group(3), int(m.group(4))
        if target not in originals or anchor not in originals:
            # The annotation names a placeholder whose original we could not re-parse. Counted
            # separately and never as a pass: an unverifiable claim is not a verified one.
            unresolvable += 1
            continue
        checked += 1
        claimed = days if sign == "+" else -days
        true = (originals[target] - originals[anchor]).days      # type: ignore[operator]
        if claimed == true:
            correct += 1
        else:
            failures.append({"target": target, "anchor": anchor,
                             "claimed_days": claimed, "true_days": true})

    return {
        "annotations_checked": checked,
        "intervals_correct": correct,
        "unresolvable": unresolvable,
        "failures": failures,
        "date_mode": mapping.date_mode,
        # No date element survives in interval_preserving mode, so this must be empty. A bare
        # 4-digit year is allowed by Safe Harbor; anything finer in the masked text is a leak,
        # and this is a cheap independent check on the date policy rather than on the detectors.
        #
        # Same separator on both sides, via backreference -- the first version of this check
        # allowed mixed separators and so flagged the pain-score range "2-3/10" as a surviving
        # date, which is exactly the false positive the rule layer had to fix. A check that
        # reports the thing it was built to disprove is worse than no check.
        "residual_date_surfaces": re.findall(
            r"\b\d{1,2}(?P<sep>[/-])\d{1,2}(?P=sep)\d{2,4}\b|"
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\b",
            masked,
        ),
    }


def _normalise(a: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", a.lower()).strip()


def qa_agreement(text: str, policy=None) -> dict | None:
    """Ask each question of the original and of the masked note; score answer agreement."""
    try:
        from rapidfuzz import fuzz

        from phi_gateway.llm.client import LLMClient
    except ImportError as exc:
        print(f"QA half skipped: {exc}")
        return None

    pol = policy or load_policy()
    client = LLMClient(pol.llm)
    if not client.available_providers():
        print("QA half skipped: no provider key (set GROQ_API_KEY or NVIDIA_API_KEY)")
        return None

    masked = deidentify_full(text, policy=policy).masked_text
    rows = []
    for q in QUESTIONS:
        pair = []
        for body in (text, masked):
            pair.append(client.complete(
                f"NOTE:\n{body}\n\nQUESTION: {q}",
                system="Answer strictly from the note. Be terse -- one short sentence, no "
                       "preamble. If the note does not say, reply exactly: not stated.",
                max_tokens=160,
            ).text.strip())
        orig, mask = pair
        # Both sides saying "not stated" is agreement about absence, not preserved utility.
        # Counted apart so a note full of missing content cannot inflate the score.
        vacuous = any(u in _normalise(orig) for u in _UNKNOWN) and \
                  any(u in _normalise(mask) for u in _UNKNOWN)
        rows.append({
            "question": q,
            "original": orig,
            "masked": mask,
            "similarity": round(fuzz.token_set_ratio(_normalise(orig), _normalise(mask)) / 100, 3),
            "vacuous": vacuous,
        })

    scored = [r for r in rows if not r["vacuous"]]
    return {
        "questions": len(rows),
        "scored": len(scored),
        "vacuous": len(rows) - len(scored),
        "mean_similarity": round(sum(r["similarity"] for r in scored) / len(scored), 4)
        if scored else 0.0,
        "agree_at_0.80": sum(r["similarity"] >= 0.80 for r in scored),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "bio" / "test.jsonl")
    ap.add_argument("--files", type=Path, nargs="*", default=[],
                    help="score these note files instead of the jsonl split")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--llm", action="store_true", help="also run the fixed-question QA half")
    ap.add_argument("--llm-notes", type=int, default=3,
                    help="notes to send through the LLM (2 calls x 10 questions each)")
    ap.add_argument("--no-neural", action="store_true",
                    help="rules only. Interval preservation is a property of the date policy, "
                         "not of the tagger, so this measures the same thing ~150x faster")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "utility.json")
    a = ap.parse_args()

    policy = load_policy()
    if a.no_neural:
        policy.neural.enabled = False
    if a.files:
        texts = [f.read_text(encoding="utf-8") for f in a.files]
    else:
        texts = [json.loads(l)["text"] for l in a.data.open(encoding="utf-8")][: a.limit]

    agg = {"notes": 0, "annotations_checked": 0, "intervals_correct": 0, "unresolvable": 0,
           "notes_with_failure": 0, "residual_date_surfaces": 0}
    failures: list[dict] = []
    residuals: list[str] = []
    for text in texts:
        r = check_intervals(text, policy)
        agg["notes"] += 1
        agg["annotations_checked"] += r["annotations_checked"]
        agg["intervals_correct"] += r["intervals_correct"]
        agg["unresolvable"] += r["unresolvable"]
        agg["residual_date_surfaces"] += len(r["residual_date_surfaces"])
        residuals.extend(r["residual_date_surfaces"])
        if r["failures"]:
            agg["notes_with_failure"] += 1
            failures.extend(r["failures"][:3])

    n = agg["annotations_checked"]
    print(f"interval preservation: {agg['intervals_correct']}/{n} exact "
          f"({agg['intervals_correct'] / n:.4f})" if n else
          "interval preservation: no annotated intervals found -- check dates.mode")
    print(f"  notes={agg['notes']}  unverifiable={agg['unresolvable']}  "
          f"notes_with_failure={agg['notes_with_failure']}")
    print(f"  residual date surfaces in masked text: {agg['residual_date_surfaces']} "
          f"(must be 0 for Safe Harbor)")
    if residuals:
        # Printed, not just counted: a residual is either a real leak the date policy missed or
        # a false positive in this very check, and the two are told apart by looking at them.
        print(f"  residual samples: {sorted(set(residuals))[:12]}")
    for f in failures[:5]:
        print(f"  FAIL {f['target']} vs {f['anchor']}: claimed {f['claimed_days']}d, "
              f"true {f['true_days']}d")

    out = {"intervals": agg, "interval_failures": failures,
           "residual_samples": sorted(set(residuals))[:40]}
    if a.llm:
        qa = [q for t in texts[: a.llm_notes] if (q := qa_agreement(t, policy)) is not None]
        if qa:
            scored = sum(q["scored"] for q in qa)
            print(f"\nQA agreement over {len(qa)} notes, {scored} scored answers: "
                  f"mean similarity "
                  f"{sum(q['mean_similarity'] * q['scored'] for q in qa) / scored:.4f}, "
                  f"{sum(q['agree_at_0.80'] for q in qa)}/{scored} agree at 0.80")
            out["qa"] = qa

    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
