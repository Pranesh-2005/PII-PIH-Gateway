"""Surface propagation: mask every *other* occurrence of something we already identified.

The single biggest recall win available to a rules-only system, and the fix for the failure
mode that would otherwise dominate: ``Mr. Wood`` is caught by the title cue, but the bare
``Wood`` three sentences later has no cue at all and walks straight out of the building.

Once *any* detector has established that "Wood" names a person in this document, every other
occurrence in the same document is the same identifier. So we sweep for it. That is both
sound (the evidence was already gathered) and cheap (one regex per surface).

Two guards, because propagation is exactly where over-redaction creeps in:

* **The eponym guard.** Propagating the token ``Parkinson`` out of ``Dr. Parkinson`` must not
  touch ``Parkinson's disease``. Every candidate occurrence is re-checked.
* **Casing.** Matching is case-insensitive so ``WOOD`` in a header finds ``Wood`` in the prose,
  but a hit only counts if it *looks* like a name occurrence -- initial capital or all caps.
  Otherwise "wood" the material starts getting redacted.
"""

from __future__ import annotations

import re
from dataclasses import replace

from ..detectors.rules import is_eponym_use
from ..types import Category, Span

_NAME_CATEGORIES = frozenset(
    {Category.NAME_PATIENT, Category.NAME_PROVIDER, Category.NAME_OTHER}
)

#: Categories where individual tokens of a multi-word surface are worth propagating.
#: Splitting an address or an organisation apart would mask "Road" and "Health"; splitting a
#: name gives you the surname, which is the whole point.
_TOKENISE = _NAME_CATEGORIES

_MIN_SURFACE = 3
_MIN_TOKEN = 4

#: Name tokens too generic to chase on their own.
_STOP_TOKENS = frozenset(
    {
        "mister", "madam", "junior", "senior", "patient", "father", "mother", "sister",
        "brother", "spouse", "wife", "husband", "daughter", "son", "child", "family",
        "unknown", "none", "male", "female", "adult", "minor",
    }
)


def _overlaps(start: int, end: int, taken: list[tuple[int, int]]) -> bool:
    return any(s < end and start < e for s, e in taken)


def _variants(span: Span) -> set[str]:
    out = {span.text.strip()}
    if span.category in _TOKENISE:
        for tok in re.split(r"[\s,]+", span.text):
            tok = tok.strip(".'’-")
            if len(tok) >= _MIN_TOKEN and tok.lower() not in _STOP_TOKENS:
                out.add(tok)
    return {v for v in out if len(v) >= _MIN_SURFACE}


def propagate(text: str, spans: list[Span]) -> list[Span]:
    """Extra spans for repeat occurrences of already-detected surfaces.

    Returned spans are additive; feed them back through ``merge`` so overlap arbitration and
    thresholds apply uniformly.
    """
    taken: list[tuple[int, int]] = [(s.start, s.end) for s in spans]
    extra: list[Span] = []

    for span in spans:
        is_name = span.category in _NAME_CATEGORIES
        for variant in _variants(span):
            pattern = re.compile(rf"(?<![\w\-]){re.escape(variant)}(?![\w\-])", re.IGNORECASE)
            for m in pattern.finditer(text):
                start, end = m.span()
                if _overlaps(start, end, taken):
                    continue
                surface = m.group(0)
                # Case-insensitive search, case-sensitive acceptance: a lowercase hit is a
                # different word, not a different spelling of the name.
                if surface[0].islower() and not variant[0].islower():
                    continue
                if is_name and is_eponym_use(text, start, end):
                    continue
                extra.append(
                    replace(
                        span,
                        start=start,
                        end=end,
                        text=surface,
                        detector=f"propagated:{span.detector or span.source.value}",
                    )
                )
                taken.append((start, end))

    return extra
