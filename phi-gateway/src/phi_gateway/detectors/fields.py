"""Form-field cue detector.

The largest recall win available without a model, and the one that covers what P2.7
explicitly demands be in the test set: "identifiers hiding in tables, headers and
signature blocks".

The idea is simple. In ``Attending Physician: SMITH, JOHN A MD`` the *key* tells us the
value is a provider name, so we do not need to recognise the name at all -- which is why
this detector handles ALL-CAPS names, transposed names, misspellings and initials that
defeat every name-shaped regex.

The corresponding danger is masking clinical content: ``Diagnosis: Parkinson's disease``
has the same key/value shape. ``FIELD_KEY_ALLOWLIST`` is checked first and wins, because
over-redaction is the other half of the failure the brief describes.
"""

from __future__ import annotations

import re

from ..lexicons import CREDENTIAL_ALT, FIELD_KEY_ALLOWLIST, FIELD_KEY_CATEGORY
from ..types import Category, Segment, Source, Span
from .structural import FormField, form_fields, in_structured_segment

_ALLOWLIST = tuple(re.compile(p, re.IGNORECASE) for p in FIELD_KEY_ALLOWLIST)

_NAME_CATEGORIES = frozenset(
    {Category.NAME_PATIENT, Category.NAME_PROVIDER, Category.NAME_OTHER}
)

#: The credential that ends a provider name. Whatever follows it is the specialty, not the
#: person. Searched, not anchored -- "WHITFIELD, MARCUS D." has a comma inside the *name*,
#: so cutting at the first comma would throw the given name away.
_NAME_VALUE_CRED = re.compile(
    rf",\s*(?:{CREDENTIAL_ALT})\.?(?![A-Za-z])", re.IGNORECASE
)
_CATEGORY_KEYS = tuple(
    (re.compile(p, re.IGNORECASE), Category(c)) for p, c in FIELD_KEY_CATEGORY
)

#: Values that carry no identifier, whatever the key says.
_NULL_VALUES = frozenset(
    {
        "", "-", "--", "---", "?", "n/a", "na", "n.a.", "none", "nil", "nka", "nkda",
        "unknown", "unk", "not applicable", "not available", "not documented",
        "not recorded", "not provided", "not given", "not specified", "not stated",
        "pending", "deferred", "tbd", "to be determined", "see below", "see above",
        "as above", "as below", "same", "same as above", "withheld", "redacted",
        "declined", "refused", "no", "yes", "y", "n", "0", "none reported",
    }
)

#: Keys where the generic fallback applies, so confidence is a notch lower.
_GENERIC_KEYS = frozenset({"name"})


def _normalise_key(key: str) -> str:
    k = re.sub(r"\([^)]*\)", " ", key)          # drop "(years)", "(mm/dd/yyyy)"
    k = k.replace(".", " ").replace("_", " ")
    k = re.sub(r"[^A-Za-z0-9#/&'\- ]+", " ", k)
    k = re.sub(r"\s+", " ", k).strip().lower()
    return k


def _classify_key(key_norm: str) -> tuple[Category | None, float]:
    """Map a field key to a category. Allowlist wins; specific keys beat generic ones."""
    if not key_norm:
        return None, 0.0

    for pat in _ALLOWLIST:
        if pat.fullmatch(key_norm):
            return None, 0.0

    for pat, cat in _CATEGORY_KEYS:
        if pat.fullmatch(key_norm):
            score = 0.85 if pat.pattern in _GENERIC_KEYS else 0.95
            return cat, score

    # Prefix fallback: "Admission Date/Time", "MRN / Chart".
    for pat in _ALLOWLIST:
        m = pat.match(key_norm)
        if m and len(key_norm) - m.end() <= 8:
            return None, 0.0
    for pat, cat in _CATEGORY_KEYS:
        m = pat.match(key_norm)
        if m and len(key_norm) - m.end() <= 8:
            return cat, 0.88
    return None, 0.0


def _is_null_value(value: str) -> bool:
    v = value.strip().strip(".,;").lower()
    return v in _NULL_VALUES or not re.search(r"[A-Za-z0-9]", v)


def detect(
    text: str,
    segments: list[Segment] | None = None,
    fields: list[FormField] | None = None,
) -> list[Span]:
    segs = segments or []
    ffs = fields if fields is not None else form_fields(text)
    out: list[Span] = []

    for f in ffs:
        cat, score = _classify_key(_normalise_key(f.key))
        if cat is None:
            continue
        if _is_null_value(f.value):
            continue

        start, end = f.value_start, f.value_end
        surface = text[start:end]
        # Trim trailing punctuation that is part of the layout, not the value.
        trimmed = surface.rstrip(" \t.,;:|")
        if not trimmed:
            continue
        end = start + len(trimmed)

        # A provider field carries the name *and* the specialty: "G. Halloway, MD,
        # Orthopedic Spine Surgery". The credential ends the name; the specialty after it is
        # clinical content, and masking it costs utility twice over -- once for the lost
        # specialty, and again in Stage E, where "spine" then single-token-merges into the
        # provider's cluster and drags every unrelated "Spine" in the document with it.
        if cat in _NAME_CATEGORIES:
            m = _NAME_VALUE_CRED.search(trimmed)
            if m:
                trimmed = trimmed[: m.end()].rstrip(" \t.,;:|")
                end = start + len(trimmed)

        out.append(
            Span(
                start=start,
                end=end,
                category=cat,
                text=trimmed,
                score=score,
                source=Source.FIELD,
                detector=f"field:{_normalise_key(f.key)[:28]}",
                in_structured_segment=in_structured_segment(start, segs) or f.in_table,
            )
        )
    return out
