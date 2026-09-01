"""Stage E: entity resolution.

Consistent pseudonymisation only works if the same entity collapses to the same
placeholder. ``Mr. Wood``, ``Wood``, ``Wood's`` and ``J. Wood`` must all become
``[NAME_PATIENT_1]``, or the downstream LLM loses co-reference -- and co-reference is
most of what makes a de-identified note still usable.

Deliberately conservative. A missed merge costs a little utility (one person becomes two
placeholders). A wrong merge corrupts rehydration and can attribute one patient's facts
to another. So thresholds are set to under-merge, and clustering never crosses category
boundaries: a patient and a provider who share a surname stay separate people.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..types import Category, Span

try:  # rapidfuzz is a hard dependency, but degrade rather than crash the demo.
    from rapidfuzz.fuzz import ratio as _ratio
except ImportError:  # pragma: no cover
    from difflib import SequenceMatcher

    def _ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio() * 100.0


_NAME_CATEGORIES = frozenset(
    {Category.NAME_PATIENT, Category.NAME_PROVIDER, Category.NAME_OTHER}
)

#: Categories compared on their alphanumeric content alone.
_ALNUM_CATEGORIES = frozenset(
    {
        Category.PHONE,
        Category.FAX,
        Category.SSN,
        Category.MRN,
        Category.ACCOUNT,
        Category.HEALTH_PLAN_ID,
        Category.LICENCE,
        Category.VEHICLE,
        Category.DEVICE,
        Category.GEO_ZIP,
        Category.ID_GENERIC,
    }
)

_TITLE_STRIP = re.compile(
    r"(?i)^\s*(?:dr|doctor|mr|mrs|ms|miss|mx|prof|professor|rev|sir|dame|nurse|"
    r"sister|father|fr|sgt|capt|lt|col|maj)\.?\s+"
)
_CRED_STRIP = re.compile(
    r"(?i)[,\s]+(?:m\.?d|d\.?o|mbbs|r\.?n|l\.?p\.?n|n\.?p|p\.?a-?c?|pharm\.?d|r\.?ph|"
    r"ph\.?d|d\.?d\.?s|d\.?m\.?d|d\.?p\.?m|o\.?d|d\.?v\.?m|c\.?r\.?n\.?a|c\.?n\.?m|"
    r"l\.?c\.?s\.?w|m\.?s\.?w|r\.?r\.?t|d\.?p\.?t|m\.?s\.?n|b\.?s\.?n|a\.?p\.?r\.?n|"
    r"f\.?a\.?c\.?[spc])\.?\s*$"
)
_POSSESSIVE = re.compile(r"(?:'s|’s|s')$")

#: Fuzzy threshold for treating two names as the same person. High on purpose.
_FUZZY_MIN = 92.0


@dataclass
class Cluster:
    """One real-world entity and every span that referred to it."""

    category: Category
    spans: list[Span] = field(default_factory=list)
    keys: set[str] = field(default_factory=set)

    @property
    def canonical(self) -> str:
        """Longest surface form -- the most complete version of the entity, and what
        rehydration inserts."""
        return max((s.text for s in self.spans), key=len, default="")

    @property
    def surfaces(self) -> list[str]:
        seen: list[str] = []
        for s in self.spans:
            if s.text not in seen:
                seen.append(s.text)
        return seen

    @property
    def first_offset(self) -> int:
        return min((s.start for s in self.spans), default=0)


def normalise(category: Category, text: str) -> str:
    if category in _ALNUM_CATEGORIES:
        return re.sub(r"[^A-Za-z0-9]", "", text).upper()
    if category in _NAME_CATEGORIES:
        return _normalise_name(text)
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s'’\-]", " ", text)).strip().lower()


def _normalise_name(text: str) -> str:
    t = text.strip()
    t = _TITLE_STRIP.sub("", t)
    t = _CRED_STRIP.sub("", t)
    t = _POSSESSIVE.sub("", t.strip())
    # "SMITH, JOHN A" and "John A Smith" must normalise alike, so drop the ordering.
    t = re.sub(r"[^\w\s'’\-]", " ", t)
    tokens = [tok.lower() for tok in t.split() if tok]
    # Single-letter initials carry no matching signal and break token comparison.
    tokens = [tok for tok in tokens if len(tok.strip(".'’-")) > 1]
    return " ".join(sorted(tokens))


def _name_tokens(key: str) -> list[str]:
    return [t for t in key.split() if t]


def _names_match(a: str, b: str) -> bool:
    """Whether two normalised names refer to the same person."""
    if not a or not b:
        return False
    if a == b:
        return True

    ta, tb = _name_tokens(a), _name_tokens(b)
    if not ta or not tb:
        return False

    # A bare surname joins the fuller form: "wood" -> "john wood".
    if len(ta) == 1 or len(tb) == 1:
        short, long = (ta, tb) if len(ta) == 1 else (tb, ta)
        if short[0] in long:
            return True
        # Tolerate a misspelled single token against any token of the longer name.
        return any(_ratio(short[0], tok) >= _FUZZY_MIN for tok in long)

    # Two multi-token names: require genuine similarity. "john wood" vs "jane wood"
    # must NOT merge, so a shared surname alone is not enough.
    if set(ta) == set(tb):
        return True
    if set(ta) <= set(tb) or set(tb) <= set(ta):
        return True
    return _ratio(a, b) >= _FUZZY_MIN


def cluster_spans(spans: list[Span]) -> list[Cluster]:
    """Group spans that refer to the same entity. Order of first appearance is preserved
    so placeholder numbering reads naturally through the document."""
    clusters: list[Cluster] = []

    for span in sorted(spans, key=lambda s: (s.start, s.end)):
        key = normalise(span.category, span.text)
        if not key:
            key = span.text.strip().lower()

        target: Cluster | None = None
        for c in clusters:
            if c.category is not span.category:
                continue
            if key in c.keys:
                target = c
                break
            if span.category in _NAME_CATEGORIES and any(
                _names_match(key, existing) for existing in c.keys
            ):
                target = c
                break
        if target is None:
            target = Cluster(category=span.category)
            clusters.append(target)

        target.spans.append(span)
        target.keys.add(key)

    return clusters
