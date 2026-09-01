"""Stage F: policy application -- turn merged spans into masked text plus a reversible map.

Masking strategy is **typed consistent pseudonyms**: ``[NAME_PATIENT_1]``, and the same
entity gets the same token everywhere. Three reasons this beats both flat redaction and
surrogate generation:

* Co-reference survives, so the downstream LLM can still tell who did what to whom.
* Rehydration is deterministic -- no alignment guessing.
* **A reviewer can eyeball the masked text for leaks.** Surrogate generation makes that
  impossible: a missed real name hides perfectly among the fake ones. Since the assessment
  is "we will read your masked output hunting for leaks", auditability is a feature.
"""

from __future__ import annotations

import re

from ..config import Policy
from ..resolve.cluster import cluster_spans
from ..types import Category, Mapping, MappingEntry, ReviewItem, Span
from . import ages as ages_policy
from . import dates as dates_policy

_AGE_CATEGORIES = frozenset({Category.AGE, Category.AGE_OVER_89})


def _zip_replacement(text: str, index: int, policy: Policy) -> str:
    """ZIP handling per Safe Harbor: the initial three digits may stay, *except* for the
    17 three-digit prefixes covering fewer than 20,000 people, which must go entirely."""
    digits = re.sub(r"\D", "", text)
    zip3 = digits[:3]
    if (
        policy.geo.zip_strategy != "truncate3"
        or len(zip3) < 3
        or zip3 in policy.geo.restricted_zip3
    ):
        return f"[GEO_ZIP_{index}]"
    return f"[GEO_ZIP_{index} = {zip3}**]"


def apply(
    text: str,
    spans: list[Span],
    policy: Policy,
    patient_key: str = "",
) -> tuple[str, Mapping, list[ReviewItem]]:
    date_spans = [s for s in spans if s.category is Category.DATE and policy.masks(s.category)]
    date_labels, shift = dates_policy.plan_dates(
        date_spans,
        policy.dates,
        patient_key,
        year_is_identifying=ages_policy.year_is_identifying,
    )

    other = [
        s
        for s in spans
        if s.category is not Category.DATE
        and s.category not in _AGE_CATEGORIES
        and policy.masks(s.category)
    ]
    clusters = cluster_spans(other)

    mapping = Mapping(
        patient_key=patient_key,
        date_shift_days=shift,
        date_mode=policy.dates.mode,
    )
    counters: dict[Category, int] = {}
    #: (start, end, replacement)
    edits: list[tuple[int, int, str]] = []

    def _record(placeholder: str, category: Category, canonical: str, span_list: list[Span]) -> None:
        entry = mapping.entries.get(placeholder)
        if entry is None:
            entry = MappingEntry(
                placeholder=placeholder,
                category=category,
                canonical=canonical,
                surfaces=[],
                offsets=[],
            )
            mapping.entries[placeholder] = entry
        for s in span_list:
            if s.text not in entry.surfaces:
                entry.surfaces.append(s.text)
            entry.offsets.append((s.start, s.end))

    # --- dates ------------------------------------------------------------------
    for s in date_spans:
        label = date_labels.get(dates_policy.value_key(s.text))
        if label is None:
            continue
        edits.append((s.start, s.end, label.replacement))
        _record(label.replacement, Category.DATE, label.canonical, [s])

    # --- ages -------------------------------------------------------------------
    # [AGE_OVER_89] is deliberately many-to-one: bucketing is lossy, so two different ages
    # over 89 share one token and rehydration restores the first. Reversibility of an age
    # we are required to suppress is not worth a second placeholder that re-splits the bucket.
    for s in spans:
        if s.category not in _AGE_CATEGORIES:
            continue
        rep = ages_policy.replacement(s, policy.ages)
        if rep is None:      # retained: an age under 90 is not an identifier
            continue
        edits.append((s.start, s.end, rep))
        _record(rep, s.category, s.text, [s])

    # --- everything else --------------------------------------------------------
    for cluster in sorted(clusters, key=lambda c: c.first_offset):
        cat = cluster.category
        counters[cat] = counters.get(cat, 0) + 1
        index = counters[cat]

        if cat is Category.GEO_ZIP:
            placeholder = _zip_replacement(cluster.canonical, index, policy)
        else:
            placeholder = f"[{cat.value}_{index}]"

        _record(placeholder, cat, cluster.canonical, cluster.spans)
        for s in cluster.spans:
            edits.append((s.start, s.end, placeholder))

    masked = _splice(text, edits)
    review = _review_queue(spans, mapping, policy)
    return masked, mapping, review


def _splice(text: str, edits: list[tuple[int, int, str]]) -> str:
    """Apply replacements right-to-left so earlier offsets stay valid."""
    out = text
    for start, end, rep in sorted(edits, key=lambda e: e[0], reverse=True):
        out = out[:start] + rep + out[end:]
    return out


def _review_queue(spans: list[Span], mapping: Mapping, policy: Policy) -> list[ReviewItem]:
    """Spans in the uncertainty band. They are masked regardless -- recall asymmetry means
    we never withhold a mask pending review -- but a human gets told which ones were
    borderline, which is what makes the precision cost auditable instead of invisible."""
    if not policy.review_queue.enabled:
        return []
    items: list[ReviewItem] = []
    offset_index = {
        off: entry.placeholder
        for entry in mapping.entries.values()
        for off in entry.offsets
    }
    for s in spans:
        if not (policy.review_queue.low <= s.score < policy.review_queue.high):
            continue
        placeholder = offset_index.get((s.start, s.end), "")
        if not placeholder:
            continue
        items.append(
            ReviewItem(
                span=s,
                placeholder=placeholder,
                reason=f"score {s.score:.2f} in review band "
                f"[{policy.review_queue.low}, {policy.review_queue.high}) "
                f"via {s.detector or s.source.value}",
            )
        )
    return items
