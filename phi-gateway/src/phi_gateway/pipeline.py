"""The gateway pipeline. Stages A-G, then the reverse direction.

Public API is exactly what the brief specifies::

    deidentify(text) -> (masked_text, mapping)
    rehydrate(response, mapping) -> text

``deidentify_full`` returns the whole ``DeidResult`` (spans, review queue, leak report,
timings) for the API, the CLI and the eval harness.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time

from .config import Policy, load_policy
from .detectors import fields as field_detector
from .detectors import rules
from .detectors import structural
from .detectors.merge import merge
from .policy import placeholders
from .resolve.propagate import propagate
from .selfcheck import leak_scan
from .selfcheck.leak_scan import PLACEHOLDER_RE
from .types import DeidResult, Mapping, Span

log = logging.getLogger("phi_gateway")

_DEFAULT_POLICY: Policy | None = None


def default_policy() -> Policy:
    global _DEFAULT_POLICY
    if _DEFAULT_POLICY is None:
        _DEFAULT_POLICY = load_policy()
    return _DEFAULT_POLICY


def derive_patient_key(text: str) -> str:
    """Stable per-note key for the date offset.

    Derived from the note itself so two requests for the same note agree, and so no caller
    has to invent a patient identifier just to de-identify. Pass an explicit
    ``patient_key`` when notes for one patient must share an offset.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _neural_spans(text: str, segments, policy: Policy) -> list[Span]:
    """Stage C. Absent until Phase 3 trains a tagger; the gateway is fully functional
    without it, which is the point of building in this order."""
    if not policy.neural.enabled:
        return []
    try:
        from .detectors import neural
    except ImportError as exc:
        log.warning("neural stage enabled but unavailable (%s); continuing rules-only", exc)
        return []
    try:
        return neural.detect(text, segments, policy)
    except Exception as exc:                      # never let the model crash the gateway
        log.error("neural stage failed (%s); continuing rules-only", exc)
        return []


def deidentify_full(
    text: str,
    *,
    patient_key: str | None = None,
    policy: Policy | None = None,
) -> DeidResult:
    pol = policy or default_policy()
    key = patient_key or derive_patient_key(text)
    t0 = time.perf_counter()

    segments = structural.segment(text)                        # A
    form_fields = structural.form_fields(text)
    candidates: list[Span] = []
    candidates += rules.detect(text, segments)                 # B
    candidates += field_detector.detect(text, segments, form_fields)
    candidates += _neural_spans(text, segments, pol)           # C
    t_detect = time.perf_counter()

    spans = merge(text, candidates, pol)                       # D
    # Sweep for repeat occurrences of anything already identified, then re-arbitrate so the
    # new spans go through the same thresholds and overlap rules.
    extra = propagate(text, spans)
    if extra:
        spans = merge(text, spans + extra, pol)
    masked, mapping, review = placeholders.apply(text, spans, pol, key)   # E + F
    t_mask = time.perf_counter()

    report = leak_scan.scan(masked, mapping, pol)              # G
    blocked = report.leaked and pol.selfcheck.on_leak == "block"
    if report.leaked:
        log.error(
            "SELF-CHECK LEAK: %d finding(s), on_leak=%s: %s",
            len(report.findings),
            pol.selfcheck.on_leak,
            [f"{f.category.value}:{f.text!r}" for f in report.findings[:5]],
        )
    t_end = time.perf_counter()

    return DeidResult(
        masked_text=masked,
        mapping=mapping,
        spans=spans,
        segments=segments,
        review_queue=review,
        leak_report=report,
        blocked=blocked,
        stats={
            "candidates": len(candidates),
            "date_mode": pol.dates.mode,
            "ms_detect": round((t_detect - t0) * 1000, 2),
            "ms_mask": round((t_mask - t_detect) * 1000, 2),
            "ms_selfcheck": round((t_end - t_mask) * 1000, 2),
            "ms_total": round((t_end - t0) * 1000, 2),
        },
    )


def deidentify(
    text: str,
    *,
    patient_key: str | None = None,
    policy: Policy | None = None,
) -> tuple[str, Mapping]:
    """Required signature: raw clinical text in, masked text and reversible mapping out."""
    result = deidentify_full(text, patient_key=patient_key, policy=policy)
    return result.masked_text, result.mapping


def rehydrate(response: str, mapping: Mapping) -> str:
    """Required signature: substitute placeholders back to real values.

    **Only** placeholders present in this mapping are substituted. Any other
    placeholder-shaped token in the response was invented by the LLM -- it maps to no real
    value, so substituting it is impossible and echoing it is a small injection vector. It
    is replaced with a visible marker and logged as a security event. This is the direct
    answer to "what happens if the LLM echoes a token that was never in the input".
    """
    out = response
    # Longest first: no placeholder can be eaten as a prefix of a longer one.
    for placeholder in sorted(mapping.entries, key=len, reverse=True):
        out = out.replace(placeholder, mapping.entries[placeholder].canonical)

    unknown: list[str] = []

    def _strip(m: re.Match[str]) -> str:
        unknown.append(m.group(0))
        return "[UNKNOWN_PLACEHOLDER]"

    out = PLACEHOLDER_RE.sub(_strip, out)
    if unknown:
        log.error(
            "SECURITY: LLM response contained %d placeholder(s) absent from the mapping, "
            "stripped: %s",
            len(unknown),
            unknown[:10],
        )
    return out
