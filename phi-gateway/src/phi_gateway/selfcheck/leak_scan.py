"""Stage G: the leak self-check.

The brief says the panel will read the masked output hunting for leaks, and to assume they
will find any that exist. The response is to be our own adversary: re-scan the text we are
about to emit, at a *lower* threshold than we detected with, and refuse to forward on a hit.

Two independent checks, because they fail differently:

1. **Re-detection.** Run the rule detectors over the masked text with the paranoid
   threshold. Anything found is, by construction, an identifier we shipped.
2. **Surface residue.** Assert that no surface form we recorded in the mapping still appears
   literally in the masked text. This catches merge/splice bugs that no detector would --
   e.g. a cluster whose second mention was trimmed away during arbitration.

Cost is one extra regex pass over one note. Cheap, and it converts a silent breach into a
loud failure.
"""

from __future__ import annotations

import re

from ..config import Policy
from ..detectors import rules
from ..detectors.merge import is_over_redaction
from ..detectors.structural import segment
from ..types import Category, LeakFinding, LeakReport, Mapping, Source, Span

#: Placeholders look like [NAME_PATIENT_1], [DATE_2 = DATE_1 + 3d], [GEO_ZIP_1 = 021**] or
#: the un-numbered [AGE_OVER_89]. The trailing index is required so that ordinary bracketed
#: prose in an LLM response ("[sic]", "[IMPORTANT]") is left alone -- the pattern also drives
#: the unknown-placeholder guard in ``pipeline.rehydrate``, which strips whatever it matches.
PLACEHOLDER_RE = re.compile(
    r"\[(?:[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*_\d+(?:\s*=\s*[^\]\n]{1,40})?|AGE_OVER_89|AGE|UNKNOWN_PLACEHOLDER)\]"
)
_PLACEHOLDER = PLACEHOLDER_RE

#: Retained-by-policy categories are not leaks. Everything else is.
def _scannable(policy: Policy) -> list[Category]:
    return [c for c in Category if policy.masks(c)]


def blank_placeholders(text: str, mapping: Mapping | None = None) -> str:
    """Replace placeholders with an equal-length run of NULs, preserving every offset so
    findings point at the real position in the masked text.

    NUL rather than a space, because a space *joins* what the placeholder separated: blanking
    "CLAIM [ACCOUNT_1]   PRODUCED" to spaces makes the account cue adjacent to "PRODUCED",
    and "Dr. [NAME_PROVIDER_3]" adjacent to whatever word follows. Every such match is a
    phantom leak that blocks a forward on text that is actually clean. NUL matches no
    ``\\w``, no ``\\s`` and no character class in any detector, so a cue can no longer reach
    across a redaction, while ``\\b`` still fires on both sides of it.

    Also blanks literal replacements recorded in the mapping. That matters in ``shifted``
    date mode, where the emitted value is a real-looking date -- deliberately, so it must
    not be reported as our own leak.
    """
    out = _PLACEHOLDER.sub(lambda m: "\x00" * (m.end() - m.start()), text)
    for placeholder in sorted(
        (mapping.entries if mapping else {}), key=len, reverse=True
    ):
        if _PLACEHOLDER.fullmatch(placeholder):
            continue
        out = out.replace(placeholder, "\x00" * len(placeholder))
    return out


def scan(masked_text: str, mapping: Mapping, policy: Policy) -> LeakReport:
    if not policy.selfcheck.enabled:
        return LeakReport(clean=True)

    findings: list[LeakFinding] = []
    scrubbed = blank_placeholders(masked_text, mapping)
    threshold = policy.selfcheck.paranoid_threshold

    segments = segment(scrubbed)
    for span in rules.detect_categories(scrubbed, _scannable(policy), segments):
        if span.score < threshold:
            continue
        # The same predicate the merge stage used when it decided this was not an identifier.
        # Stage G used to apply only the eponym guard, and only for name categories, so every
        # over-redaction merge correctly declined came back here as a "leak": four fixes to
        # merge raised this report from five findings to twelve, all of them phantom. An auditor
        # that disagrees with the emitter about what an identifier is blocks clean documents,
        # which is the same failure as leaking, pointed the other way.
        if is_over_redaction(scrubbed, span):
            continue
        findings.append(
            LeakFinding(
                category=span.category,
                text=span.text,
                start=span.start,
                end=span.end,
                detector=f"rescan:{span.detector or span.source.value}",
            )
        )

    for entry in mapping.entries.values():
        for surface in entry.surfaces:
            s = surface.strip()
            if len(s) < 3:
                continue          # too short to search for without matching everything
            for m in re.finditer(rf"(?<!\w){re.escape(s)}(?!\w)", scrubbed):
                # Same predicate again, over a span reconstructed at the residue site. "Dr.
                # Parkinson" is masked, so "Parkinson" is a mapped surface -- but the retained
                # "Parkinson's disease" three words later is not residue of it, and neither is
                # the word "operative" on a later page after one was wrongly masked as ``ORG``.
                # Without this guard the self-check blocks every note containing both senses,
                # which is the exact case the brief names.
                probe = Span(start=m.start(), end=m.end(), category=entry.category, text=s,
                             score=1.0, source=Source.RULES, detector="residue")
                if is_over_redaction(scrubbed, probe):
                    continue
                findings.append(
                    LeakFinding(
                        category=entry.category,
                        text=s,
                        start=m.start(),
                        end=m.end(),
                        detector="residue:mapped-surface-still-present",
                    )
                )
                break

    return LeakReport(clean=not findings, findings=findings)
