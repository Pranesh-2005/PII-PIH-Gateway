"""Stage D: merge and arbitration.

Three detector layers (rules, form-field cues, neural tagger) produce overlapping,
disagreeing spans. This stage resolves them **recall-first**: nothing is silently
dropped, only trimmed. If a name span and an organisation span fight over the token
"Wood", the loser keeps whatever remainder it still covers rather than vanishing --
because a dropped span is a leak, and a trimmed one is at worst an odd-looking mask.

Priority order, per category:

* ``FIELD`` always wins. A form-field key naming its own value ("Attending Physician:")
  is stronger evidence than any pattern or model score.
* Structured identifiers (SSN, phone, dates, ...): ``RULES`` beats ``NEURAL``. A verified
  checksum is worth more than a probability.
* Names, geography, organisations: ``NEURAL`` beats ``RULES``. This is the half of the
  problem rules cannot do, which is why a model is being trained at all.

**Why every over-redaction guard lives here and not in the detector that needs it.**
Six guards in this package were written correctly and installed in the wrong place -- inside
the rule detector that motivated them. Each was then bypassed the moment the tagger was
switched on, because a model can emit *any* category at *any* offset, and a guard reached only
via one detector's code path or only for one set of categories is not a guard:

===============================  ==================================  =========================
guard                            bypassed as                         was gated by
===============================  ==================================  =========================
``rules.is_clinical_label``      ``[NAME_PATIENT_5]  7/10``          call site (rules only)
``rules.is_eponym_use``          ``[DEVICE_1]'s disease``            ``_NAME_CATEGORIES``
``validators.is_clinical_code``  ``ICD-[DATE_1]``                    call site (rules only)
``rules.is_bare_facility``       ``[ORG_1] Center``                  call site (rules only)
``rules.is_credential_only``     ``[NAME_PROVIDER_10]`` = ", DPT"    call site (rules only)
``_snap`` (word boundaries)      ``PRO[NAME_PATIENT_1] NOTE``        ran only after a subtract
===============================  ==================================  =========================

``merge`` is the one point every span from every source passes through, so it is the only place
a guard is actually universal. The cost of getting this wrong is not cosmetic: masking the word
"Pain" deleted the cue the date guard looks backwards for, so fourteen pain scores were then
reported as surviving dates and Stage G blocked a 22-page record outright.

**And the cost of getting an over-redaction guard *too wide* is a leak, not a cosmetic loss.**
Two guards here are deliberately narrower than their name suggests, both for the same reason.
``_code_fragment`` fires only when the span is a strict fragment of a code token, because
``is_clinical_code`` matches ``^\\d{5}$`` and so does every ZIP. ``_bare_facility_span`` fires
only when the surrounding capitalised run is *also* pure facility vocabulary, because the first
version dropped the ``ORTHOPEDIC SPINE INSTITUTE`` half of ``HALLOWAY ORTHOPEDIC SPINE
INSTITUTE`` and published a readable organisation. A guard that removes a mask from part of a
real identifier has done more damage than the false positive it was written to prevent.
"""

from __future__ import annotations

import re
from dataclasses import replace

from ..config import Policy
from ..types import Category, Source, Span
from .rules import is_bare_facility, is_clinical_label, is_credential_only, is_eponym_use
from .validators import is_clinical_code

#: Categories where a validated pattern outranks a model score.
STRUCTURED_ID_CATEGORIES: frozenset[Category] = frozenset(
    {
        Category.SSN,
        Category.PHONE,
        Category.FAX,
        Category.EMAIL,
        Category.URL,
        Category.IP,
        Category.MRN,
        Category.ACCOUNT,
        Category.HEALTH_PLAN_ID,
        Category.LICENCE,
        Category.VEHICLE,
        Category.DEVICE,
        Category.GEO_ZIP,
        Category.DATE,
        Category.AGE,
        Category.AGE_OVER_89,
        Category.ID_GENERIC,
    }
)

#: Categories the eponym guard applies to: **all of them.** It used to be names only, on the
#: reasoning that an organisation called "Parkinson Clinic" is still an identifier. That
#: reasoning was right about organisations and wrong about the guard: ``is_eponym_use`` already
#: requires the span's last token to be a known eponym *and* a disease context around it, so
#: "Parkinson Clinic" never matched it in the first place. Meanwhile the tagger labelled the
#: "Parkinson" of "Parkinson's disease" as ``DEVICE``, which walked straight through a
#: names-only gate and produced ``[DEVICE_1]'s disease``. A category gate on a guard is only
#: ever as good as the guess about which label a model will pick.

_MIN_TRIMMED_LEN = 2


def _rank(span: Span) -> int:
    if span.source is Source.FIELD:
        return 3
    if span.category in STRUCTURED_ID_CATEGORIES:
        return 2 if span.source is Source.RULES else 1
    return 2 if span.source is Source.NEURAL else 1


def _priority(span: Span) -> tuple[int, float, int]:
    return (_rank(span), span.score, len(span))


def _passes_threshold(span: Span, policy: Policy) -> bool:
    return span.score >= policy.threshold(
        span.category, in_structured_segment=span.in_structured_segment
    )


def _snap(span: Span, text: str, min_len: int = 1) -> Span | None:
    """Snap a span to word boundaries and drop it if nothing real is left.

    Applied to **every** accepted span, not only to overlap-trimmed remainders. It used to run
    only after a subtraction, which meant a detector that emitted a mid-word span had it passed
    through verbatim -- and the tagger does exactly that. Observed on a real record:

    * ``PRO[NAME_PATIENT_1] NOTE`` -- "PROGRESS" cut in half, "GRESS" masked as a name.
    * ``Dr[NAME_PROVIDER_1][NAME_PROVIDER_2]`` -- the ". " between title and name masked.
    * ``6[DATE_1]7/10`` -- the hyphen of a pain-score range masked as a date.

    Each is a garbled document *and* a spurious placeholder in the mapping. Rejecting spans
    with no alphanumeric content at all is what kills the punctuation-only cases; snapping is
    what kills the mid-word ones.

    ``min_len`` stays at 1 for ordinary spans -- a single character can be a real initial --
    and is raised only for overlap remainders, where a 1-character leftover is by construction
    a fragment of something already masked.
    """
    start, end = span.start, span.end
    while start < end and start > 0 and text[start - 1].isalnum() and text[start].isalnum():
        start += 1
    while end > start and end < len(text) and text[end].isalnum() and text[end - 1].isalnum():
        end -= 1

    surface = text[start:end]
    stripped = surface.strip()
    if len(stripped) < min_len or not any(c.isalnum() for c in stripped):
        return None
    lead = len(surface) - len(surface.lstrip())
    start += lead
    end = start + len(stripped)
    return replace(span, start=start, end=end, text=stripped)


def _subtract(span: Span, taken: list[tuple[int, int]], text: str) -> Span | None:
    """Return ``span`` with any already-claimed regions removed.

    Keeps the longest surviving contiguous run. Returning the *longest* remainder rather
    than all of them keeps the mapping readable; anything genuinely separate will have
    been detected as its own span anyway.
    """
    intervals = sorted((s, e) for s, e in taken if s < span.end and span.start < e)
    if not intervals:
        return span

    best: tuple[int, int] | None = None
    cursor = span.start
    for s, e in intervals:
        if s > cursor:
            gap = (cursor, min(s, span.end))
            if best is None or (gap[1] - gap[0]) > (best[1] - best[0]):
                best = gap
        cursor = max(cursor, e)
        if cursor >= span.end:
            break
    if cursor < span.end:
        gap = (cursor, span.end)
        if best is None or (gap[1] - gap[0]) > (best[1] - best[0]):
            best = gap

    if best is None:
        return None

    # Snap the remainder to word boundaries. A trim that lands mid-word leaves a fragment
    # ("Halloway" minus a claimed "H" -> "alloway"), and a fragment is not an identifier:
    # whatever mattered is already covered by the span that won the overlap. Worse, the
    # fragment poisons Stage E -- "alloway" fuzzy-matches "Halloway" at 100, so it drags
    # unrelated spans into one cluster and one placeholder.
    return _snap(replace(span, start=best[0], end=best[1], text=text[best[0]:best[1]]),
                 text, min_len=_MIN_TRIMMED_LEN)


def _code_fragment(text: str, start: int, end: int) -> bool:
    """True when the span is only *part* of a surrounding clinical-code token.

    The tagger tagged the "10" inside "ICD-10" as a DATE, giving ``ICD-[DATE_1]``. Widening the
    span to its whitespace-delimited token and asking ``is_clinical_code`` catches that.

    The ``strictly larger`` condition is load-bearing and is why this is not simply
    ``is_clinical_code(span.text)``: that predicate matches ``^\\d{5}$`` for CPT codes, and a
    5-digit ZIP is also five digits. Suppressing a whole-token match would silently stop
    masking every ZIP in the corpus -- trading a cosmetic false positive for a real leak. A
    span that *is* the entire token keeps whatever category the detectors gave it; only a
    fragment of a larger code token is dropped.
    """
    left = start
    while left > 0 and not text[left - 1].isspace():
        left -= 1
    right = end
    while right < len(text) and not text[right].isspace():
        right += 1
    token = text[left:right].strip(".,;:()[]")
    return len(token) > (end - start) and is_clinical_code(token)


def _widen_caps(text: str, start: int, end: int) -> str:
    """Grow a span across adjacent capitalised words on the same line.

    Used only by the bare-facility test below, to tell a fragment from a whole phrase.
    """
    left = start
    while True:
        probe = left
        while probe > 0 and text[probe - 1] == " ":
            probe -= 1
        word_end = probe
        while probe > 0 and text[probe - 1].isalpha():
            probe -= 1
        word = text[probe:word_end]
        if not word or not (word.istitle() or word.isupper()):
            break
        left = probe
    right = end
    while True:
        probe = right
        while probe < len(text) and text[probe] == " ":
            probe += 1
        word_start = probe
        while probe < len(text) and text[probe].isalpha():
            probe += 1
        word = text[word_start:probe]
        if not word or not (word.istitle() or word.isupper()):
            break
        right = probe
    return text[left:right]


def _bare_facility_span(text: str, start: int, end: int) -> bool:
    """True when a span is facility vocabulary *and* is not part of a named facility.

    The second half is a fix for a leak this guard itself introduced. Dropping every
    all-facility span looked safe on "the Medical Center", and on a real letterhead reading
    ``HALLOWAY ORTHOPEDIC SPINE INSTITUTE`` it turned the output into
    ``[ORG_19] ORTHOPEDIC SPINE INSTITUTE`` -- the tagger had emitted the proper name and the
    facility phrase as two spans, the guard dropped the second, and Stage G correctly reported a
    readable organisation. An over-redaction guard that fires on a fragment of a real name is
    not a cosmetic improvement, it is a leak, which is the same trap ``_code_fragment`` avoids
    by requiring the span to be a strict fragment.

    So the span is dropped only when widening it across adjacent capitalised words yields
    *still* nothing but facility vocabulary:

    * "the Medical Center" -> widens to "Medical Center" (lowercase "the" stops it) -> all
      facility vocabulary -> a kind of place, dropped.
    * "HALLOWAY ORTHOPEDIC SPINE INSTITUTE" -> widening from the facility part picks up
      "HALLOWAY", which is not facility vocabulary -> a named place, kept and masked.
    """
    return (is_bare_facility(text[start:end])
            and is_bare_facility(_widen_caps(text, start, end)))


#: Minimum digit count each numeric identifier category actually requires. A category gate is
#: legitimate here, unlike the ones the module docstring warns about, because it encodes what
#: the category *is* rather than a guess about which label a model will emit: a telephone number
#: has at least seven digits and an SSN has exactly nine, so a three-digit span under those
#: labels is not a low-confidence identifier, it is arithmetically not one.
_MIN_DIGITS = {
    Category.PHONE: 7, Category.FAX: 7, Category.SSN: 9,
    Category.MRN: 4, Category.ACCOUNT: 4, Category.HEALTH_PLAN_ID: 4,
    Category.LICENCE: 4, Category.ID_GENERIC: 4, Category.DEVICE: 4,
}

#: Money, which a billing table is full of and which is never an identifier: an optional
#: currency symbol, grouped digits, and exactly two decimal places.
_MONEY = re.compile(r"^[$€£]?\s?[\d,]*\d(?:\.\d{2})?$")


#: Categories whose digit count is a structural fact about the identifier rather than a habit:
#: a telephone number has at least seven digits and an SSN exactly nine, whatever letters the
#: surrounding cue words contribute. Letters are ignored for these, which is what catches
#: "extension                  4" -- one digit under a seven-digit category.
_DIGITS_ONLY_IF_NUMERIC = frozenset(
    {Category.MRN, Category.ACCOUNT, Category.HEALTH_PLAN_ID,
     Category.LICENCE, Category.ID_GENERIC, Category.DEVICE}
)


def _implausible_identifier(span: Span) -> bool:
    """True when a numeric span cannot be an instance of the category claimed for it.

    From the same 22-page record, all from one billing page: ``[FAX_1] ,310.00``,
    ``[FAX_2] 950``, ``[HEALTH_PLAN_ID_2] 756``, ``[HEALTH_PLAN_ID_3] $18,204.36``, and
    ``[PHONE_4] extension                5``. The rule layer requires a context cue before
    calling anything a fax; the tagger does not, and a table of charges is a page of bare
    numbers. Masking them destroys the billing table -- the utility half of the brief -- while
    removing no identifier.

    Two rules, because the categories differ in kind:

    * ``PHONE``/``FAX``/``SSN``: count digits and ignore letters. "extension 4" carries one
      digit, and no cue word makes a one-digit string a telephone number. The base number it
      belongs to is masked on its own.
    * everything in ``_DIGITS_ONLY_IF_NUMERIC``: only *purely* numeric spans are judged, so
      "MER-778213" and "PCG-4471902" are untouched -- an alphanumeric identifier is allowed to
      be short because its letters carry entropy too.

    AGE and DATE are absent from ``_MIN_DIGITS`` on purpose: a two-digit age and a one-digit
    day are both real.
    """
    floor = _MIN_DIGITS.get(span.category)
    if floor is None:
        return False
    if span.category in _DIGITS_ONLY_IF_NUMERIC and any(c.isalpha() for c in span.text):
        return False
    digits = [c for c in span.text if c.isdigit()]
    if not digits:
        return False
    if len(digits) < floor:
        return True
    return _MONEY.match(span.text.strip()) is not None and "." in span.text


def is_over_redaction(text: str, span: Span) -> bool:
    """Every over-redaction guard, as one predicate, so there is one thing to call.

    Written because Stage G re-detects with the rule layer and applied only the eponym guard --
    gated on name categories, the same mistake this module's docstring is about. The result was
    that anything ``merge`` deliberately declined to mask got re-found by the self-check and
    reported as a leak, so fixing four over-redactions raised the self-check count from five
    findings to twelve. A guard the emitter honours and the auditor does not is worse than no
    guard: it turns a clean document into a blocked one.

    So both stages call this, and any guard added here reaches both by construction.
    """
    return (
        is_eponym_use(text, span.start, span.end)
        or is_clinical_label(span.text)
        or _code_fragment(text, span.start, span.end)
        or _bare_facility_span(text, span.start, span.end)
        or is_credential_only(span.text)
        or _implausible_identifier(span)
    )


def merge(text: str, spans: list[Span], policy: Policy) -> list[Span]:

    """Threshold, guard, and de-overlap a candidate span list."""
    candidates: list[Span] = []
    for s in spans:
        if not _passes_threshold(s, policy):
            continue
        # Every over-redaction guard, in one call, shared with Stage G. See
        # ``is_over_redaction``: the emitter and the auditor have to agree about what is not an
        # identifier, or the auditor blocks the document over the emitter's correct decision.
        if is_over_redaction(text, s):
            continue
        # Snap before thresholding order is decided, so a mid-word or punctuation-only span
        # never reaches the document. See ``_snap`` for the three ways this went wrong.
        snapped = _snap(s, text)
        if snapped is None:
            continue
        candidates.append(snapped)

    candidates.sort(key=_priority, reverse=True)

    accepted: list[Span] = []
    taken: list[tuple[int, int]] = []
    for s in candidates:
        trimmed = _subtract(s, taken, text)
        if trimmed is None:
            continue
        accepted.append(trimmed)
        taken.append((trimmed.start, trimmed.end))

    accepted.sort(key=lambda s: (s.start, s.end))
    return accepted
