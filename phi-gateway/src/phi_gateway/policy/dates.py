"""Date handling -- the most interesting utility/leak trade-off in the whole project.

HIPAA Safe Harbor removes *all* date elements finer than a year. Taken literally that
destroys the clinical timeline, and the timeline is often the point of the note
("readmitted 3 days after discharge"). The brief asks for date *shifting* that preserves
intervals; three modes are implemented so the trade-off is a measured config knob rather
than an opinion:

``safe_harbor``
    ``[DATE_1]``. No date element survives. Maximum compliance, worst utility.

``interval_preserving`` (default)
    ``[DATE_1]``, then ``[DATE_2 = DATE_1 + 3d]``. Strictly better than shifting: no date
    element is emitted at all -- so it is Safe-Harbor-clean -- yet every interval is exact
    and the downstream LLM can still answer "how long between admission and surgery?".
    Costs nothing but placeholder verbosity.

``shifted``
    Every date moved by one offset derived as ``HMAC(salt, patient_key)``. Intervals exact,
    dates look real (better for LLMs that reason poorly over placeholders), but a real-looking
    date *is* a date element, so this is Expert-Determination territory, not Safe Harbor.
    Offered, documented, not the default.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta

from ..config import DatePolicy
from ..types import Span
from ..detectors.validators import month_number, normalise_year, plausible_date

#: Salt for the per-patient date offset. Random per process unless pinned, so an offset is
#: not reproducible across deployments by anyone who merely knows the patient key.
_SALT_ENV = "PHI_DATE_SALT"

_MONTH_WORD = r"[A-Za-z]{3,9}"


@dataclass(frozen=True)
class ParsedDate:
    year: int | None = None
    month: int | None = None
    day: int | None = None

    @property
    def full(self) -> bool:
        return self.year is not None and self.month is not None and self.day is not None

    def as_date(self) -> date | None:
        if not self.full:
            return None
        try:
            return date(self.year, self.month, self.day)  # type: ignore[arg-type]
        except ValueError:
            return None

    def key(self) -> str:
        return f"{self.year or '?'}-{self.month or '?'}-{self.day or '?'}"


def parse(text: str) -> ParsedDate | None:
    """Parse the surface forms the date detectors actually emit.

    Deliberately not ``dateutil.parse``: that helpfully invents a year for "3/14" and
    silently reads ambiguous input, which is how a de-identifier ends up shifting two
    dates by different amounts and destroying the interval it was trying to preserve.
    """
    t = text.strip().strip(".,;")

    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if m:
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
        return ParsedDate(y, mo, d) if plausible_date(y, mo, d) else None

    m = re.fullmatch(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})", t)
    if m:
        mo, d, y = int(m[1]), int(m[2]), normalise_year(int(m[3]))
        # US month-first, with a day-first fallback for impossible months (13/04/2024).
        if mo > 12 and d <= 12:
            mo, d = d, mo
        return ParsedDate(y, mo, d) if plausible_date(y, mo, d) else None

    m = re.fullmatch(rf"({_MONTH_WORD})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})", t)
    if m:
        mo = month_number(m[1])
        if mo:
            y, d = int(m[3]), int(m[2])
            return ParsedDate(y, mo, d) if plausible_date(y, mo, d) else None

    m = re.fullmatch(rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_WORD})\.?,?\s+(\d{{4}})", t)
    if m:
        mo = month_number(m[2])
        if mo:
            y, d = int(m[3]), int(m[1])
            return ParsedDate(y, mo, d) if plausible_date(y, mo, d) else None

    m = re.fullmatch(rf"({_MONTH_WORD})\.?\s+(\d{{4}})", t)
    if m:
        mo = month_number(m[1])
        if mo:
            return ParsedDate(int(m[2]), mo, None)

    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})", t)   # bare M/D, no year
    if m:
        mo, d = int(m[1]), int(m[2])
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return ParsedDate(None, mo, d)
        return None

    m = re.fullmatch(r"(19|20)\d{2}", t)
    if m:
        return ParsedDate(int(t), None, None)

    return None


def shift_days(patient_key: str, window_days: int) -> int:
    """Deterministic per-patient offset in ``[-window, +window]``, excluding zero.

    Deterministic so the same patient's notes stay mutually consistent across requests --
    otherwise two notes about one patient would place the same admission on two different
    days, which is worse than useless for longitudinal reasoning.
    """
    if window_days <= 0:
        return 0
    salt = os.environ.get(_SALT_ENV, "").encode() or hashlib.sha256(b"phi-gateway-dev-salt").digest()
    digest = hmac.new(salt, patient_key.encode("utf-8"), hashlib.sha256).digest()
    raw = int.from_bytes(digest[:8], "big")
    magnitude = raw % window_days + 1
    return -magnitude if digest[8] & 1 else magnitude


@dataclass
class DateLabel:
    """What one distinct date value becomes in the masked text."""

    replacement: str
    #: False only in ``shifted`` mode, where the replacement is a real-looking date.
    is_placeholder: bool
    parsed: ParsedDate | None
    canonical: str


def _fmt_delta(days: int) -> str:
    return f"+ {days}d" if days >= 0 else f"- {abs(days)}d"


def value_key(text: str) -> str:
    """Grouping key for a date surface form: the parsed value when we understood it, the
    normalised text when we did not. Two spellings of one date must share a key."""
    parsed = parse(text)
    if parsed:
        return parsed.key()
    flat = re.sub(r"\s+", " ", text.strip().lower())
    return f"raw:{flat}"


def reference_year(labels: dict[str, ParsedDate | None]) -> int | None:
    """Latest year mentioned anywhere -- our best proxy for "when was this note written",
    needed to tell whether a retained birth year implies an age over 89."""
    years = [p.year for p in labels.values() if p and p.year]
    return max(years) if years else None


#: Widest interval worth annotating. Beyond this the two dates are not part of one
#: clinical episode and the delta is not something the downstream LLM reasons over.
_MAX_INTERVAL_DAYS = 730


def _pick_anchor(order) -> tuple[int | None, date | None]:
    """Choose the date that interval annotations are measured from.

    Not simply the earliest: a DOB sits decades before everything else, and anchoring on
    it turns every clinical interval into ``+26298d`` -- useless to the LLM, and an
    interval that hands the birth year straight back. So anchor on the date with the most
    company inside one clinical window; earliest wins ties.
    """
    resolved: list[tuple[int, date]] = []
    for idx, (_, (parsed, _, _)) in enumerate(order, start=1):
        d = parsed.as_date() if parsed else None
        if d is not None:
            resolved.append((idx, d))
    if not resolved:
        return None, None
    best_idx, best_date, best_company = resolved[0][0], resolved[0][1], -1
    for idx, d in resolved:
        company = sum(1 for _, o in resolved if abs((o - d).days) <= _MAX_INTERVAL_DAYS)
        if company > best_company:
            best_idx, best_date, best_company = idx, d, company
    return best_idx, best_date


def plan_dates(
    spans: list[Span],
    policy: DatePolicy,
    patient_key: str,
    *,
    year_is_identifying=None,
) -> tuple[dict[str, DateLabel], int]:
    """Assign a replacement to every distinct date value in the document.

    Returns ``(value_key -> DateLabel, shift_days)``. Grouping is by *parsed value*, not by
    surface form, so ``03/14/2024`` and ``March 14, 2024`` collapse to one placeholder --
    which is the whole point of consistent pseudonymisation.
    """
    groups: dict[str, tuple[ParsedDate | None, str, int]] = {}
    for s in sorted(spans, key=lambda s: (s.start, s.end)):
        key = value_key(s.text)
        if key not in groups:
            groups[key] = (parse(s.text), s.text.strip(), s.start)

    order = sorted(groups.items(), key=lambda kv: kv[1][2])
    shift = shift_days(patient_key, policy.shift_window_days) if policy.mode == "shifted" else 0

    #: Anchor that interval annotations are measured from. See ``_pick_anchor``.
    anchor_idx, anchor = _pick_anchor(order)

    ref_year = reference_year({k: v[0] for k, v in groups.items()})

    labels: dict[str, DateLabel] = {}
    for idx, (key, (parsed, surface, _)) in enumerate(order, start=1):
        d = parsed.as_date() if parsed else None

        if policy.mode == "shifted" and d is not None:
            labels[key] = DateLabel(
                replacement=(d + timedelta(days=shift)).isoformat(),
                is_placeholder=False,
                parsed=parsed,
                canonical=surface,
            )
            continue

        base = f"DATE_{idx}"
        annotation = ""

        if (
            policy.mode == "interval_preserving"
            and policy.emit_interval_annotations
            and d is not None
            and anchor is not None
            and d != anchor
            and abs((d - anchor).days) <= _MAX_INTERVAL_DAYS
        ):
            annotation = f" = DATE_{anchor_idx} {_fmt_delta((d - anchor).days)}"
        elif policy.retain_year and parsed and parsed.year:
            # Safe Harbor permits the year -- unless the note's own timespan makes it an
            # age over 89, which is an aggregation leak the year alone would carry.
            identifying = bool(
                year_is_identifying and ref_year and year_is_identifying(parsed.year, ref_year)
            )
            if not identifying:
                annotation = f" = {parsed.year}"

        labels[key] = DateLabel(
            replacement=f"[{base}{annotation}]",
            is_placeholder=True,
            parsed=parsed,
            canonical=surface,
        )

    return labels, shift
