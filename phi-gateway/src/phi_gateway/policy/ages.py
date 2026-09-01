"""Age handling. HIPAA identifier 3 includes "all elements of dates ... and all ages over
89 and all elements of dates (including year) indicative of such age".

So ages 0-89 may be retained -- and should be, because age drives most clinical reasoning.
Ages 90+ collapse to a single bucket. The subtle part is the *indicative of such age* half:
"born in 1929" in a 2024 note is an age over 89 expressed as a date, and a system that only
looks at numbers followed by "year old" leaks it.
"""

from __future__ import annotations

import re

from ..config import AgePolicy
from ..types import Category, Span

AGE_CUTOFF = 90

#: Words that state an age band over 89 without a number.
_OVER_89_WORDS = re.compile(
    r"(?i)\b(?:nonagenarian|centenarian|supercentenarian|"
    r"in\s+(?:her|his|their)\s+(?:(?:early|mid|late)\s+)?(?:90s|nineties|100s))\b"
)


def age_value(text: str) -> int | None:
    m = re.search(r"\d{1,3}", text)
    return int(m.group()) if m else None


def is_over_89(text: str) -> bool:
    if _OVER_89_WORDS.search(text):
        return True
    v = age_value(text)
    return v is not None and v >= AGE_CUTOFF


def year_is_identifying(birth_year: int, reference_year: int) -> bool:
    """Whether retaining ``birth_year`` would reveal an age over 89.

    Uses the loose bound (no birthday known), because the recall-asymmetric choice is to
    suppress a year that *might* imply 90 rather than publish one that does.
    """
    return reference_year - birth_year >= AGE_CUTOFF - 1


#: The field label, not the value. The tagger learns ``Age: 93`` as one span and at inference
#: sometimes splits it, emitting ``AGE_OVER_89`` for the bare label as well as for the number.
#: Trusting the category alone then masked both halves and the pair rehydrated to ``Age: Age:``,
#: because ``[AGE_OVER_89]`` is deliberately many-to-one. Matching the *whole* surface keeps
#: this narrow: a spelled-out age ("ninety-four") does not match, so it is still suppressed.
_LABEL_ONLY = re.compile(
    r"(?i)^\W*(?:age|ages|aged|dob|d\.?o\.?b\.?|birth\s*date|date\s*of\s*birth"
    r"|yrs?|years?|y[./]?o|y\.o\.|old)\W*$"
)


def replacement(span: Span, policy: AgePolicy) -> str | None:
    """What an age span becomes. ``None`` means leave the original text in place."""
    if _LABEL_ONLY.match(span.text):
        return None
    if span.category is Category.AGE_OVER_89 or is_over_89(span.text):
        return policy.over_89_placeholder
    if policy.retain_under_90:
        return None
    return "[AGE]"
