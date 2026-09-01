"""Checksum and range validators.

These are what separate "matches a pattern" from "is actually an identifier". A bare
10-digit number is not an NPI; an NPI passes Luhn against the 80840 prefix. Validating
lets the structured-ID detectors run at high confidence *without* generating the flood
of false positives that would make the masked text unusable -- which is the whole
leak-versus-utility trade-off, applied at the detector level.
"""

from __future__ import annotations

import ipaddress
import re
from datetime import date

# ---------------------------------------------------------------------------------
# Provider identifiers
# ---------------------------------------------------------------------------------

def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def valid_npi(value: str) -> bool:
    """NPI: 10 digits, last is a Luhn check digit computed over '80840' + first 9."""
    digits = re.sub(r"\D", "", value)
    if len(digits) != 10:
        return False
    return _luhn_ok("80840" + digits)


def valid_dea(value: str) -> bool:
    """DEA number: 2 letters + 7 digits.

    Check digit = (d1+d3+d5 + 2*(d2+d4+d6)) mod 10, compared against d7.
    """
    m = re.fullmatch(r"([A-Za-z]{2})(\d{7})", value.strip())
    if not m:
        return False
    d = [int(c) for c in m.group(2)]
    return (d[0] + d[2] + d[4] + 2 * (d[1] + d[3] + d[5])) % 10 == d[6]


# ---------------------------------------------------------------------------------
# Vehicle
# ---------------------------------------------------------------------------------

_VIN_TRANSLIT = {
    **{str(i): i for i in range(10)},
    "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8,
    "J": 1, "K": 2, "L": 3, "M": 4, "N": 5, "P": 7, "R": 9,
    "S": 2, "T": 3, "U": 4, "V": 5, "W": 6, "X": 7, "Y": 8, "Z": 9,
}
_VIN_WEIGHTS = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)


def valid_vin(value: str) -> bool:
    """17-character VIN with the position-9 check digit verified.

    Without the check digit this pattern would fire on any 17-character alphanumeric
    run, of which clinical text has plenty (accession numbers, barcodes).
    """
    v = value.strip().upper()
    if len(v) != 17 or re.search(r"[IOQ]", v):
        return False
    total = 0
    for ch, w in zip(v, _VIN_WEIGHTS):
        if ch not in _VIN_TRANSLIT:
            return False
        total += _VIN_TRANSLIT[ch] * w
    rem = total % 11
    expected = "X" if rem == 10 else str(rem)
    return v[8] == expected


# ---------------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------------

def valid_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value.strip())
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------------
# SSN
# ---------------------------------------------------------------------------------

def valid_ssn(value: str) -> bool:
    """Reject SSNs the SSA never issues: area 000/666/900-999, group 00, serial 0000."""
    digits = re.sub(r"\D", "", value)
    if len(digits) != 9:
        return False
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    if area in ("000", "666") or area[0] == "9":
        return False
    return group != "00" and serial != "0000"


# ---------------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------------

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

MIN_YEAR = 1900
MAX_YEAR = 2100


def month_number(name: str) -> int | None:
    return _MONTHS.get(name.strip(". ").lower())


def normalise_year(y: int) -> int:
    """Two-digit years: 00-30 -> 2000s, 31-99 -> 1900s. Clinical notes contain both
    recent dates and mid-century birth dates, so the pivot has to allow for DOBs."""
    if y >= 100:
        return y
    return 2000 + y if y <= 30 else 1900 + y


def plausible_date(year: int, month: int, day: int | None = None) -> bool:
    if not (MIN_YEAR <= year <= MAX_YEAR):
        return False
    if not (1 <= month <= 12):
        return False
    if day is None:
        return True
    try:
        date(year, month, day)
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------------------------
# ID_GENERIC guard rails -- the HIPAA-18 catch-all
# ---------------------------------------------------------------------------------

#: Codes that look like identifiers but are clinical vocabulary. Masking these would
#: destroy exactly the meaning the downstream LLM needs, so they are excluded from the
#: catch-all detector. This is the "redact too much is also a failure" side of the
#: trade-off, made explicit.
_CLINICAL_CODE_PATTERNS = (
    r"^[A-TV-Z]\d{2}(?:\.[A-Z0-9]{1,4})?$",          # ICD-10-CM
    r"^\d{5}$",                                       # CPT / HCPCS numeric
    r"^[A-Z]\d{4}$",                                  # HCPCS level II
    r"^\d{1,5}-\d$",                                  # LOINC
    r"^(?:19|20)\d{2}$",                              # bare year
    r"^\d{1,4}(?:\.\d+)?\s*(?:MG|MCG|G|ML|L|MEQ|IU|U|MMOL|MMHG|KG|LB|CM|MM|F|C)$",
    r"^(?:COVID|SARS|HIV|HBV|HCV|HPV|MRSA|VRE|CMV|EBV|RSV|TB)[- ]?\d*$",
    r"^(?:A1C|HGB|HCT|WBC|RBC|MCV|MCH|BUN|ALT|AST|ALP|LDH|CPK|BNP|PSA|TSH|"
    r"INR|PTT|PT|CRP|ESR|GFR|CO2|O2|CT|MRI|PET|EKG|ECG|EEG|CXR|IV|IM|PO|"
    r"PRN|BID|TID|QID|QHS|QAM|NPO|DNR|DNI)\d*$",
    r"^[LCTS]\d{1,2}(?:-[LCTS]?\d{1,2})?$",           # spinal levels L4-L5, C3
    r"^(?:GRADE|STAGE|CLASS|TYPE|LEVEL|PHASE)[- ]?[IV0-9]+$",
    r"^[IVX]{1,4}[AB]?$",                             # roman numeral staging
    #: Names of the coding systems themselves. "ICD-10" is letters+digits at a plausible
    #: length, so without this the catch-all masks the vocabulary the note is citing.
    r"^(?:ICD|DSM|CPT|HCPCS|LOINC|SNOMED|RXNORM|NDC|ATC|DRG|MS-DRG|HCC|CVX|NANDA)"
    r"(?:[- ]?(?:9|10|11|IV|5|CM|PCS|CT))*$",
)
_CLINICAL_CODE = re.compile("|".join(_CLINICAL_CODE_PATTERNS), re.IGNORECASE)


def is_clinical_code(token: str) -> bool:
    return bool(_CLINICAL_CODE.match(token.strip()))


def looks_like_opaque_id(token: str, *, min_len: int = 6, max_len: int = 24) -> bool:
    """Heuristic for HIPAA category 18: 'any other unique identifying number or code'.

    Requires a mix of letters and digits at a plausible length, and excludes clinical
    vocabulary. Category 18 is unbounded and cannot be covered exhaustively -- this is a
    net, not a guarantee, and the leak self-check plus re-identification risk scoring
    are what compensate. Stated plainly rather than papered over.
    """
    t = token.strip()
    if not (min_len <= len(t) <= max_len):
        return False
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-_/]*[A-Za-z0-9]", t):
        return False
    if not (re.search(r"\d", t) and re.search(r"[A-Za-z]", t)):
        return False
    if is_clinical_code(t):
        return False
    # Pure words with a trailing digit ("Patient1") are usually not identifiers.
    if re.fullmatch(r"[A-Za-z]{4,}\d{1,2}", t):
        return False
    # Hyphenated quantities read as letters+digits but are clinical prose. "36-year-old"
    # flagged as a category-18 identifier blocks the pipeline on the self-check rescan.
    if re.fullmatch(
        r"(?i)\d{1,3}-(?:year|yr|month|mo|week|wk|day)s?-old|\d{1,3}-(?:day|week|month|year)s?",
        t,
    ):
        return False
    return True
