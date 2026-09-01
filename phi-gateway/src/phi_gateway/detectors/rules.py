"""Stage B: validated rule detectors.

These handle the identifier categories that are *structurally* recognisable -- SSN,
phone, fax, email, URL, IP, MRN, account, NPI/DEA, VIN, device serial, ZIP, dates, ages.
Where a checksum exists it is enforced (see ``validators.py``), which is what lets these
run at high confidence without flooding the output with false positives.

What this layer deliberately does NOT do is guess at names from a gazetteer. A surname
list is exactly what turns "Parkinson's disease" into a false positive. Names are
detected here only from an explicit cue -- a title, a trailing credential, a signature
line, or a form-field key -- and broad name recall is the neural tagger's job (Stage C).

Consequence worth stating plainly: run rules-only and name recall is low. That is the
honest floor, not the answer. The floor exists so the gateway round-trips before any
model is trained, and so there is a published baseline (P2.7 requires beating regex-only).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from ..lexicons import (
    CREDENTIAL_ALT,
    EPONYM_CONTEXT,
    EPONYM_PRECEDING,
    MEDICAL_EPONYMS,
    PROVIDER_TITLES,
)
from ..types import Category, Segment, Source, Span
from . import validators as V
from .structural import in_structured_segment

# =================================================================================
# Building blocks
# =================================================================================

#: Leading initials are how a produced record writes a provider ("J. Okafor, NRP") and the
#: trailing initial is how it writes the patient ("WHITFIELD, MARCUS D."). Without both, the
#: cue fires but the span stops short and the initial is left sitting in the masked output --
#: a lone initial next to a masked surname is a partial identifier, not a clean redaction.
#: Single spaces, not ``\s``: a name never spans a line break or a table cell boundary, and
#: with ``\s+`` "Service\n\n   Referral" matched as one person's name.
_NAME_CORE = (
    r"(?:[A-Z]\.[ ]?){0,2}"
    r"[A-Z][A-Za-z'’\-]{1,19}(?:[ ][A-Z]\.?)?(?:[ ][A-Z][A-Za-z'’\-]{1,19}){0,2}"
    r"(?:[ ][A-Z]\.)?"
)

_TITLE_ALT = (
    r"Doctor|Dr\.?|Professor|Prof\.?|Mrs\.?|Mr\.?|Ms\.?|Miss|Mx\.?|Rev\.?|"
    r"Sir|Dame|Nurse|Sister|Father|Fr\.?"
)

_CRED_ALT = CREDENTIAL_ALT

#: A name that is nothing but a credential is not a name. See ``_detect_names``.
_CRED_ONLY = re.compile(rf"(?:{CREDENTIAL_ALT})", re.IGNORECASE)

#: Organisation head nouns strong enough to anchor a match on their own.
#: Multi-word phrases come first so the alternation prefers the longer form.
_ORG_STRONG = (
    r"Health\s+System|Health\s+Network|Health\s+Services|Health\s+Partners|"
    r"Medical\s+Group|Medical\s+Cent(?:er|re)|Surgery\s+Cent(?:er|re)|"
    r"Cancer\s+Cent(?:er|re)|Care\s+Cent(?:er|re)|Urgent\s+Care|Nursing\s+Home|"
    r"Medical\s+Practice|Hospitals?|Clinics?|Memorial|Cent(?:er|re)|Infirmary|"
    r"Sanatorium|Institutes?|Laborator(?:y|ies)|Labs|Pharmacy|Hospice|Associates|"
    r"Physicians|Partners|Healthcare|Rehabilitation|Rehab|Pavilion|Deaconess|"
    # Anchors that survive an index table truncating the column: the produced record's own
    # page index lists "Piedmont County General" and "Northgate Family Medicine" with the
    # head noun cut off, so without these the one table naming every facility leaks them.
    r"Family\s+Medicine|County\s+General|County\s+EMS|Interventional\s+Pain|"
    r"Sports\s+Therapy|Neurology|Electrodiagnostics|EMS|"
    # Company suffixes. The defendant employer is an identifier by association, and
    # "Corrigan" alone is suppressed as a medical eponym (Corrigan pulse) -- correctly, so
    # the organisation has to be caught from the suffix instead.
    r"LLC|L\.L\.C\.|LLP|Incorporated|Inc\.?|Corporation|Corp\.?"
)

#: Tokens allowed *around* a strong anchor but never sufficient alone. "Medical" on its
#: own must not match, or "Past Medical History" becomes an organisation.
_ORG_WEAK = (
    r"Medical|Health|General|Regional|Surgery|Surgical|Nursing|Practice|University|"
    r"Foundation|Presbyterian|Baptist|Methodist|Lutheran|Mercy|Sinai|Oncology|"
    r"Cardiology|Orthopa?edic|Imaging|Radiology|Diagnostics|Dialysis|Childrens?|"
    r"Women’?s?|Veterans|Community|County|District|Saint|St\.?|"
    # Head nouns a produced record's index page truncates to: the column cuts
    # "Piedmont County General Hospital" off at "General", so the strong anchor never
    # appears and the facility survives masking in the very table listing every facility.
    r"Medicine|Neurology|Electrodiagnostics|Spine|Sports|Therapy|EMS|Urgent|Interventional"
)

_CAP_TOKEN = r"[A-Z][A-Za-z'’\-]{1,20}"


def _re(pattern: str, flags: int = 0) -> re.Pattern[str]:
    return re.compile(pattern, flags)


# =================================================================================
# Patterns
# =================================================================================

P_EMAIL = _re(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
P_URL = _re(r"\b(?:https?://|ftp://|www\.)[^\s<>\"')\]]+", re.IGNORECASE)
P_IPV4 = _re(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
P_IPV6 = _re(r"\b(?:[A-Fa-f0-9]{0,4}:){2,7}[A-Fa-f0-9]{0,4}\b")

P_SSN = _re(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")

P_PHONE = _re(
    r"(?:\+?\d{1,2}[-.\s]?)?"
    r"(?:\(\d{3}\)|\b\d{3})[-.\s]\d{3}[-.\s]\d{4}\b"
)
P_PHONE_EXT = _re(r"(?i)\b(?:ext\.?|extension)\s*\d{1,6}\b|\bx\s*\d{3,6}\b")
_FAX_CUE = _re(r"(?i)\bfax\b[^.\n]{0,20}$")

#: Cued identifier detectors. The capture group named ``id`` is the span.
CUED_IDS: tuple[tuple[str, Category, str, float], ...] = (
    (
        "mrn",
        Category.MRN,
        r"(?i)\b(?:mrn|m\.r\.n\.|medical\s+record(?:\s+(?:no\.?|number|#))?|"
        r"chart\s*(?:no\.?|number|#)?|record\s*#|patient\s+id)\s*[:#]?\s*"
        r"(?P<id>[A-Z0-9][A-Z0-9\-]{3,19})\b",
        0.96,
    ),
    (
        "account",
        Category.ACCOUNT,
        r"(?i)\b(?:account|acct\.?|billing|invoice|claim)\s*(?:no\.?|number|#|id)?\s*[:#]?\s*"
        r"(?P<id>[A-Z0-9][A-Z0-9\-]{3,24})\b",
        0.94,
    ),
    (
        "health_plan",
        Category.HEALTH_PLAN_ID,
        r"(?i)\b(?:member|subscriber|policy|group|insurance|payer|beneficiary|plan|"
        r"medicare|medicaid|hicn|mbi)\s*(?:no\.?|number|#|id)\s*[:#]?\s*"
        r"(?P<id>[A-Z0-9][A-Z0-9\-]{3,24})\b",
        0.94,
    ),
    (
        "licence",
        Category.LICENCE,
        r"(?i)\b(?:licen[cs]e|lic\.?|cert(?:ificate)?|permit|registration)\s*"
        r"(?:no\.?|number|#|id)?\s*[:#]?\s*(?P<id>[A-Z0-9][A-Z0-9\-]{3,19})\b",
        0.92,
    ),
    (
        "device",
        Category.DEVICE,
        r"(?i)\b(?:serial\s*(?:no\.?|number|#)?|s/n|sn#|device\s*(?:id|no\.?|number|#)|"
        r"implant\s*(?:id|no\.?|#)?|lot\s*(?:no\.?|number|#)?|udi|catalog(?:ue)?\s*#)\s*"
        r"[:#]?\s*(?P<id>[A-Z0-9][A-Z0-9\-]{3,29})\b",
        0.92,
    ),
    (
        "plate",
        Category.VEHICLE,
        r"(?i)\b(?:licen[cs]e\s*plate|plate\s*(?:no\.?|number|#)?|tag\s*#)\s*[:#]?\s*"
        r"(?P<id>[A-Z0-9][A-Z0-9\- ]{3,9})\b",
        0.92,
    ),
)

P_NPI_CUED = _re(r"(?i)\bnpi\s*(?:no\.?|number|#|id)?\s*[:#]?\s*(?P<id>\d{10})\b")
P_NPI_BARE = _re(r"\b(?P<id>\d{10})\b")
P_DEA_CUED = _re(r"(?i)\bdea\s*(?:no\.?|number|#|id)?\s*[:#]?\s*(?P<id>[A-Z]{2}\d{7})\b")
P_DEA_BARE = _re(r"\b(?P<id>[A-Z]{2}\d{7})\b")
P_VIN = _re(r"\b(?P<id>[A-HJ-NPR-Z0-9]{17})\b")

P_ZIP_STATE = _re(
    r"\b(?:A[KLRZ]|C[AOT]|D[CE]|FL|GA|HI|I[ADLN]|K[SY]|LA|M[ADEINOST]|N[CDEHJMVY]|"
    r"O[HKR]|P[AR]|RI|S[CD]|T[NX]|UT|V[AT]|W[AIVY])[.,]?\s+(?P<id>\d{5})(?:-\d{4})?\b"
)
P_ZIP_CUED = _re(
    r"(?i)\b(?:zip(?:\s*code)?|postal\s*code|postcode)\s*[:#]?\s*(?P<id>\d{5})(?:-\d{4})?\b"
)

_MONTH_ALT = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t)?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)

#: Both separators must be the *same* character. No real date mixes them, while clinical
#: prose does: "low back pain 2-3/10" parsed as 2-3-10 and masked the pain range as a date.
P_DATE_NUMERIC = _re(r"\b(\d{1,2})(?P<sep>[/\-.])(\d{1,2})(?P=sep)(\d{2,4})\b")
P_DATE_ISO = _re(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
P_DATE_MDY = _re(
    rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.IGNORECASE
)
P_DATE_DMY = _re(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_ALT})\.?,?\s+(\d{{4}})\b", re.IGNORECASE
)
P_DATE_MY = _re(rf"\b({_MONTH_ALT})\.?\s+(\d{{4}})\b", re.IGNORECASE)
P_DATE_MD = _re(r"\b(\d{1,2})/(\d{1,2})\b")

#: Units and verbs that mean a bare "3/14" is a dose or ratio, not a date.
_MD_REJECT_AFTER = _re(
    r"(?i)^\s*(?:tab|tabs|tablet|tablets|cap|caps|capsule|capsules|tsp|tbsp|mg|mcg|g|"
    r"ml|l|cc|unit|units|iu|meq|mmol|drop|drops|puff|puffs|patch|inch|inches|\"|hr|hrs|"
    r"hour|hours|min|mins|day|days|week|weeks|month|months|year|years|%)\b"
)
_MD_REJECT_BEFORE = _re(
    r"(?i)(?:take|takes|taking|give|given|gave|administer(?:ed)?|inject(?:ed)?|"
    r"dose[d]?|dosing|receive[d]?|infuse[d]?|dilut(?:e|ed)|ratio|of)\s+$"
)

#: Scored scales are written exactly like a bare M/D date: "Pain 7/10", "EHL 4+/5",
#: "dorsiflexion 4/5". Masking those as dates is a false positive *and* a utility loss --
#: the score is the clinical content, and it is what the whole note is about.
#:
#: A window rather than the anchored cue above, because the cue is rarely adjacent: "neck
#: pain has improved from 6/10 to roughly 3/10" puts 35 characters and a second score
#: between "pain" and the match, and in a form row ("Pain" + a column of spaces + "1/10 at
#: rest, 3/10 with overhead reach") the gap is wider still. Bounded at 120 and stopped at
#: ``.`` / ``;`` / newline so the cue cannot reach out of its own sentence and suppress a
#: real date one clause later. Body parts are cues too: "leg 6/10, back 4/10" carries no
#: scale word at all, and the graded region is the only thing marking it as a score.
_SCALE_REJECT_BEFORE = _re(
    r"(?i)\b(?:pain|score[d]?|scale|rated|rating|severity|nrs|vas|gcs|grade|motor|"
    r"strength|power|reflex(?:es)?|dorsiflexion|plantarflexion|flexion|extension|"
    r"eversion|inversion|abduction|adduction|ehl|fhl|"
    r"leg|back|neck|arm|shoulder|knee|hip|ankle|wrist|elbow|foot|hand|radicular)"
    r"\b[^.;\n]{0,120}$"
)

P_AGE_YO = _re(
    r"(?i)\b(?P<age>\d{1,3})\s*[-\s]?\s*(?:years?|yrs?|y)\s*[-\s]?\s*(?:old|o)\b"
)
P_AGE_YO_SHORT = _re(r"(?i)\b(?P<age>\d{1,3})\s*y[/.]?o\b")
P_AGE_CUED = _re(r"(?i)\bage[d]?\s*(?:is|of|:|=)?\s*(?P<age>\d{1,3})\b")
P_AGE_OVER_89_WORD = _re(
    r"(?i)\b(?:nonagenarian|nonagenarians|centenarian|centenarians|supercentenarian|"
    r"(?:in\s+(?:his|her|their|the)\s+(?:90s|nineties|late\s+90s)))\b"
)

P_TITLE_NAME = _re(
    rf"\b(?P<title>{_TITLE_ALT})\s+(?P<name>{_NAME_CORE})"
)
P_NAME_CRED = _re(
    rf"\b(?P<name>{_NAME_CORE})\s*,\s*(?P<cred>{_CRED_ALT})(?![A-Za-z])"
)
P_SIG_NAME = _re(
    r"(?im)^\s*(?:electronically\s+|e-)?"
    r"(?:signed|dictated|transcribed|authenticated|verified|reviewed|entered|approved)"
    # The lookahead is load-bearing. "by" is optional, so on "Signed By [NAME_PROVIDER_1]"
    # the engine matches the optional group, fails on the blanked placeholder, backtracks
    # with the group empty, and captures "By" itself as the name -- a phantom leak that
    # blocks the forward on text we correctly masked a moment earlier.
    rf"\s*(?:by)?\s*[:\-]?\s*(?!by\b)(?P<name>{_NAME_CORE})"
)

#: Note the literal single spaces rather than ``\s+``: a run of two-plus spaces is a cell
#: separator and a newline is a new field, so an unbounded ``\s+`` here walks out of the
#: organisation and eats the *next* column's key ("[ORG_1]: [NAME_PROVIDER_1]"). The
#: repetition counts are bounded for the same reason -- greed here is over-redaction.
P_ORG = _re(
    rf"\b(?P<org>(?:(?:{_ORG_WEAK}|{_CAP_TOKEN}) ){{0,4}}"
    rf"(?:{_ORG_STRONG})"
    rf"(?: (?:of|for|at|the) {_CAP_TOKEN})?"
    rf"(?: (?:{_ORG_WEAK}|{_ORG_STRONG}|{_CAP_TOKEN})){{0,3}})\b"
)

P_STREET = _re(
    r"(?i)\b(?P<addr>\d{1,6}\s+(?:[NSEW]\.?|North|South|East|West|NE|NW|SE|SW)?\s*"
    # Scoped-off ignorecase: the street-name words must be *capitalised*. Under the outer
    # (?i) this run also matched lowercase prose, so "Priority 2 to eastbound Route 9"
    # parsed as a street address and redacted clinical text.
    r"(?-i:(?:[A-Z][A-Za-z'’\-]{1,20}\s+){1,4})"
    r"(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|Lane|Ln\.?|Drive|"
    r"Dr\.?|Court|Ct\.?|Circle|Cir\.?|Place|Pl\.?|Terrace|Ter\.?|Way|Parkway|Pkwy\.?|"
    r"Highway|Hwy\.?|Route|Rte\.?|Trail|Trl\.?|Square|Sq\.?)"
    r"(?:\s*(?:Apt\.?|Unit|Suite|Ste\.?|#)\s*[A-Za-z0-9\-]{1,6})?)\b"
)

P_OPAQUE = _re(r"\b[A-Za-z0-9][A-Za-z0-9\-_/]{4,23}\b")

#: Bare runs of nine or more digits. ``looks_like_opaque_id`` requires *both* letters and
#: digits, so an uncued all-numeric MRN, account or unformatted phone ("6175550311") fell
#: through every detector above: ``P_PHONE`` needs a separator and ``P_NPI_BARE`` only fires
#: on the ~1-in-10 Luhn coincidence. Nine contiguous digits is not clinical content -- years
#: are four, dosages carry units, ZIP+4 carries a hyphen -- so this closes the hole rather
#: than trading it for over-redaction.
P_LONG_DIGITS = _re(r"\b\d{9,}\b")


# =================================================================================
# Eponym guard -- the ambiguity problem, applied defensively
# =================================================================================

_TITLE_BEFORE = _re(rf"\b(?:{_TITLE_ALT})\s+$")


#: Words that are clinical measurement labels and never identifiers, whatever a model thinks.
#:
#: Exists because the tagger masked the literal word "Pain" as ``NAME_PATIENT`` on a real
#: 22-page record: in a two-column vitals row, ``Pain    7/10`` has the same shape as
#: ``Patient    Marcus Whitfield``, and the training generator only ever placed clinical scores
#: in prose sentences, never in a form row. So the model had no reason to learn the difference.
#:
#: The cascade is what makes this worth a guard rather than a data fix alone: masking "Pain"
#: deletes the very cue ``_SCALE_REJECT_BEFORE`` looks backwards for, so the score "7/10" then
#: gets flagged as a surviving DATE by the Stage G rescan and the whole document is blocked.
#: One over-redaction manufactured fourteen phantom leaks.
#:
#: Matched against the *entire* stripped surface, case-insensitively, so a real surname that
#: merely contains one of these ("Painter", "Temple", "Scoresby") is untouched.
_CLINICAL_LABELS = frozenset(
    w.lower() for w in (
        # Scored scales and exam vocabulary.
        "pain", "score", "scores", "scored", "scale", "rating", "rated", "severity",
        "numeric", "nrs", "vas", "gcs", "grade", "motor", "sensory", "strength", "power",
        "reflex", "reflexes", "tone", "range", "rom", "dorsiflexion", "plantarflexion",
        "flexion", "extension", "eversion", "inversion", "abduction", "adduction",
        "ehl", "fhl", "oswestry", "disability", "index",
        # Vitals and their units -- the same two-column rows carry these.
        "bp", "hr", "rr", "temp", "temperature", "spo2", "sat", "sats", "o2", "pulse",
        "resp", "respirations", "weight", "height", "bmi", "vitals",
        # Body parts, which appear as row labels beside a graded score with no scale word at
        # all ("Leg    6/10") and are the only thing marking the row as clinical.
        "leg", "back", "neck", "arm", "shoulder", "knee", "hip", "ankle", "wrist", "elbow",
        "foot", "hand", "radicular", "lumbar", "cervical", "thoracic",
        # Document-section headers. Added after a 22-page record came back with
        # ``[ORG_30] RADIOLOGY REPORT``, ``[ORG_33] DISPENSING RECORD``, ``[ORG_27] OPERATIVE``
        # and ``[ORG_36] ASSESSMENT USE ONLY``: an ALLCAPS banner centred on its own line is
        # exactly the shape of the letterhead the tagger *was* trained to call an organisation,
        # and the generator never produced an operative report or a pharmacy printout, so the
        # model had nothing to separate the two. Section names carry no identifier at all.
        "operative", "preoperative", "postoperative", "intraoperative", "report", "record",
        "records", "note", "notes", "assessment", "plan", "impression", "findings", "history",
        "subjective", "objective", "progress", "discharge", "admission", "consultation",
        "referral", "summary", "addendum", "chronology", "dispensing", "radiology",
        "handwritten", "illegible", "signature", "signed", "page", "continued", "use", "only",
        "confidential", "pharmacist", "resources", "authorization", "explanation", "benefits",
        "statement", "invoice", "remittance",
        # Care-setting words that name a service, not a place with a name: ``ORG 'TRAINING
        # RECORD'`` and ``NAME_PATIENT 'Therapy'`` were the last two residue findings on that
        # record. "Physical Therapy" is a department in the same way "Radiology" is.
        "training", "therapy", "therapies", "therapeutic", "rehabilitation", "rehab",
        "physical", "occupational", "speech", "session", "sessions", "visit", "visits",
        # Provenance banners. A synthetic-record disclaimer repeated on all 22 pages
        # (``SYNTHETIC TRAINING RECORD - FICTIONAL PATIENT, PROVIDERS AND FACILITIES - FOR
        # INTERVIEW ASSESSMENT USE ONLY``) is in the letterhead position and reads as an
        # organisation to the tagger. These words describe the document, never a person or place.
        "synthetic", "fictional", "patient", "patients", "provider", "providers", "facility",
        "facilities", "for", "and", "interview", "sample", "example", "demonstration", "test",
        # Procedure and supply vocabulary the tagger read as identifiers: ``[NAME_PATIENT_46]
        # Gelfoam``, ``[NAME_PATIENT_44] rongeurs``, ``[ID_GENERIC_22] 0-Vicryl``,
        # ``[ID_GENERIC_17] 22-gauge``. A finite list cannot cover clinical vocabulary, and is
        # not trying to -- see the ``ponytail`` note under ``is_clinical_label``.
        "gelfoam", "vicryl", "rongeurs", "curette", "gauge", "hemostasis", "haemostasis",
        "fluoroscopy", "fluoroscopic", "electromyography", "needle", "contrast", "irrigation",
        "closure", "anesthesia", "anaesthesia", "sedation", "specimen",
    )
)

#: Tokenises a candidate label: whitespace plus the punctuation a table row carries
#: ("Temp:", "SpO2 (RA)", "Pain/score"). Splitting rather than stripping is what lets the
#: multi-token rule below see three words instead of one unmatched string.
_LABEL_SPLIT = re.compile(r"[\s:;,./()\[\]-]+")


def is_clinical_label(surface: str) -> bool:
    """True when the whole span is a clinical measurement label rather than an identifier.

    Consulted by the merge stage for *every* category and every source. Deliberately not
    limited to name categories: the same rows produced ``ORG`` predictions on "pain" too, and
    a measurement label is not an identifier under any label the tagger might pick.

    Multi-token surfaces match only when **every** token is a label word, because real row
    labels are phrases -- "Oswestry Disability Index", "Numeric Pain Rating", "Pain Score" --
    and a single-word set silently failed all of them. Requiring every token is what keeps a
    two-word person's name out: "Scott Hand" fails on "scott" even though "hand" is a label.

    Pure-digit tokens are **skipped** rather than required to be label words, so long as at
    least one alphabetic token remains and every alphabetic token is a label. That is what
    catches "22-gauge" and "0-Vicryl", which the tagger emitted as ``ID_GENERIC``. It cannot
    widen onto an identifier, because the alphabetic part still has to be entirely label
    vocabulary: "MER-778213" fails on "mer", "PCG-4471902" on "pcg", "Jan 5" on "jan".

    ponytail: a word list is a floor, not a solution -- clinical vocabulary is unbounded and
    the next unfamiliar instrument name will be tagged the same way. The real fix is negatives
    in the generator (operative reports, pharmacy printouts, billing tables), which is a
    retrain; this keeps the demo honest until then.
    """
    tokens = [t for t in _LABEL_SPLIT.split(surface.strip()) if t]
    alpha = [t for t in tokens if not t.isdigit()]
    return bool(alpha) and all(t.casefold() in _CLINICAL_LABELS for t in alpha)


def is_credential_only(surface: str) -> bool:
    """True when a span is nothing but a professional credential.

    ``_CRED_ONLY`` has guarded ``_detect_names`` since it was written, at that one call site.
    The sixth guard in this file to need lifting: on a real record the merge stage trimmed
    "Karine Petrosyan, PT, DPT" against an overlapping span and the remainder ", DPT" became a
    placeholder of its own. Stage G then found the literal text ", DPT" still present on four
    other lines and reported a mapped surface as a leak -- an over-redaction manufacturing a
    phantom breach, the same cascade the "Pain" case produced.

    Leading and trailing punctuation is stripped first, which is the whole point: the remainder
    of a trim arrives as ", DPT", not "DPT".
    """
    stripped = surface.strip().strip(".,;:()[]|-").strip()
    return bool(stripped) and _CRED_ONLY.fullmatch(stripped) is not None



def is_eponym_use(text: str, start: int, end: int) -> bool:
    """True when a capitalised surname is being used as a clinical term, not a person.

    Consulted by the merge stage for *every* name span regardless of which detector
    produced it, so it protects the neural tagger's output too. A title immediately
    before the span always wins -- "Dr. Parkinson" is a person even though "Parkinson's"
    is a disease three words later.
    """
    surface = text[start:end]
    tokens = surface.split()
    if not tokens:
        return False
    last = tokens[-1].strip(".,;:'’s").lower()
    if last not in MEDICAL_EPONYMS:
        return False

    before = text[max(0, start - 16):start]
    if _TITLE_BEFORE.search(before):
        return False

    after = text[end:end + 48]
    # Bare possessive: "Parkinson's" with no title cue is the disease.
    m = re.match(r"(?:'s|’s|s')\s*", after)
    if m:
        follow = re.match(r"\s*([A-Za-z]+)", after[m.end():])
        if follow is None or follow.group(1).lower() in EPONYM_CONTEXT:
            return True
    follow = re.match(r"(?:'s|’s|s')?\s+([A-Za-z]+)", after)
    if follow and follow.group(1).lower() in EPONYM_CONTEXT:
        return True

    prev_words = re.findall(r"[\w/]+", before.lower())
    if prev_words and prev_words[-1] in EPONYM_PRECEDING:
        return True
    return False


# =================================================================================
# Detection
# =================================================================================

def _span(
    text: str,
    start: int,
    end: int,
    category: Category,
    score: float,
    detector: str,
    segments: list[Segment],
    source: Source = Source.RULES,
) -> Span | None:
    surface = text[start:end].strip()
    if not surface:
        return None
    # Re-anchor after stripping whitespace so offsets stay exact.
    lead = len(text[start:end]) - len(text[start:end].lstrip())
    start += lead
    end = start + len(surface)
    return Span(
        start=start,
        end=end,
        category=category,
        text=surface,
        score=score,
        source=source,
        detector=detector,
        in_structured_segment=in_structured_segment(start, segments),
    )


def _emit(spans: list[Span], span: Span | None) -> None:
    if span is not None:
        spans.append(span)


def _detect_contact(text: str, segments: list[Segment]) -> list[Span]:
    out: list[Span] = []
    for m in P_EMAIL.finditer(text):
        _emit(out, _span(text, m.start(), m.end(), Category.EMAIL, 0.99, "email", segments))
    for m in P_URL.finditer(text):
        _emit(out, _span(text, m.start(), m.end(), Category.URL, 0.97, "url", segments))
    for m in P_IPV4.finditer(text):
        if V.valid_ip(m.group(0)):
            _emit(out, _span(text, m.start(), m.end(), Category.IP, 0.95, "ipv4", segments))
    for m in P_IPV6.finditer(text):
        if V.valid_ip(m.group(0)):
            _emit(out, _span(text, m.start(), m.end(), Category.IP, 0.95, "ipv6", segments))

    for m in P_PHONE.finditer(text):
        line_start = text.rfind("\n", 0, m.start()) + 1
        prefix = text[line_start:m.start()]
        is_fax = bool(_FAX_CUE.search(prefix))
        cat = Category.FAX if is_fax else Category.PHONE
        _emit(out, _span(text, m.start(), m.end(), cat, 0.93, "fax" if is_fax else "phone", segments))
    for m in P_PHONE_EXT.finditer(text):
        _emit(out, _span(text, m.start(), m.end(), Category.PHONE, 0.70, "phone_ext", segments))
    return out


def _detect_structured_ids(text: str, segments: list[Segment]) -> list[Span]:
    out: list[Span] = []

    for m in P_SSN.finditer(text):
        if V.valid_ssn(m.group(0)):
            _emit(out, _span(text, m.start(), m.end(), Category.SSN, 0.96, "ssn", segments))

    for name, cat, pattern, score in CUED_IDS:
        for m in _re(pattern).finditer(text):
            s, e = m.span("id")
            token = text[s:e]
            # Every one of these categories is a *number*. Without this the leak self-check,
            # which blanks placeholders to spaces to preserve offsets, glues a cue to the
            # next word -- "CLAIM [ACCOUNT_1]   PRODUCED" rescans as account "PRODUCED" and
            # blocks the forward. An all-alpha token is never an account number.
            if not any(ch.isdigit() for ch in token):
                continue
            if V.is_clinical_code(token):
                continue
            _emit(out, _span(text, s, e, cat, score, name, segments))

    for m in P_NPI_CUED.finditer(text):
        s, e = m.span("id")
        _emit(out, _span(text, s, e, Category.LICENCE, 0.98, "npi_cued", segments))
    for m in P_NPI_BARE.finditer(text):
        s, e = m.span("id")
        if V.valid_npi(m.group("id")):
            # Uncued: the checksum could be coincidence, so score below a cued hit but
            # still above threshold. Recall asymmetry -- we would rather over-mask here.
            _emit(out, _span(text, s, e, Category.LICENCE, 0.70, "npi_luhn", segments))

    for m in P_DEA_CUED.finditer(text):
        s, e = m.span("id")
        _emit(out, _span(text, s, e, Category.LICENCE, 0.97, "dea_cued", segments))
    for m in P_DEA_BARE.finditer(text):
        if V.valid_dea(m.group("id")):
            s, e = m.span("id")
            _emit(out, _span(text, s, e, Category.LICENCE, 0.72, "dea_check", segments))

    for m in P_VIN.finditer(text):
        if V.valid_vin(m.group("id")):
            s, e = m.span("id")
            _emit(out, _span(text, s, e, Category.VEHICLE, 0.98, "vin_check", segments))

    for pattern, det in ((P_ZIP_STATE, "zip_state"), (P_ZIP_CUED, "zip_cued")):
        for m in pattern.finditer(text):
            s, e = m.span("id")
            _emit(out, _span(text, s, e, Category.GEO_ZIP, 0.93, det, segments))

    for m in P_STREET.finditer(text):
        s, e = m.span("addr")
        _emit(out, _span(text, s, e, Category.GEO_STREET, 0.90, "street", segments))
    return out


#: A 4-digit year behind an explicit birth cue. Deliberately narrow: only the cue words that
#: mean "was born", never a bare year and never "since 1929" or "the 1929 criteria".
P_BIRTH_YEAR = re.compile(
    r"(?i)\b(?:born|b\.|birth(?:day|\s*year)?|dob|d\.o\.b\.?|date\s*of\s*birth)"
    r"[\s:,]*(?:in|on|year)?[\s:,]*(?P<yr>1[89]\d{2}|20[0-2]\d)\b"
)


def _detect_dates(text: str, segments: list[Segment]) -> list[Span]:
    out: list[Span] = []

    # A bare year is normally NOT an identifier -- HIPAA permits retaining the year, and masking
    # every "1985" in a note would destroy history-of-present-illness utility. But a year behind a
    # *birth* cue is an age in disguise, and "born in 1929" is an age over 89. The suppression
    # logic already exists (``policy.dates`` calls ``ages.year_is_identifying`` against the note's
    # reference year); it just never ran, because nothing emitted a span for it to act on. So the
    # fix is here, at the detector, not in the policy: require the cue, let policy decide.
    for m in P_BIRTH_YEAR.finditer(text):
        s, e = m.span("yr")
        _emit(out, _span(text, s, e, Category.DATE, 0.88, "birth_year", segments))

    for m in P_DATE_ISO.finditer(text):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if V.plausible_date(y, mo, d):
            _emit(out, _span(text, m.start(), m.end(), Category.DATE, 0.97, "date_iso", segments))

    for m in P_DATE_NUMERIC.finditer(text):
        # 2 is the named separator group, not a number. See ``P_DATE_NUMERIC``.
        a, b, c = int(m.group(1)), int(m.group(3)), V.normalise_year(int(m.group(4)))
        # US month/day first, then day/month as a fallback ordering.
        if V.plausible_date(c, a, b) or V.plausible_date(c, b, a):
            _emit(out, _span(text, m.start(), m.end(), Category.DATE, 0.96, "date_numeric", segments))

    for pattern, det, mi, di, yi in (
        (P_DATE_MDY, "date_mdy", 1, 2, 3),
        (P_DATE_DMY, "date_dmy", 2, 1, 3),
    ):
        for m in pattern.finditer(text):
            mo = V.month_number(m.group(mi))
            if mo and V.plausible_date(int(m.group(yi)), mo, int(m.group(di))):
                _emit(out, _span(text, m.start(), m.end(), Category.DATE, 0.97, det, segments))

    for m in P_DATE_MY.finditer(text):
        mo = V.month_number(m.group(1))
        if mo and V.plausible_date(int(m.group(2)), mo):
            _emit(out, _span(text, m.start(), m.end(), Category.DATE, 0.90, "date_my", segments))

    # Bare month/day: genuinely ambiguous with doses and ratios, so it is guarded and
    # scored low rather than dropped. A missed date is a leak; a masked dose is caught
    # by the reject lists below.
    for m in P_DATE_MD.finditer(text):
        a, b = int(m.group(1)), int(m.group(2))
        if a == 0 or b == 0:
            continue
        if not (V.plausible_date(2000, a, b) or V.plausible_date(2000, b, a)):
            continue
        if _MD_REJECT_AFTER.match(text[m.end():m.end() + 24]):
            continue
        line_start = text.rfind("\n", 0, m.start()) + 1
        prefix = text[line_start:m.start()]
        if _MD_REJECT_BEFORE.search(prefix) or _SCALE_REJECT_BEFORE.search(prefix):
            continue
        _emit(out, _span(text, m.start(), m.end(), Category.DATE, 0.55, "date_md", segments))
    return out


def _detect_ages(text: str, segments: list[Segment]) -> list[Span]:
    out: list[Span] = []
    for pattern, det in ((P_AGE_YO, "age_yo"), (P_AGE_YO_SHORT, "age_yo_short"), (P_AGE_CUED, "age_cued")):
        for m in pattern.finditer(text):
            try:
                age = int(m.group("age"))
            except (ValueError, IndexError):
                continue
            if age > 130:
                continue
            cat = Category.AGE_OVER_89 if age >= 90 else Category.AGE
            score = 0.97 if age >= 90 else 0.90
            _emit(out, _span(text, m.start(), m.end(), cat, score, det, segments))
    for m in P_AGE_OVER_89_WORD.finditer(text):
        _emit(out, _span(text, m.start(), m.end(), Category.AGE_OVER_89, 0.95, "age_over89_word", segments))
    return out


def _detect_names(text: str, segments: list[Segment]) -> list[Span]:
    out: list[Span] = []

    for m in P_TITLE_NAME.finditer(text):
        s, e = m.span("name")
        title = m.group("title").strip().lower()
        cat = Category.NAME_PROVIDER if title in PROVIDER_TITLES else Category.NAME_PATIENT
        # The title itself is not PHI, and keeping "Dr." preserves the clinical role.
        _emit(out, _span(text, s, e, cat, 0.92, "title_name", segments))

    for m in P_NAME_CRED.finditer(text):
        s, e = m.span("name")
        # "K. Petrosyan, PT, DPT" -- the second credential makes the first one look like a
        # name. A credential is never the name it qualifies.
        if _CRED_ONLY.fullmatch(text[s:e]):
            continue
        _emit(out, _span(text, s, e, Category.NAME_PROVIDER, 0.92, "name_credential", segments))

    for m in P_SIG_NAME.finditer(text):
        s, e = m.span("name")
        _emit(out, _span(text, s, e, Category.NAME_PROVIDER, 0.93, "signature_name", segments))

    return out


#: A match made *only* of facility vocabulary ("Medical Center", "Clinic", "Labs") names a
#: kind of place, not a place. Capitalised bare heads like "Clinic:" are field keys and
#: headings far more often than organisations, so masking them is pure utility loss with no
#: identifier removed. Requires at least one token outside the vocabulary -- the proper name.
_FACILITY_ONLY = _re(rf"(?:{_ORG_WEAK}|{_ORG_STRONG})(?: (?:{_ORG_WEAK}|{_ORG_STRONG}))*")

#: Letterhead orgs are set in ALLCAPS, which the mixed-case ``P_ORG`` cannot see at all --
#: and letterhead is where the facility name lives on almost every produced page. Requires a
#: head noun so ALLCAPS section headings ("VITAL SIGNS", "PATIENT ASSESSMENT") are untouched.
P_ORG_ALLCAPS = _re(
    r"\b(?P<org>(?:[A-Z&][A-Z&'’\-]{0,19}\s+){1,5}"
    r"(?:HOSPITALS?|CLINICS?|CENT(?:ER|RE)S?|MEDICAL|MEDICINE|HEALTH|HEALTHCARE|SERVICES|"
    r"INSTITUTES?|ASSOCIATES|PARTNERS|LABORATOR(?:Y|IES)|LABS|PHARMACY|SYSTEM|GROUP|"
    r"PRACTICE|IMAGING|RADIOLOGY|THERAPY|REHABILITATION|SPECIALISTS|PHYSICIANS|"
    r"SURGERY|INFIRMARY|HOSPICE|UNIVERSITY|FOUNDATION|NEUROLOGY|CARDIOLOGY|ONCOLOGY|"
    r"ORTHOPA?EDICS?|SPINE|DIAGNOSTICS|ELECTRODIAGNOSTICS|EMS|CARE|PLAN|LOGISTICS)"
    r"(?:\s+[A-Z&][A-Z&'’\-]{0,19}){0,3})\b"
)


def _detect_orgs(text: str, segments: list[Segment]) -> list[Span]:
    out: list[Span] = []
    for m in P_ORG.finditer(text):
        s, e = m.span("org")
        surface = text[s:e]
        if _FACILITY_ONLY.fullmatch(surface):
            continue
        _emit(out, _span(text, s, e, Category.ORG, 0.85, "org", segments))
    for m in P_ORG_ALLCAPS.finditer(text):
        s, e = m.span("org")
        _emit(out, _span(text, s, e, Category.ORG, 0.80, "org_allcaps", segments))
    return out


def is_bare_facility(surface: str) -> bool:
    """True when a span is made *only* of facility vocabulary and so names no facility.

    ``_FACILITY_ONLY`` has guarded the rule layer since it was written, at the one call site
    inside ``_detect_orgs``. The tagger does not go through that call site: it emitted ``ORG``
    on the lone word "Medical" of "the Medical Center", giving ``[ORG_1] Center`` -- a mask that
    removes no identifier and mangles the sentence. So the predicate is lifted here for the
    merge stage to apply to every source, the fifth guard in this file to need that treatment.

    Case and inner whitespace are normalised because ``_FACILITY_ONLY`` is a mixed-case,
    single-space pattern and a tagger span is neither: letterhead arrives ALLCAPS and a span
    crossing a table cell arrives with a run of spaces. Normalising *only* for this test cannot
    widen it onto a real name -- "PIEDMONT MEDICAL CENTER" still fails the fullmatch on
    "Piedmont", which is the proper name and the actual identifier.
    """
    normalised = " ".join(surface.split()).strip(".,;:")
    if not normalised:
        return False
    return bool(
        _FACILITY_ONLY.fullmatch(normalised)
        or _FACILITY_ONLY.fullmatch(normalised.title())
    )


def _detect_opaque_ids(text: str, segments: list[Segment]) -> list[Span]:
    """HIPAA category 18 catch-all. Scored low on purpose -- see ``validators`` for why
    this is a net rather than a guarantee."""
    out: list[Span] = []
    for m in P_OPAQUE.finditer(text):
        token = m.group(0)
        if not V.looks_like_opaque_id(token):
            continue
        _emit(out, _span(text, m.start(), m.end(), Category.ID_GENERIC, 0.40, "opaque_id", segments))
    for m in P_LONG_DIGITS.finditer(text):
        # Scored above the opaque-id net: a nine-digit run is a stronger signal than a
        # mixed token, since nothing in clinical prose is nine contiguous digits.
        _emit(out, _span(text, m.start(), m.end(), Category.ID_GENERIC, 0.55, "long_digits", segments))
    return out


def detect(text: str, segments: list[Segment] | None = None) -> list[Span]:
    """Run every rule detector. Overlaps are expected and resolved in ``merge.py``."""
    segs = segments or []
    spans: list[Span] = []
    spans += _detect_contact(text, segs)
    spans += _detect_structured_ids(text, segs)
    spans += _detect_dates(text, segs)
    spans += _detect_ages(text, segs)
    spans += _detect_names(text, segs)
    spans += _detect_orgs(text, segs)
    spans += _detect_opaque_ids(text, segs)
    return spans


def detect_categories(
    text: str, categories: Iterable[Category], segments: list[Segment] | None = None
) -> list[Span]:
    """Rule detection restricted to some categories. Used by the leak self-check, which
    only re-scans for the structurally detectable identifiers."""
    wanted = set(categories)
    return [s for s in detect(text, segments) if s.category in wanted]
