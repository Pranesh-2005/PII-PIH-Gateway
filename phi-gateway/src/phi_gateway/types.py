"""Core types for the PHI gateway.

Kept dependency-free and dataclass-only so every stage of the pipeline can be tested
and serialised without pulling in torch, fastapi, or anything else heavy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum


class Category(str, Enum):
    """PHI categories. The mapping back to the 18 HIPAA Safe Harbor identifiers lives
    in ``HIPAA_CATEGORY`` below, because reporting per-HIPAA-number is a deliverable."""

    NAME_PATIENT = "NAME_PATIENT"
    NAME_PROVIDER = "NAME_PROVIDER"
    NAME_OTHER = "NAME_OTHER"
    ORG = "ORG"
    GEO_STREET = "GEO_STREET"
    GEO_CITY = "GEO_CITY"
    GEO_ZIP = "GEO_ZIP"
    DATE = "DATE"
    AGE = "AGE"
    AGE_OVER_89 = "AGE_OVER_89"
    PHONE = "PHONE"
    FAX = "FAX"
    EMAIL = "EMAIL"
    SSN = "SSN"
    MRN = "MRN"
    HEALTH_PLAN_ID = "HEALTH_PLAN_ID"
    ACCOUNT = "ACCOUNT"
    LICENCE = "LICENCE"
    VEHICLE = "VEHICLE"
    DEVICE = "DEVICE"
    URL = "URL"
    IP = "IP"
    ID_GENERIC = "ID_GENERIC"
    PROFESSION = "PROFESSION"


#: Category -> HIPAA Safe Harbor identifier number (P2.4). Used by the eval report to
#: produce per-HIPAA-category recall, which is what the brief actually asks for.
HIPAA_CATEGORY: dict[Category, int] = {
    Category.NAME_PATIENT: 1,
    Category.NAME_PROVIDER: 1,
    Category.NAME_OTHER: 1,
    Category.ORG: 2,
    Category.GEO_STREET: 2,
    Category.GEO_CITY: 2,
    Category.GEO_ZIP: 2,
    Category.DATE: 3,
    Category.AGE: 3,
    Category.AGE_OVER_89: 3,
    Category.PHONE: 4,
    Category.FAX: 5,
    Category.EMAIL: 6,
    Category.SSN: 7,
    Category.MRN: 8,
    Category.HEALTH_PLAN_ID: 9,
    Category.ACCOUNT: 10,
    Category.LICENCE: 11,
    Category.VEHICLE: 12,
    Category.DEVICE: 13,
    Category.URL: 14,
    Category.IP: 15,
    Category.ID_GENERIC: 18,
    Category.PROFESSION: 18,
}

#: HIPAA identifiers 16 and 17 are deliberately out of scope for a text gateway.
#: Declared explicitly rather than quietly omitted (P2.4 asks which were excluded and why).
EXCLUDED_HIPAA: dict[int, str] = {
    16: (
        "Biometric identifiers (finger/voice prints) are binary artefacts, not free-text "
        "tokens. Textual references to them are caught as ID_GENERIC; the biometric data "
        "itself is out of modality."
    ),
    17: (
        "Full-face photographs are out of modality for a text gateway. This is the boundary "
        "at which an attachment/image pipeline would be required."
    ),
}


class Source(str, Enum):
    """Which stage produced a span. Drives arbitration in ``detectors/merge.py``."""

    RULES = "rules"
    FIELD = "field"        # form-field key/value cue (Patient Name: ...)
    NEURAL = "neural"
    STRUCTURAL = "structural"


@dataclass(frozen=True)
class Span:
    """A detected PHI span over the *original* text. Half-open interval [start, end)."""

    start: int
    end: int
    category: Category
    text: str
    score: float
    source: Source
    detector: str = ""
    #: True when the span sits in a header, footer, table, signature block or form field.
    in_structured_segment: bool = False

    def __len__(self) -> int:
        return self.end - self.start

    def overlaps(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end

    def contains(self, other: Span) -> bool:
        return self.start <= other.start and other.end <= self.end


@dataclass
class Segment:
    """A structural region of the note (header, table, signature block, ...)."""

    start: int
    end: int
    kind: str
    def contains_pos(self, pos: int) -> bool:
        return self.start <= pos < self.end


@dataclass
class MappingEntry:
    """One placeholder and everything needed to reverse it.

    ``surfaces`` holds every original surface form that collapsed into this placeholder
    ("Mr. Wood", "Wood", "Wood's"), which is what makes rehydration of a consistent
    pseudonym unambiguous. ``canonical`` is what rehydration actually inserts.
    """

    placeholder: str
    category: Category
    canonical: str
    surfaces: list[str] = field(default_factory=list)
    #: Original character offsets this placeholder replaced, for provenance.
    offsets: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class Mapping:
    """The reversible half of de-identification. Never leaves the vault in plaintext."""

    entries: dict[str, MappingEntry] = field(default_factory=dict)
    #: Deterministic per-patient key used to derive the date offset.
    patient_key: str = ""
    #: Applied date offset in days (0 unless dates.mode == shifted).
    date_shift_days: int = 0
    date_mode: str = "interval_preserving"

    def placeholders(self) -> set[str]:
        return set(self.entries)

    def to_json(self) -> str:
        return json.dumps(
            {
                "entries": {
                    k: {
                        "placeholder": v.placeholder,
                        "category": v.category.value,
                        "canonical": v.canonical,
                        "surfaces": v.surfaces,
                        "offsets": [list(o) for o in v.offsets],
                    }
                    for k, v in self.entries.items()
                },
                "patient_key": self.patient_key,
                "date_shift_days": self.date_shift_days,
                "date_mode": self.date_mode,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> Mapping:
        obj = json.loads(raw)
        entries = {
            k: MappingEntry(
                placeholder=v["placeholder"],
                category=Category(v["category"]),
                canonical=v["canonical"],
                surfaces=v["surfaces"],
                offsets=[tuple(o) for o in v["offsets"]],
            )
            for k, v in obj["entries"].items()
        }
        return cls(
            entries=entries,
            patient_key=obj.get("patient_key", ""),
            date_shift_days=obj.get("date_shift_days", 0),
            date_mode=obj.get("date_mode", "interval_preserving"),
        )


@dataclass
class LeakFinding:
    """Something the self-check found in our own masked output. Each of these is a bug."""

    category: Category
    text: str
    start: int
    end: int
    detector: str


@dataclass
class LeakReport:
    clean: bool
    findings: list[LeakFinding] = field(default_factory=list)

    @property
    def leaked(self) -> bool:
        return not self.clean


@dataclass
class ReviewItem:
    """A span whose confidence landed in the uncertainty band. Masked anyway (recall
    asymmetry) but surfaced for a human to confirm."""

    span: Span
    placeholder: str
    reason: str


@dataclass
class DeidResult:
    masked_text: str
    mapping: Mapping
    spans: list[Span] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    review_queue: list[ReviewItem] = field(default_factory=list)
    leak_report: LeakReport | None = None
    #: Set when selfcheck.on_leak == block and a leak was found. The caller MUST NOT
    #: forward to a foundation LLM when this is True.
    blocked: bool = False
    stats: dict = field(default_factory=dict)

    def summary(self) -> dict:
        by_cat: dict[str, int] = {}
        for s in self.spans:
            by_cat[s.category.value] = by_cat.get(s.category.value, 0) + 1
        return {
            "spans": len(self.spans),
            "placeholders": len(self.mapping.entries),
            "by_category": by_cat,
            "review_queue": len(self.review_queue),
            "leaked": bool(self.leak_report and self.leak_report.leaked),
            "blocked": self.blocked,
            **self.stats,
        }
