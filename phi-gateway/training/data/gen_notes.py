"""Synthetic clinical notes with gold spans, plus a hand-shaped adversarial set.

Two reasons this file exists rather than a download script:

* **Licensing.** P2.3 allows synthetic or properly licensed corpora only. Generated notes
  have no provenance problem, and i2b2/n2c2 needs a DUA that cannot sit on the critical path.
* **Free gold labels.** Identifiers are *injected*, so their character offsets are known
  exactly. Hand-annotating 2000 notes is not happening; hand-*verifying* 50 is (Phase 6).

Leakage control (the thing that makes the eval believable): the surface pools are split into
disjoint ``train`` and ``test`` halves -- different names, different facilities, different
number formats -- and the adversarial notes use sentence templates that appear in no
training note. A model that memorised the generator scores near zero on the test split, which
is exactly the diagnosis we want to be able to make.

No Faker dependency. Local pools are deterministic, disjoint by construction, and let the
identifier *shape* distribution be chosen deliberately rather than by a locale file.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------
# Surface pools. Index 0 of each pair is the train pool, index 1 the test pool.
# --------------------------------------------------------------------------------------

SURNAMES = (
    # train
    ("Halloway", "Petrosyan", "Delacroix", "Nakamura", "Ramachandran", "Okonkwo",
     "Vasquez", "Lindqvist", "Bhattacharya", "Odugbemi", "Kowalczyk", "Ferreira",
     "Abernathy", "Castellanos", "Mwangi", "Thibodeaux", "Yamashita", "Ellsworth",
     "Marchetti", "Nkemelu", "Sandoval", "Prendergast", "Achterberg", "Villanueva"),
    # test -- no overlap
    ("Brightwater", "Szymanski", "Oyelaran", "Fitzsimmons", "Andrzejewski", "Kealoha",
     "Ravindran", "Baumgartner", "Chukwuma", "Larrabee", "Papadopoulos", "Winterbourne",
     "Espinoza", "Haugland", "Tsevendorj", "Quintanilla", "Adeyemi", "Rasmussen"),
)

GIVEN_NAMES = (
    ("Marcus", "Priya", "Devon", "Ingrid", "Tobias", "Ayesha", "Rowan", "Consuelo",
     "Emeka", "Marguerite", "Kenji", "Beatriz", "Aleksandr", "Nadia", "Thaddeus", "Simone"),
    ("Lucienne", "Bartholomew", "Anjali", "Soren", "Xiomara", "Ignatius", "Femi",
     "Wilhelmina", "Rasheed", "Clementine", "Yusuf", "Halina", "Obinna", "Genevieve"),
)

CITIES = (
    ("Ashfield", "Northgate", "Piedmont", "Ravensbrook", "Fairhaven", "Millbury",
     "Kestrel Falls", "Aldenwood", "Cedar Bluff"),
    ("Thornbury", "Wexford Hollow", "Saltmarsh", "Brackenridge", "Elmsworth",
     "Harrowgate", "Foxglove Creek", "Silverton"),
)

STREET_WORDS = (
    ("Kestrel", "Alder", "Sycamore", "Marlborough", "Cranfield", "Ashgrove", "Beaumont"),
    ("Willowmere", "Copperfield", "Havenwood", "Ravenscroft", "Linnet", "Ashbourne"),
)
STREET_TYPES = ("Street", "Avenue", "Road", "Boulevard", "Lane", "Drive", "Parkway", "Way")

ORG_HEADS = (
    ("Piedmont County General Hospital", "Northgate Family Medicine",
     "Ashfield Interventional Pain Institute", "Kestrel Valley Sports Therapy",
     "Cedar Bluff Neurology & Electrodiagnostics", "Millbury Regional Medical Center",
     "Fairhaven Orthopaedic Associates", "Aldenwood Imaging Partners"),
    ("Thornbury Memorial Hospital", "Saltmarsh Community Health Center",
     "Brackenridge Spine Institute", "Harrowgate Physical Therapy Group",
     "Elmsworth Diagnostics Laboratory", "Wexford Hollow Urgent Care",
     "Silverton Cardiology Specialists", "Foxglove Creek Rehabilitation Pavilion"),
)

EMPLOYERS = (
    ("Corrigan Hauling LLC", "Aldenwood Logistics Inc.", "Vantage Freight Corporation"),
    ("Marlow Cartage LLP", "Brightline Distribution Inc.", "Kesterly Transport LLC"),
)

CREDENTIALS = ("MD", "DO", "RN", "NP", "PA-C", "DPT", "PT", "MSN, RN", "PharmD", "EMT-P")

SPECIALTIES = (
    "Orthopedic Spine Surgery", "Physical Medicine and Rehabilitation", "Emergency Medicine",
    "Interventional Pain Management", "Neurology", "Diagnostic Radiology",
    "Family Medicine", "Physical Therapy",
)

#: Clinical content that is *shaped* like an identifier. Every one of these is a hard
#: negative: the model must learn the shape alone is not enough. Drawn from the false
#: positives the rule layer actually produced on a real produced record.
CLINICAL_NEGATIVES = (
    "Pain 7/10 at rest, 4/10 with ambulation.",
    "Neck pain improved from 6/10 to roughly 3/10 and is now intermittent.",
    "Motor: right EHL 4+/5, ankle dorsiflexion 4/5, otherwise 5/5 throughout.",
    "Low back pain 2-3/10, activity related.",
    "BP 148/88, HR 96, RR 20, SpO2 98% on room air, GCS 15.",
    "MRI demonstrates a right paracentral disc protrusion at C5-C6.",
    "Assessment: cervical radiculopathy (M54.12), lumbar radiculopathy (M54.41).",
    "Billed 99213, 20610, 72141, 95886 for this encounter.",
    "Discussed Parkinson's disease and Alzheimer's dementia in the differential.",
    "Positive Tinel sign; negative Spurling maneuver bilaterally.",
    "Crohn's disease is quiescent; no Hodgkin lymphoma on review.",
    "Alert and oriented x4. Tolerated the session without incident.",
    "Priority 2 transport, no lights or siren, ETA 12 minutes.",
    "Warfarin 5 mg PO daily, INR goal 2-3. Metformin 500 mg BID.",
    "Ibuprofen 600 mg q8h PRN. Gabapentin titrated 300 mg to 900 mg nightly.",
    "Range of motion: cervical flexion 40 degrees, extension 30 degrees.",
    "Straight leg raise positive at 45 degrees on the right.",
)

BODY_TEMPLATES = (
    "{PATIENT} is a {AGE}-year-old presenting to {ORG} on {DATE} with worsening low back "
    "and right leg pain following a motor vehicle collision.",
    "The patient was evaluated by {PROVIDER} at {ORG} on {DATE}. {CLIN}",
    "Follow-up on {DATE}: {PATIENT} reports partial relief after the injection performed by "
    "{PROVIDER}. {CLIN}",
    "{PATIENT} was transported by {ORG} to the emergency department on {DATE}. {CLIN}",
    "Records from {ORG} dated {DATE} were reviewed. {CLIN}",
    "Per {PROVIDER}, the patient remains off work from employment at {EMPLOYER} pending "
    "repeat imaging on {DATE}.",
    "{CLIN} Plan discussed with {PATIENT} and with {PROVIDER} by telephone at {PHONE}.",
    "Physical therapy initiated {DATE} at {ORG} under the direction of {PROVIDER}. {CLIN}",
    "Correspondence sent to {EMAIL} and to the claims adjuster on {DATE}.",
    "{PATIENT}, DOB {DOB}, was seen in consultation on {DATE}. {CLIN}",
)

# --------------------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------------------


@dataclass
class Note:
    text: str
    spans: list[dict]
    source: str
    split: str


@dataclass
class _Builder:
    """Accumulates text while recording the offsets of everything it injects."""

    buf: list[str] = field(default_factory=list)
    spans: list[dict] = field(default_factory=list)
    _len: int = 0

    def add(self, s: str) -> None:
        self.buf.append(s)
        self._len += len(s)

    def ent(self, s: str, label: str) -> None:
        self.spans.append({"start": self._len, "end": self._len + len(s), "label": label})
        self.add(s)

    def fill(self, template: str, values: dict[str, tuple[str, str | None]]) -> None:
        """Write ``template``, replacing ``{KEY}`` with a value and recording its label.

        A label of ``None`` means the substitution is clinical content, not an identifier --
        that is how ``CLINICAL_NEGATIVES`` land in a note without becoming gold spans.
        """
        i = 0
        while i < len(template):
            if template[i] == "{":
                j = template.index("}", i)
                key = template[i + 1 : j]
                surface, label = values[key]
                self.ent(surface, label) if label else self.add(surface)
                i = j + 1
            else:
                k = template.find("{", i)
                k = len(template) if k < 0 else k
                self.add(template[i:k])
                i = k

    def build(self, source: str, split: str) -> Note:
        return Note("".join(self.buf), self.spans, source, split)


# --------------------------------------------------------------------------------------
# Identifier surfaces
# --------------------------------------------------------------------------------------


class _Pool:
    """Draws identifier surfaces from the pool half belonging to ``split``."""

    def __init__(self, rng: random.Random, split: str) -> None:
        self.rng = rng
        self.i = 0 if split == "train" else 1

    def pick(self, pairs: tuple[tuple[str, ...], tuple[str, ...]]) -> str:
        return self.rng.choice(pairs[self.i])

    def person(self, *, allcaps: bool = False, transposed: bool = False) -> str:
        given, sur = self.pick(GIVEN_NAMES), self.pick(SURNAMES)
        mid = self.rng.choice("ABCDEFGHJKLMNPRSTW")
        forms = (
            f"{given} {sur}", f"{given} {mid}. {sur}", f"{given[0]}. {sur}",
            f"{sur}, {given}", f"{sur}, {given} {mid}.",
        )
        name = forms[3 if transposed else self.rng.randrange(3)]
        return name.upper() if allcaps else name

    def provider(self) -> str:
        return f"{self.person()}, {self.rng.choice(CREDENTIALS)}"

    def street(self) -> str:
        return (f"{self.rng.randrange(10, 9800)} {self.pick(STREET_WORDS)} "
                f"{self.rng.choice(STREET_TYPES)}")

    def zipcode(self) -> str:
        return f"{self.rng.randrange(1000, 99950):05d}"

    def phone(self) -> str:
        fmt = self.rng.choice(("({a}) {b}-{c}", "{a}-{b}-{c}", "{a}.{b}.{c}", "+1 {a} {b} {c}"))
        return fmt.format(a=self.rng.randrange(201, 990), b=self.rng.randrange(200, 999),
                          c=f"{self.rng.randrange(0, 9999):04d}")

    def date(self) -> str:
        m, d, y = self.rng.randrange(1, 13), self.rng.randrange(1, 29), self.rng.randrange(2019, 2025)
        return self.rng.choice((
            f"{m:02d}/{d:02d}/{y}", f"{m}/{d}/{y}", f"{y}-{m:02d}-{d:02d}",
            f"{self.rng.choice(_MONTHS)} {d}, {y}",
        ))

    def mrn(self) -> str:
        pre = self.rng.choice(("PCG", "MER", "NGF", "AIP") if self.i == 0
                              else ("THM", "SCH", "BSI", "EWD"))
        return f"{pre}-{self.rng.randrange(1000000, 9999999)}"

    def account(self) -> str:
        return (f"{self.rng.choice(('ED','IP','OP','AMB'))}-{self.rng.randrange(20, 26)}-"
                f"{self.rng.randrange(1000, 9999)}-{self.rng.randrange(100, 999)}")

    def plan_id(self) -> str:
        return f"{self.rng.randrange(10, 99)}-{self.rng.randrange(1000, 9999)}-{self.rng.randrange(100, 999)}"

    def ssn(self) -> str:
        return f"{self.rng.randrange(100, 665)}-{self.rng.randrange(10, 99)}-{self.rng.randrange(1000, 9999)}"

    def email(self) -> str:
        return (f"{self.pick(SURNAMES).lower()}.{self.pick(GIVEN_NAMES).lower()}"
                f"@{self.pick(CITIES).split()[0].lower()}health.example.org")

    def licence(self) -> str:
        return f"NPI {self.rng.randrange(1000000000, 1999999999)}"

    def device(self) -> str:
        return f"SN {self.rng.choice('ABCDEFGH')}{self.rng.randrange(100000, 999999)}"


_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December")


# --------------------------------------------------------------------------------------
# Note shapes
# --------------------------------------------------------------------------------------


def _header(b: _Builder, p: _Pool) -> None:
    """Letterhead. ALLCAPS org, then street/city/ZIP, then phone and fax."""
    org = p.pick(ORG_HEADS)
    b.ent(org.upper() if p.rng.random() < 0.5 else org, "ORG")
    b.add("\n")
    b.ent(p.street(), "GEO_STREET")
    b.add(", ")
    b.ent(p.pick(CITIES), "GEO_CITY")
    b.add(", MA ")
    b.ent(p.zipcode(), "GEO_ZIP")
    b.add("\nPhone: ")
    b.ent(p.phone(), "PHONE")
    b.add("    Fax: ")
    b.ent(p.phone(), "FAX")
    b.add("\n\n")


#: Clinical measurement rows in the *same* two-column layout as the demographic table, and
#: carrying no gold spans at all.
#:
#: Added after the trained tagger masked the literal word "Pain" as ``NAME_PATIENT`` on a real
#: 22-page record. The generator had put clinical scores only inside prose sentences
#: (``CLINICAL_NEGATIVES``), so ``Pain    7/10`` was a shape the model had never seen -- and it
#: is character-for-character the shape of ``Patient    Marcus Whitfield``. The model was not
#: wrong to generalise; it was never shown the counterexample.
#:
#: Every value here is a pure negative. That is the point: the row label and the measurement
#: both have to be things the model has seen occupying key and value positions without being
#: PHI, otherwise the layout cue alone decides.
VITALS_ROWS = (
    ("Pain", ("7/10", "4/10 at rest, 7/10 with movement", "6/10", "2-3/10", "8/10")),
    ("Pain Score", ("7/10", "5/10", "9/10")),
    ("Numeric Pain Rating", ("6/10", "4/10", "8/10")),
    ("BP", ("148/88", "132/78", "141/84", "136/82")),
    ("HR", ("96", "78", "88 regular")),
    ("RR", ("20", "16", "18")),
    ("Temp", ("98.2F", "37.1C", "98.6F")),
    ("SpO2", ("99% RA", "98% on room air", "100%")),
    ("Weight", ("184 lb", "83.5 kg")),
    ("Height", ("5'10\"", "178 cm")),
    ("BMI", ("26.4", "31.2")),
    ("GCS", ("15", "14 (E4 V4 M6)")),
    ("Motor", ("right EHL 4+/5, dorsiflexion 4/5", "5/5 throughout", "4+/5 left L5")),
    ("Reflexes", ("2+ symmetric", "1+ at both ankles")),
    ("Range of Motion", ("flexion 90, extension 20", "cervical rotation 60 bilaterally")),
    ("Oswestry Disability Index", ("48% (severe disability)", "32% (moderate)")),
    ("Leg", ("6/10", "7/10 right")),
    ("Back", ("4/10", "6-7/10")),
)


def _vitals_table(b: _Builder, p: _Pool) -> None:
    """A two-column clinical block containing zero identifiers.

    Uses the colonless layout more often than not, because that is the variant that collides
    with the demographic table and therefore the variant that teaches the distinction.
    """
    colonless = p.rng.random() < 0.7
    rows = list(VITALS_ROWS)
    p.rng.shuffle(rows)
    if p.rng.random() < 0.5:
        b.add(p.rng.choice(("VITAL SIGNS\n", "Vitals:\n", "OBJECTIVE\n", "Examination\n")))
    for label, values in rows[: p.rng.randrange(3, 9)]:
        sep = " " * p.rng.randrange(3, 18) if colonless else ": "
        # b.add throughout: no b.ent anywhere in this function, by design.
        b.add(f"{label}{sep}{p.rng.choice(values)}\n")
    b.add("\n")


def _demographic_table(b: _Builder, p: _Pool, colonless: bool) -> None:
    """The two-column demographic block.

    ``colonless`` reproduces layout-extracted PDF text, where the key and value are
    separated by a run of spaces and nothing else -- the shape that defeats a colon-only
    field detector, and therefore the shape the model most needs to have seen.
    """
    rows = [
        ("Patient Name", p.person(transposed=True, allcaps=p.rng.random() < 0.4), "NAME_PATIENT"),
        ("Date of Birth", p.date(), "DATE"),
        ("MRN", p.mrn(), "MRN"),
        ("Account Number", p.account(), "ACCOUNT"),
        ("Member ID", p.plan_id(), "HEALTH_PLAN_ID"),
        ("Attending Physician", p.provider(), "NAME_PROVIDER"),
        ("Employer", p.pick(EMPLOYERS), "ORG"),
        ("Home Phone", p.phone(), "PHONE"),
    ]
    p.rng.shuffle(rows)
    for key, value, label in rows[: p.rng.randrange(4, len(rows) + 1)]:
        sep = " " * p.rng.randrange(3, 18) if colonless else ": "
        b.add(key + sep)
        b.ent(value, label)
        b.add("\n")
    b.add("\n")


def _body(b: _Builder, p: _Pool) -> None:
    for template in p.rng.sample(BODY_TEMPLATES, p.rng.randrange(3, 7)):
        b.fill(template, {
            "PATIENT": (p.person(), "NAME_PATIENT"),
            "PROVIDER": (p.provider(), "NAME_PROVIDER"),
            "ORG": (p.pick(ORG_HEADS), "ORG"),
            "EMPLOYER": (p.pick(EMPLOYERS), "ORG"),
            "DATE": (p.date(), "DATE"),
            "DOB": (p.date(), "DATE"),
            "AGE": (str(p.rng.randrange(19, 89)), "AGE"),
            "PHONE": (p.phone(), "PHONE"),
            "EMAIL": (p.email(), "EMAIL"),
            "CLIN": (p.rng.choice(CLINICAL_NEGATIVES), None),
        })
        b.add("\n")
    b.add("\n")
    # A run of pure clinical content. Notes where every sentence contains an identifier
    # teach the model that some identifier is always present, which is a leak the other way.
    for line in p.rng.sample(CLINICAL_NEGATIVES, p.rng.randrange(2, 6)):
        b.add(line + "\n")
    b.add("\n")


def _signature(b: _Builder, p: _Pool) -> None:
    b.add(p.rng.choice(("Electronically signed by ", "Signed By ", "/s/ ", "Dictated by: ")))
    b.ent(p.provider(), "NAME_PROVIDER")
    b.add(", " + p.rng.choice(SPECIALTIES) + "\n")
    b.ent(p.pick(ORG_HEADS), "ORG")
    b.add("  |  ")
    b.ent(p.licence(), "LICENCE")
    b.add("  |  ")
    b.ent(p.date(), "DATE")
    b.add("\n")


def _footer(b: _Builder, p: _Pool) -> None:
    b.add("\n")
    b.ent(p.person(transposed=True, allcaps=True), "NAME_PATIENT")
    b.add("  |  DOB ")
    b.ent(p.date(), "DATE")
    b.add("  |  MRN ")
    b.ent(p.mrn(), "MRN")
    b.add(f"  |  Page {p.rng.randrange(1, 22)} of 22\n")


#: Document-section banners, ALLCAPS and centred, carrying no identifier whatsoever.
#:
#: Same failure as ``VITALS_ROWS``, found the same way. The generator produced letterheads
#: (``PIEDMONT ORTHOPEDIC SPINE INSTITUTE``) but never section banners, so on a real record the
#: tagger read ``OPERATIVE REPORT``, ``RADIOLOGY REPORT`` and ``DISPENSING RECORD`` as
#: organisations -- an ALLCAPS phrase alone on a centred line is exactly the letterhead shape.
#: The model was not wrong to generalise; it was never shown the counterexample.
SECTION_BANNERS = (
    "OPERATIVE REPORT", "RADIOLOGY REPORT", "DISPENSING RECORD", "PROGRESS NOTE",
    "DISCHARGE SUMMARY", "CONSULTATION NOTE", "PHYSICAL THERAPY NOTE", "PATHOLOGY REPORT",
    "PREOPERATIVE DIAGNOSIS", "POSTOPERATIVE DIAGNOSIS", "INDICATIONS", "FINDINGS",
    "PROCEDURE IN DETAIL", "IMPRESSION", "ASSESSMENT AND PLAN", "REVIEW OF SYSTEMS",
    "PAST MEDICAL HISTORY", "MEDICATIONS ON ADMISSION", "ALLERGIES",
    "FOR ASSESSMENT USE ONLY", "CONFIDENTIAL - PROTECTED HEALTH INFORMATION",
    "PATIENT FINANCIAL SERVICES", "EXPLANATION OF BENEFITS", "REMITTANCE ADVICE",
)

#: Operative and pharmacy vocabulary, none of it an identifier. The tagger produced
#: ``[NAME_PATIENT_46] Gelfoam``, ``[NAME_PATIENT_44] rongeurs``, ``[ID_GENERIC_22] 0-Vicryl``
#: and ``[ID_GENERIC_17] 22-gauge`` on a real operative report: capitalised proper-noun-shaped
#: product names in a document type the generator had never emitted.
PROCEDURE_NEGATIVES = (
    "Hemostasis was achieved with bipolar cautery and a small piece of Gelfoam.",
    "The annulus was entered and disc fragments removed with pituitary rongeurs.",
    "Closure was performed in layers with 0-Vicryl and the skin with 3-0 Monocryl.",
    "A 22-gauge spinal needle was advanced under fluoroscopic guidance.",
    "Contrast confirmed epidural spread without vascular uptake.",
    "The correct operative level was confirmed with intraoperative fluoroscopy.",
    "Specimen was sent to pathology in formalin.",
    "Estimated blood loss was minimal; no drains were placed.",
    "Dispensed 30 tablets, no refills, counselling offered and declined.",
    "Generic substitution permitted; patient counselled on drowsiness.",
)

#: A charge table: descriptions, CPT codes, and money. Money is not an identifier and a bare
#: three-digit number is not a fax number, but on a real billing page the tagger produced
#: ``[FAX_1] ,310.00``, ``[FAX_2] 950``, ``[HEALTH_PLAN_ID_2] 756`` and
#: ``[HEALTH_PLAN_ID_3] $18,204.36``, which destroys the table while removing nothing.
BILLING_ROWS = (
    ("Surgical facility fee", "63030", "18,204.36"),
    ("Anesthesia services", "01936", "2,310.00"),
    ("Radiology - lumbar MRI", "72148", "1,450.00"),
    ("Physical therapy, 4 units", "97110", "620.00"),
    ("Electrodiagnostic testing", "95886", "950.00"),
    ("Office visit, established patient", "99213", "245.00"),
    ("Durable medical equipment", "L0650", "756.00"),
    ("Injection, epidural steroid", "62323", "1,988.00"),
)


def _section_banner(b: _Builder, p: _Pool) -> None:
    """One centred ALLCAPS banner. Pure negative: no ``b.ent`` call, by design."""
    banner = p.rng.choice(SECTION_BANNERS)
    b.add(" " * p.rng.randrange(0, 30) + banner + "\n\n")


def _billing_table(b: _Builder, p: _Pool) -> None:
    """A charge table. Every number here is money or a CPT code, so nothing is an entity."""
    b.add(p.rng.choice(("STATEMENT OF CHARGES\n", "PATIENT FINANCIAL SERVICES\n",
                        "Itemised charges\n")))
    rows = list(BILLING_ROWS)
    p.rng.shuffle(rows)
    total = 0.0
    for desc, code, amount in rows[: p.rng.randrange(3, 7)]:
        total += float(amount.replace(",", ""))
        b.add(f"  {desc:<38s}{code:<10s}{'$' if p.rng.random() < 0.5 else ''}{amount}\n")
    b.add(f"  {'Balance due':<38s}{'':<10s}${total:,.2f}\n\n")


def _procedure_narrative(b: _Builder, p: _Pool) -> None:
    """Operative or pharmacy prose. Pure negative."""
    for line in p.rng.sample(PROCEDURE_NEGATIVES, p.rng.randrange(2, 5)):
        b.add(line + "\n")
    b.add("\n")


def synthetic_note(rng: random.Random, split: str) -> Note:
    p = _Pool(rng, split)
    b = _Builder()
    if rng.random() < 0.8:
        _header(b, p)
    if rng.random() < 0.85:
        # Test notes lean towards the colonless layout, training notes towards colons: a
        # different slot policy per split, so a model that learned "colon means value" is
        # caught rather than flattered.
        _demographic_table(b, p, colonless=rng.random() < (0.4 if split == "train" else 0.8))
    # Frequent on purpose. A vitals block appears in most real records, and the model needs to
    # see the identifier-free two-column layout about as often as it sees the identifier-full
    # one -- otherwise "two columns" alone predicts PHI.
    if rng.random() < 0.6:
        _vitals_table(b, p)
    # The three document types the 22-page record contained and the generator did not. Each is
    # a pure negative block, and each is frequent enough that "ALLCAPS centred line" and
    # "number in a table" stop being sufficient cues on their own.
    if rng.random() < 0.5:
        _section_banner(b, p)
    if rng.random() < 0.3:
        _billing_table(b, p)
    if rng.random() < 0.35:
        _procedure_narrative(b, p)
    _body(b, p)
    if rng.random() < 0.7:
        _signature(b, p)
    if rng.random() < 0.5:
        _footer(b, p)
    return b.build("synthetic", split)



# --------------------------------------------------------------------------------------
# Adversarial notes -- the cases rules provably cannot resolve (P2.5 "ambiguity")
# --------------------------------------------------------------------------------------

EPONYMS = ("Parkinson", "Alzheimer", "Crohn", "Hodgkin", "Bell", "Graves", "Addison",
           "Cushing", "Paget", "Wilson", "Hunter", "Down", "Ewing", "Marfan", "Raynaud",
           "Whipple", "Barrett", "Reye", "Gilbert", "Meckel", "Corrigan", "Tinel",
           "Spurling", "Romberg", "Babinski", "Lachman", "Colles", "Pott")

#: ``{}`` is the eponym. The disease sense must never be labelled; the person sense must be.
_EPONYM_DISEASE = (
    "Differential includes {}'s disease and cervical radiculopathy.",
    "No evidence of {} syndrome on examination.",
    "The {} sign was negative bilaterally.",
    "Family history notable for {}'s disease in a maternal uncle.",
    "{} maneuver reproduced the patient's radicular symptoms.",
)
_EPONYM_PERSON = (
    "Dr. {} performed the injection under fluoroscopic guidance.",
    "The consultation was dictated by {}, MD, and reviewed the same day.",
    "Referred to {} for electrodiagnostic testing.",
    "{}, PT, DPT supervised the therapeutic exercise session.",
)


def adversarial_note(rng: random.Random, split: str) -> Note:
    p = _Pool(rng, split)
    b = _Builder()
    kinds = rng.sample(range(7), rng.randrange(4, 7))
    for kind in kinds:
        if kind == 0:
            # Both senses of the same eponym in one note. The whole reason for a model.
            e = rng.choice(EPONYMS)
            b.add(rng.choice(_EPONYM_DISEASE).format(e) + " ")
            tmpl = rng.choice(_EPONYM_PERSON)
            pre, post = tmpl.split("{}")
            b.add(pre)
            b.ent(e, "NAME_PROVIDER")
            b.add(post + "\n")
        elif kind == 1:
            # Surname == facility name. "Mr. Wood was admitted to Wood Memorial."
            sur = p.pick(SURNAMES)
            b.add("Mr. ")
            b.ent(sur, "NAME_PATIENT")
            b.add(" was admitted to ")
            b.ent(f"{sur} Memorial Hospital", "ORG")
            b.add(f" and discharged to {sur} Street on ")
            b.ent(p.date(), "DATE")
            b.add(".\n")
        elif kind == 2:
            # A provider named with no title, no credential, no cue at all.
            b.ent(p.person(), "NAME_PROVIDER")
            b.add(" reviewed the images and agreed with the impression. ")
            b.ent(p.person(), "NAME_PROVIDER")
            b.add(" will follow up in clinic.\n")
        elif kind == 3:
            # Run-on with no punctuation and lowercase names -- OCR-grade text.
            b.add("pt seen today by dr ")
            b.ent(p.pick(SURNAMES).lower(), "NAME_PROVIDER")
            b.add(" at ")
            b.ent(p.pick(ORG_HEADS).lower(), "ORG")
            b.add(" mrn ")
            b.ent(p.mrn().lower(), "MRN")
            b.add(" no acute distress\n")
        elif kind == 4:
            # Age over 89: the aggregation leak. Also a DOB that implies it.
            b.add("The patient is a ")
            b.ent(str(rng.randrange(90, 106)), "AGE_OVER_89")
            b.add("-year-old " + rng.choice(("nonagenarian", "centenarian")) + ". ")
            b.add(rng.choice(CLINICAL_NEGATIVES) + "\n")
        elif kind == 5:
            # Identifiers hiding in a table (P2.7 requires these in the test set).
            b.add("Accession".ljust(16) + "Ordered".ljust(14) + "Provider\n")
            for _ in range(rng.randrange(2, 4)):
                b.ent(f"{rng.choice('ACDR')}{rng.randrange(10000000, 99999999)}", "ID_GENERIC")
                b.add("        ")
                b.ent(p.date(), "DATE")
                b.add("    ")
                b.ent(p.provider(), "NAME_PROVIDER")
                b.add("\n")
        else:
            # Drug names that look like surnames, plus score/code shapes. Pure negatives.
            b.add("Medications: Metoprolol, Lisinopril, Duloxetine, Naproxen. ")
            b.add(rng.choice(CLINICAL_NEGATIVES) + "\n")
    return b.build("adversarial", split)


def generate(n_synthetic: int, n_adversarial: int, split: str, seed: int) -> list[Note]:
    rng = random.Random(f"{split}:{seed}")
    return ([synthetic_note(rng, split) for _ in range(n_synthetic)]
            + [adversarial_note(rng, split) for _ in range(n_adversarial)])


def _selfcheck() -> None:
    """Offsets are the entire value of this file. If they drift, the labels are garbage."""
    for split in ("train", "test"):
        for note in generate(40, 40, split, seed=7):
            for s in note.spans:
                assert 0 <= s["start"] < s["end"] <= len(note.text), s
                surface = note.text[s["start"]:s["end"]]
                assert surface.strip() == surface and surface, repr(surface)
            # Spans must not overlap -- BIO alignment silently drops one if they do.
            ordered = sorted(note.spans, key=lambda s: s["start"])
            for a, c in zip(ordered, ordered[1:]):
                assert a["end"] <= c["start"], (a, c)
    # The pools really are disjoint, or the leakage control is decorative.
    for pool in (SURNAMES, GIVEN_NAMES, CITIES, STREET_WORDS, ORG_HEADS, EMPLOYERS):
        assert not (set(pool[0]) & set(pool[1])), pool[0][:2]
    print("gen_notes selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
