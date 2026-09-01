"""Shapes a real produced medical record puts identifiers in.

Every case here comes from a 22-page synthetic litigation record that the rule layer got
wrong at least once. Two of them are the failure the brief calls out on both sides: the
colonless table row was a *leak*, the pain score was *over-redaction*.
"""

from __future__ import annotations

from phi_gateway.detectors import fields, rules
from phi_gateway.detectors.structural import form_fields, segment


def _cats(text: str) -> dict[str, str]:
    spans = rules.detect(text, segment(text)) + fields.detect(text, segment(text))
    return {s.text: s.category.name for s in spans}


def test_colonless_two_column_form():
    """Layout-extracted PDF text has no colons. A colon-only detector leaks the whole page."""
    text = "Patient Name          Whitfield, Marcus D.      MRN            PCG-4471902\n"
    keys = {f.key: f.value for f in form_fields(text)}
    assert keys["Patient Name"] == "Whitfield, Marcus D."
    assert keys["MRN"] == "PCG-4471902"


def test_timestamped_vitals_row_is_not_a_form():
    """Two cells does not mean key/value -- "17:56  BP 148/88" would mask the vitals."""
    assert form_fields("17:56    BP 148/88, HR 96, RR 20\n") == []


def test_clinical_scores_are_not_dates():
    """Bare M/D shape, but the score is the clinical content the note exists to record."""
    lines = [
        "17:56    BP 148/88, HR 96, RR 20, SpO2 98% RA, Pain 7/10\n",
        "Neck pain has improved from 6/10 to roughly 3/10 and is now intermittent.\n",
        "Numeric Pain Rating: 6/10 at rest.\n",
        "Motor: right ankle dorsiflexion 4/5, otherwise 5/5.\n",
    ]
    for line in lines:
        dates = [s.text for s in rules.detect(line, segment(line)) if s.category.name == "DATE"]
        assert not dates, f"{line.strip()!r} -> {dates}"


def test_real_bare_date_still_detected():
    """The scale guard is windowed, so prove it did not swallow bare dates outright."""
    text = "Patient returns to clinic 4/18 for repeat imaging.\n"
    assert "4/18" in _cats(text)


def test_provider_field_keeps_the_specialty():
    """The credential ends the name; the specialty after it is clinical content."""
    text = "Attending Physician: G. Halloway, MD, Orthopedic Spine Surgery\n"
    got = _cats(text)
    assert "G. Halloway, MD" in got
    assert not any("Orthopedic" in k for k in got)


def test_dob_field_does_not_swallow_the_age():
    """"03/14/1987 (Age 36)" -- an age under 90 is retained by policy on purpose."""
    text = "Date of Birth        03/14/1987 (Age 36)\n"
    assert [f.value for f in form_fields(text) if f.key == "Date of Birth"] == ["03/14/1987"]


# --------------------------------------------------------------------------------------
# Merge-stage guards. Both of these were found only once the tagger was switched on: the
# rule layer alone never produced either shape, which is the argument for putting the guard
# in ``merge`` where every source passes through rather than at each detector's emit site.
# --------------------------------------------------------------------------------------


def test_clinical_label_is_never_an_identifier():
    """The tagger masked the literal word "Pain" as NAME_PATIENT in a vitals row.

    ``Pain    7/10`` is character-for-character the shape of ``Patient    Marcus Whitfield``.
    Worse than a cosmetic false positive: masking "Pain" deletes the cue that the date guard
    looks backwards for, so the score is then reported as a surviving DATE by Stage G and the
    document is blocked. One over-redaction manufactured fourteen phantom leaks.
    """
    assert rules.is_clinical_label("Pain")
    assert rules.is_clinical_label("pain")
    assert rules.is_clinical_label("BP")
    assert rules.is_clinical_label("Oswestry Disability Index")
    assert rules.is_clinical_label("Temp:")
    # Surnames that merely contain a label word are identifiers and must survive the guard.
    for name in ("Painter", "Temple", "Scoresby", "Backhouse", "Armstrong"):
        assert not rules.is_clinical_label(name), name


def test_spans_never_land_mid_word_or_on_punctuation():
    """Merge snaps every span, not only overlap remainders.

    Snapping used to run only after a subtraction, so a detector emitting a mid-word span had
    it passed through verbatim. The tagger does that, and produced ``PRO[NAME_PATIENT_1] NOTE``
    (from "PROGRESS"), ``Dr[NAME_PROVIDER_1][NAME_PROVIDER_2]`` (the ". " masked), and
    ``6[DATE_1]7/10`` (the hyphen of a pain range masked as a date).
    """
    from phi_gateway.config import load_policy
    from phi_gateway.detectors.merge import merge
    from phi_gateway.types import Category, Source, Span

    text = "PROGRESS NOTE for Dr. Elena Petrosyan, pain 6-7/10.\n"
    # Offsets are computed, not counted by hand. The first version of this test hardcoded 44 for
    # the hyphen, landed on the "6" instead, and failed with ``[('6', 'DATE')]`` -- a test that
    # asserts the right thing about the wrong offset is a test of nothing.
    gress = text.index("GRESS")
    dot_space = text.index("Dr.") + 2          # the ". " between title and name
    hyphen = text.index("6-7/10") + 1
    # Exactly the three malformed shapes, injected as if a tagger had emitted them.
    bad = [
        Span(start=gress, end=gress + 5, category=Category.NAME_PATIENT,
             text=text[gress:gress + 5],
             score=0.9, source=Source.NEURAL, detector="tagger"),
        Span(start=dot_space, end=dot_space + 2, category=Category.NAME_PROVIDER,
             text=text[dot_space:dot_space + 2],
             score=0.9, source=Source.NEURAL, detector="tagger"),
        Span(start=hyphen, end=hyphen + 1, category=Category.DATE, text=text[hyphen:hyphen + 1],
             score=0.9, source=Source.NEURAL, detector="tagger"),
    ]
    assert [s.text for s in bad] == ["GRESS", ". ", "-"]
    kept = merge(text, bad, load_policy())
    assert kept == [], [(s.text, s.category.name) for s in kept]


def test_eponym_guard_is_not_gated_on_category():
    """The tagger labelled the "Parkinson" of "Parkinson's disease" ``DEVICE``.

    The eponym guard existed and was correct, but was reached only for name categories -- so a
    model that picked any other label walked past it and produced ``[DEVICE_1]'s disease``. The
    guard itself already requires a known eponym in a disease context, so widening it to every
    category costs nothing and closes the bypass.
    """
    from phi_gateway.config import load_policy
    from phi_gateway.detectors.merge import merge
    from phi_gateway.types import Category, Source, Span

    text = "The patient has Parkinson's disease and sees Dr. Parkinson quarterly.\n"
    start = text.index("Parkinson")
    for cat in (Category.DEVICE, Category.NAME_PATIENT, Category.ORG, Category.ID_GENERIC):
        span = Span(start=start, end=start + 9, category=cat, text="Parkinson",
                    score=0.99, source=Source.NEURAL, detector="tagger")
        assert merge(text, [span], load_policy()) == [], cat.name

    # The person of the same name, in the same sentence, must still be masked.
    person = text.index("Parkinson", start + 1)
    span = Span(start=person, end=person + 9, category=Category.NAME_PROVIDER,
                text="Parkinson", score=0.99, source=Source.NEURAL, detector="tagger")
    assert [s.text for s in merge(text, [span], load_policy())] == ["Parkinson"]


def test_a_fragment_of_a_coding_system_token_is_not_an_identifier():
    """"ICD-10" came back as ``ICD-[DATE_1]``: the tagger tagged the version number.

    Guarded by widening the span to its whitespace token and asking ``is_clinical_code``. The
    span must be *strictly smaller* than the token for this to fire, which is the whole safety
    argument -- ``is_clinical_code`` matches ``^\\d{5}$`` for CPT codes and a ZIP is also five
    digits, so a whole-token match would stop masking every ZIP in the corpus.
    """
    from phi_gateway.config import load_policy
    from phi_gateway.detectors.merge import merge
    from phi_gateway.types import Category, Source, Span

    text = "Coded as ICD-10 G20 per the problem list; patient resides in 02134.\n"
    frag = text.index("10")
    span = Span(start=frag, end=frag + 2, category=Category.DATE, text="10",
                score=0.99, source=Source.NEURAL, detector="tagger")
    assert merge(text, [span], load_policy()) == []

    # The ZIP is a whole token, so the same guard must leave it alone.
    zip_at = text.index("02134")
    zip_span = Span(start=zip_at, end=zip_at + 5, category=Category.GEO_ZIP, text="02134",
                    score=0.99, source=Source.RULES, detector="zip")
    assert [s.text for s in merge(text, [zip_span], load_policy())] == ["02134"]


def test_bare_facility_vocabulary_is_not_an_organisation():
    """``[ORG_1] Center`` came from the tagger tagging the "Medical" of "the Medical Center".

    ``_FACILITY_ONLY`` had guarded the rule layer since it was written, at its single call site
    inside ``_detect_orgs`` -- which the tagger does not pass through. Fifth guard in this
    codebase to be bypassed by being attached to a detector instead of to the merge stage.
    """
    assert rules.is_bare_facility("Medical")
    assert rules.is_bare_facility("Medical Center")
    assert rules.is_bare_facility("MEDICAL CENTER")       # letterhead arrives ALLCAPS
    assert rules.is_bare_facility("Clinic:")
    assert rules.is_bare_facility("Medical    Center")     # span crossed a table cell
    # A proper name anywhere in the span makes it a real organisation, and the guard must not
    # touch it -- this is the identifier the ORG category exists for.
    for org in ("Piedmont Medical Center", "PIEDMONT MEDICAL CENTER", "Wood Memorial Hospital",
                "Northgate Family Medicine"):
        assert not rules.is_bare_facility(org), org



def test_a_named_facility_survives_the_bare_facility_guard():
    """The guard that fixed ``[ORG_1] Center`` then leaked ``ORTHOPEDIC SPINE INSTITUTE``.

    On a real letterhead reading ``HALLOWAY ORTHOPEDIC SPINE INSTITUTE`` the tagger emitted the
    proper name and the facility phrase as two spans. Dropping every all-facility span dropped
    the second one, and the record went out as ``[ORG_19] ORTHOPEDIC SPINE INSTITUTE`` with
    Stage G correctly calling it a leak. An over-redaction guard that fires on part of a real
    identifier is worse than the false positive it prevents, so the guard now requires the
    surrounding capitalised run to be facility vocabulary too.
    """
    from phi_gateway.config import load_policy
    from phi_gateway.detectors.merge import merge
    from phi_gateway.types import Category, Source, Span

    def org_span(text: str, surface: str) -> list[Span]:
        at = text.index(surface)
        return [Span(start=at, end=at + len(surface), category=Category.ORG, text=surface,
                     score=0.99, source=Source.NEURAL, detector="tagger")]

    # The leak case: facility vocabulary that is part of a named facility must be masked.
    letterhead = "                    HALLOWAY ORTHOPEDIC SPINE INSTITUTE\n"
    kept = merge(letterhead, org_span(letterhead, "ORTHOPEDIC SPINE INSTITUTE"), load_policy())
    assert [s.text for s in kept] == ["ORTHOPEDIC SPINE INSTITUTE"]

    # The over-redaction case it was written for still has to be dropped.
    prose = "Seen in Clinic. Referred to the Medical Center for imaging.\n"
    assert merge(prose, org_span(prose, "Medical"), load_policy()) == []


def test_a_credential_is_not_a_name():
    """Trimming an overlap left ", DPT" as its own placeholder, and Stage G called it a leak.

    ``_CRED_ONLY`` already existed, at one call site inside ``_detect_names``. Sixth guard in
    this codebase bypassed by living next to the detector that needed it instead of at the merge
    stage every detector passes through.
    """
    assert rules.is_credential_only(", DPT")
    assert rules.is_credential_only("MD")
    assert rules.is_credential_only(" PA-C |")
    # The name the credential qualifies is still a name.
    assert not rules.is_credential_only("Petrosyan, PT")
    assert not rules.is_credential_only("Karine Petrosyan")


def test_billing_numbers_are_not_identifiers():
    """One billing page produced ``[FAX_1] ,310.00``, ``[FAX_2] 950``, ``[HEALTH_PLAN_ID] 756``.

    The rule layer needs a context cue before calling anything a fax; the tagger does not, and a
    charge table is a page of bare numbers. A three-digit span under a category that requires
    seven digits is not a weak identifier, it is not one at all.
    """
    from phi_gateway.config import load_policy
    from phi_gateway.detectors.merge import merge
    from phi_gateway.types import Category, Source, Span

    text = "Adjustment            950      Balance $18,204.36     Ref MER-778213\n"

    def span(surface: str, cat: Category) -> Span:
        at = text.index(surface)
        return Span(start=at, end=at + len(surface), category=cat, text=surface,
                    score=0.99, source=Source.NEURAL, detector="tagger")

    assert merge(text, [span("950", Category.FAX)], load_policy()) == []
    assert merge(text, [span("$18,204.36", Category.HEALTH_PLAN_ID)], load_policy()) == []
    # Real identifiers on the same line must be untouched: alphanumeric, and long enough.
    assert [s.text for s in merge(text, [span("MER-778213", Category.MRN)], load_policy())] \
        == ["MER-778213"]
    # An age is two digits and a real identifier under HIPAA's rules, so the digit floor must
    # not reach it -- AGE is deliberately absent from ``_MIN_DIGITS``.
    age_text = "Age 36 at the time of injury.\n"
    age = Span(start=4, end=6, category=Category.AGE, text="36",
               score=0.99, source=Source.NEURAL, detector="tagger")
    assert [s.text for s in merge(age_text, [age], load_policy())] == ["36"]


def test_section_headers_are_not_organisations():
    """``[ORG_30] RADIOLOGY REPORT``, ``[ORG_27] OPERATIVE``, ``[ORG_33] DISPENSING RECORD``.

    An ALLCAPS banner centred on its own line has the shape of the letterhead the tagger was
    trained to call an organisation, and the generator never produced an operative report or a
    pharmacy printout. A section name carries no identifier.
    """
    for header in ("OPERATIVE", "RADIOLOGY REPORT", "DISPENSING RECORD", "OPERATIVE REPORT",
                   "ASSESSMENT USE ONLY", "Handwritten Note", "22-gauge", "0-Vicryl"):
        assert rules.is_clinical_label(header), header
    # Real organisations and names containing one of those words are not section headers.
    for real in ("Greenleaf Community Pharmacy", "MER-778213", "PCG-4471902", "Jan 5",
                 "Northgate Family Medicine"):
        assert not rules.is_clinical_label(real), real


def test_snapping_keeps_a_real_name_intact():
    """The snap must not eat spans that are already well formed -- it is a guard, not a filter."""
    from phi_gateway.config import load_policy
    from phi_gateway.detectors.merge import merge
    from phi_gateway.types import Category, Source, Span

    text = "Seen by Dr. Elena Petrosyan today.\n"
    span = Span(start=12, end=27, category=Category.NAME_PROVIDER, text=text[12:27],
                score=0.9, source=Source.NEURAL, detector="tagger")
    assert span.text == "Elena Petrosyan"
    kept = merge(text, [span], load_policy())
    assert [s.text for s in kept] == ["Elena Petrosyan"]

