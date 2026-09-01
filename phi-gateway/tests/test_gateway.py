"""Acceptance-shaped tests: the properties that, if broken, mean a leak or a broken demo.

Not exhaustive unit coverage -- each test here maps to a claim the assessment makes.
"""

from __future__ import annotations

import time

import pytest

from phi_gateway import deidentify, rehydrate
from phi_gateway.config import load_policy
from phi_gateway.pipeline import deidentify_full
from phi_gateway.policy import ages, dates
from phi_gateway.resolve.cluster import cluster_spans
from phi_gateway.selfcheck import leak_scan
from phi_gateway.types import Category, Mapping, MappingEntry, Source, Span
from phi_gateway.vault.mapping_store import MappingExpired, MappingStore

NOTE = """Patient Name: WOOD, JOHN A          MRN: 4482910
DOB: 03/14/1952                     Age: 72
SSN: 123-45-6789                    Phone: 617-555-0311
Email: jwood52@example.com

Mr. Wood was admitted on 03/14/2024. Dr. Parkinson confirmed Parkinson's
disease on 03/17/2024. Wood improved and was discharged 03/21/2024.
"""


def _span(start, end, cat, text, score=0.9, source=Source.RULES):
    return Span(start=start, end=end, category=cat, text=text, score=score, source=source)


# ------------------------------------------------------------------ round trip

def test_deidentify_returns_masked_text_and_mapping():
    masked, mapping = deidentify(NOTE)
    assert masked != NOTE
    assert mapping.entries


def test_direct_identifiers_do_not_survive():
    masked, _ = deidentify(NOTE)
    for leaked in ("123-45-6789", "4482910", "jwood52@example.com", "617-555-0311"):
        assert leaked not in masked, f"{leaked} survived masking"


def test_same_entity_gets_one_placeholder():
    """The point of consistent pseudonyms: co-reference must survive."""
    result = deidentify_full(NOTE)
    name_entries = [
        e for e in result.mapping.entries.values()
        if e.category in (Category.NAME_PATIENT, Category.NAME_PROVIDER, Category.NAME_OTHER)
    ]
    wood = [e for e in name_entries if any("Wood" in s for s in e.surfaces)]
    assert len(wood) == 1, f"John Wood split across {len(wood)} placeholders: {wood}"


def test_rehydrate_restores_known_placeholders():
    masked, mapping = deidentify(NOTE)
    placeholder = next(iter(mapping.entries))
    restored = rehydrate(f"The value was {placeholder}.", mapping)
    assert mapping.entries[placeholder].canonical in restored


def test_rehydrate_strips_placeholders_never_issued():
    """The LLM echoing an invented token must not be substituted, and must be visible."""
    mapping = Mapping(
        entries={
            "[NAME_PATIENT_1]": MappingEntry(
                placeholder="[NAME_PATIENT_1]",
                category=Category.NAME_PATIENT,
                canonical="John Wood",
                surfaces=["John Wood"],
            )
        }
    )
    out = rehydrate("[NAME_PATIENT_1] met [NAME_PATIENT_99] at [MRN_7].", mapping)
    assert "John Wood" in out
    assert "[NAME_PATIENT_99]" not in out
    assert "[MRN_7]" not in out
    assert out.count("[UNKNOWN_PLACEHOLDER]") == 2


# ------------------------------------------------------------------ ambiguity

def test_eponym_is_not_masked_as_a_person():
    masked, _ = deidentify("The patient has Parkinson's disease and Crohn's disease.")
    assert "Parkinson" in masked
    assert "Crohn" in masked


def test_person_with_an_eponym_surname_is_masked():
    result = deidentify_full("Dr. Parkinson reviewed the scan.")
    assert "Parkinson" not in result.masked_text


def test_both_senses_in_one_sentence():
    """'Dr. Parkinson diagnosed Parkinson's disease' -- the case the brief names."""
    masked, _ = deidentify("Dr. Parkinson diagnosed Parkinson's disease.")
    assert "Parkinson's disease" in masked
    assert masked.count("Parkinson") == 1, masked


# ------------------------------------------------------------------ dates

def test_interval_is_preserved_without_emitting_a_date():
    masked, _ = deidentify("Admitted 03/14/2024. Surgery 03/17/2024.")
    assert "[DATE_1]" in masked
    assert "= DATE_1 + 3d" in masked
    assert "03/14/2024" not in masked and "03/17/2024" not in masked


def test_identical_dates_collapse_to_one_placeholder():
    masked, mapping = deidentify("Seen 03/14/2024, again on March 14, 2024.")
    assert masked.count("[DATE_1]") == 2
    assert len([e for e in mapping.entries.values() if e.category is Category.DATE]) == 1


def test_date_shift_is_deterministic_per_patient():
    a = dates.shift_days("patient-a", 180)
    assert a == dates.shift_days("patient-a", 180)
    assert a != 0 and abs(a) <= 180


def test_shifted_mode_preserves_intervals_exactly():
    policy = load_policy()
    policy.dates.mode = "shifted"
    masked, _ = deidentify("Admitted 03/14/2024. Surgery 03/17/2024.", policy=policy)
    import re
    from datetime import date

    found = [date.fromisoformat(d) for d in re.findall(r"\d{4}-\d{2}-\d{2}", masked)]
    assert len(found) == 2
    assert (found[1] - found[0]).days == 3


def test_date_parser_rejects_nonsense():
    assert dates.parse("13/45/2024") is None
    assert dates.parse("03/14/2024") == dates.ParsedDate(2024, 3, 14)
    assert dates.parse("March 2024") == dates.ParsedDate(2024, 3, None)


def test_dob_does_not_become_the_interval_anchor():
    """Anchoring on a DOB decades earlier turns every clinical interval into "+26298d":
    useless to the LLM, and an interval that hands the birth year straight back."""
    masked, _ = deidentify("DOB: 03/14/1952. Admitted 03/14/2024. Surgery 03/17/2024.")
    assert "26298" not in masked
    assert "+ 3d" in masked
    # The bare year *is* retained -- Safe Harbor permits it below age 90, and the >89 case is
    # handled by ages.year_is_identifying, not here.


# ------------------------------------------------------------------ ages

def test_age_under_90_is_retained():
    masked, _ = deidentify("The patient is a 72-year-old male.")
    assert "72" in masked


def test_age_over_89_is_bucketed():
    masked, _ = deidentify("The patient is a 93-year-old male.")
    assert "93" not in masked
    assert "[AGE_OVER_89]" in masked


def test_nonagenarian_counts_as_an_age_over_89():
    assert ages.is_over_89("nonagenarian")
    assert ages.is_over_89("in her 90s")
    assert not ages.is_over_89("72")


def test_birth_year_implying_over_89_is_identifying():
    assert ages.year_is_identifying(1931, 2024)
    assert not ages.year_is_identifying(1952, 2024)


# ------------------------------------------------------------------ clustering

def test_surface_variants_cluster():
    text = "Mr. Wood ... Wood ... John Wood ... Wood's"
    spans = [
        _span(0, 8, Category.NAME_PATIENT, "Mr. Wood"),
        _span(13, 17, Category.NAME_PATIENT, "Wood"),
        _span(22, 32, Category.NAME_PATIENT, "John Wood"),
        _span(37, 43, Category.NAME_PATIENT, "Wood's"),
    ]
    assert len(cluster_spans(spans)) == 1


def test_different_people_sharing_a_surname_do_not_cluster():
    spans = [
        _span(0, 9, Category.NAME_PATIENT, "John Wood"),
        _span(20, 29, Category.NAME_PATIENT, "Jane Wood"),
    ]
    assert len(cluster_spans(spans)) == 2


def test_clustering_never_crosses_categories():
    spans = [
        _span(0, 4, Category.NAME_PATIENT, "Wood"),
        _span(10, 14, Category.NAME_PROVIDER, "Wood"),
    ]
    assert len(cluster_spans(spans)) == 2


# ------------------------------------------------------------------ propagation

def test_bare_repeat_of_a_name_is_propagated():
    """No cue on the second mention. The evidence from the first still applies."""
    masked, _ = deidentify("Mr. Wood was admitted. Wood improved overnight.")
    assert "Wood" not in masked


def test_propagation_respects_the_eponym_guard():
    masked, _ = deidentify("Dr. Parkinson called. The patient has Parkinson's disease.")
    assert "Parkinson's disease" in masked


def test_propagation_is_case_insensitive_but_not_case_blind():
    masked, _ = deidentify("Patient Name: WOOD, JOHN A\n\nWood was seen in clinic.\n")
    assert "Wood" not in masked and "WOOD" not in masked


# ------------------------------------------------------------------ self-check

def test_selfcheck_is_clean_on_our_own_output():
    result = deidentify_full(NOTE)
    assert result.leak_report is not None
    assert result.leak_report.clean, result.leak_report.findings
    assert not result.blocked


def test_selfcheck_catches_an_identifier_we_failed_to_mask():
    policy = load_policy()
    planted = "Contact [NAME_PATIENT_1] at 617-555-0311 or ssn 123-45-6789."
    report = leak_scan.scan(planted, Mapping(), policy)
    assert report.leaked
    assert {f.category for f in report.findings} >= {Category.PHONE, Category.SSN}


def test_selfcheck_catches_a_mapped_surface_left_behind():
    policy = load_policy()
    mapping = Mapping(
        entries={
            "[NAME_PATIENT_1]": MappingEntry(
                placeholder="[NAME_PATIENT_1]",
                category=Category.NAME_PATIENT,
                canonical="John Wood",
                surfaces=["John Wood"],
            )
        }
    )
    report = leak_scan.scan("[NAME_PATIENT_1] arrived. John Wood signed.", mapping, policy)
    assert report.leaked
    assert any(f.detector.startswith("residue") for f in report.findings)


def test_a_mapped_name_reused_as_an_eponym_is_not_residue():
    """"Dr. Parkinson" is masked, so "Parkinson" is a mapped surface. The retained
    "Parkinson's disease" is not a leak of it -- and if the residue check thinks it is,
    every note containing both senses gets blocked."""
    policy = load_policy()
    mapping = Mapping(
        entries={
            "[NAME_PROVIDER_1]": MappingEntry(
                placeholder="[NAME_PROVIDER_1]",
                category=Category.NAME_PROVIDER,
                canonical="Parkinson",
                surfaces=["Parkinson"],
            )
        }
    )
    report = leak_scan.scan(
        "[NAME_PROVIDER_1] confirmed Parkinson's disease.", mapping, policy
    )
    assert report.clean, report.findings


def test_placeholders_do_not_trip_the_scanner():
    scrubbed = leak_scan.blank_placeholders("[DATE_2 = DATE_1 + 3d] and [MRN_1]")
    assert "DATE" not in scrubbed and "MRN" not in scrubbed
    assert len(scrubbed) == len("[DATE_2 = DATE_1 + 3d] and [MRN_1]")   # offsets preserved


def test_ordinary_brackets_in_an_llm_reply_are_not_stripped():
    mapping = Mapping()
    assert rehydrate("The note said [sic] and [IMPORTANT].", mapping) == (
        "The note said [sic] and [IMPORTANT]."
    )


def test_clinical_vocabulary_survives_the_category_18_catch_all():
    """"ICD-10" is letters+digits at identifier length. Masking it is the other failure."""
    masked, _ = deidentify("Coded as ICD-10 G20 per the problem list.")
    assert "ICD-10" in masked
    assert "G20" in masked


def test_an_uncued_bare_digit_run_is_not_left_behind():
    """The catch-all needs letters AND digits, P_PHONE needs a separator, and the NPI
    checksum only coincides ~1 in 10. So an unformatted all-numeric identifier used to
    fall through every detector."""
    masked, _ = deidentify("Callback 6175550311, chart 448291077.")
    assert "6175550311" not in masked
    assert "448291077" not in masked


def test_org_detection_does_not_eat_the_next_field_key():
    """P_ORG's tail must stop at a cell separator. Unbounded, it walks out of the
    organisation and swallows the following column's key."""
    masked, _ = deidentify("Clinic: Wood Memorial Family Practice          Attending: E. Smith, MD\n")
    assert "Memorial" not in masked          # the organisation itself is masked
    assert "Attending" in masked             # the next field's key is not


def test_a_bare_facility_type_is_not_an_organisation():
    """"Clinic" and "Medical Center" name a kind of place, not a place. Masking them
    removes no identifier and costs utility."""
    masked, _ = deidentify("Seen in Clinic. Referred to the Medical Center for imaging.")
    assert "Clinic" in masked
    assert "Medical Center" in masked


# ------------------------------------------------------------------ vault

def test_vault_round_trips_and_does_not_store_plaintext():
    store = MappingStore(ttl_seconds=60)
    _, mapping = deidentify(NOTE)
    sid = store.put(mapping)
    blob = b"".join(r.ciphertext for r in store._records.values())
    assert b"123-45-6789" not in blob
    assert store.get(sid).entries.keys() == mapping.entries.keys()


def test_vault_rejects_another_session_id():
    store = MappingStore(ttl_seconds=60)
    store.put(Mapping())
    with pytest.raises(MappingExpired):
        store.get("not-a-session")


def test_vault_expires():
    store = MappingStore(ttl_seconds=0)
    sid = store.put(Mapping())
    time.sleep(0.01)
    with pytest.raises(MappingExpired):
        store.get(sid)


# ------------------------------------------------------------------ robustness

@pytest.mark.parametrize("text", ["", "   ", "\n\n", "No identifiers here at all."])
def test_degenerate_input_does_not_crash(text):
    masked, mapping = deidentify(text)
    assert isinstance(masked, str)
    assert isinstance(mapping.entries, dict)


def test_structured_segments_are_scanned():
    """P2.7 requires identifiers hidden in tables, headers and signature blocks."""
    result = deidentify_full(
        "Patient Name: WOOD, JOHN A          MRN: 4482910\n\n/s/ E. Smith, MD\n"
    )
    assert "4482910" not in result.masked_text
    assert "WOOD" not in result.masked_text
