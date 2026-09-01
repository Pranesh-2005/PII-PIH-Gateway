"""The hand-authored half of the held-out test set.

Why this file exists rather than more generator output: everything in ``data/bio/`` came out of
``training/data/gen_notes.py``, so scoring the tagger on it measures how well it learned that
generator. The acceptance gate the brief implies -- a panel hands over an unseen note and reads
the masked text hunting for leaks -- is only tested by notes whose phrasing, layout and traps
share no code with training. These were written by hand, one trap per line, from the list of
hard cases in the brief:

* eponyms used as both a person and a disease (Parkinson, Crohn, Hodgkin, Graves, Bell, Down)
* a patient whose surname is also the organisation's name (Wood / Wood Memorial)
* providers named with no ``Dr.`` cue at all (a signature block, a bare "seen by")
* identifiers hidden in tables, headers, footers and signature blocks
* ages at and over the 89 boundary, plus a birth year that *implies* an age over 89
* clinical vocabulary that looks like PHI (Fahrenheit, Foley, Braden, APGAR, ICU-4B)
* de-identification-resistant formats: ALLCAPS, lowercase, "Last, First", initials, no punctuation

Gold spans are written as ``(label, surface, nth)`` rather than character offsets. Offsets in a
hand-edited literal go stale the moment a word is added; ``nth`` occurrence of an exact surface
does not, and ``build()`` fails loudly if a surface is absent or the count is short. That is the
whole reason for the indirection -- a silently mis-offset gold span scores a correct system as
leaking, which is worse than no test.

``python data/test_set/handauthored.py`` writes ``data/test_set/handauthored.jsonl`` in the same
shape ``eval/harness.py --data`` already reads::

    python eval/harness.py --data data/test_set/handauthored.jsonl --coarse
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "handauthored.jsonl"

#: ``(text, [(label, surface, nth), ...])``. ``nth`` is 1-based over exact, non-overlapping,
#: word-boundary-free ``str.find`` occurrences -- i.e. literal substring order in the note.
NOTES: list[tuple[str, list[tuple[str, str, int]]]] = [
    # ---- 1. the canonical eponym collision, both roles, in one sentence ------------------
    (
        "PROGRESS NOTE\n"
        "Dr. Parkinson reviewed the tremor workup. Parkinson's disease was ruled out.\n"
        "The patient, Ronald Ashgrove, remains on carbidopa-levodopa pending a second read\n"
        "by Parkinson at the movement disorders clinic.\n",
        [
            ("NAME_PROVIDER", "Parkinson", 1),
            ("NAME_PATIENT", "Ronald Ashgrove", 1),
            ("NAME_PROVIDER", "Parkinson", 3),
        ],
    ),
    # ---- 2. surname == organisation name -------------------------------------------------
    (
        "Mr. Wood was admitted to Wood Memorial Hospital on 04/11/2023 under service of\n"
        "the hospitalist team. Wood tolerated the transfer from Wood Memorial's ED without\n"
        "incident. Discharge planning to follow.\n",
        [
            ("NAME_PATIENT", "Wood", 1),
            ("ORG", "Wood Memorial Hospital", 1),
            ("DATE", "04/11/2023", 1),
            ("NAME_PATIENT", "Wood", 3),
            ("ORG", "Wood Memorial", 2),
        ],
    ),
    # ---- 3. provider named with no title cue, in a signature block ------------------------
    (
        "ASSESSMENT/PLAN\n"
        "Continue vancomycin. Repeat trough in 48 hours.\n"
        "\n"
        "Electronically signed:\n"
        "  Marguerite Oyelaran, MD\n"
        "  NPI 1245319599   Pager x4471\n"
        "  Seen by Halvorsen at 0620.\n",
        [
            ("NAME_PROVIDER", "Marguerite Oyelaran", 1),
            ("LICENCE", "1245319599", 1),
            ("NAME_PROVIDER", "Halvorsen", 1),
        ],
    ),
    # ---- 4. PHI buried in a form-field table, ALLCAPS, no prose ---------------------------
    (
        "PATIENT NAME:      QUINTANILLA, ROSA M\n"
        "MRN:               00-4471-882\n"
        "ACCOUNT:           AR9930571\n"
        "DOB:               02/29/1932\n"
        "SSN:               412-88-7391\n"
        "HOME PHONE:        (617) 555-0148\n"
        "EMAIL:             r.quintanilla@example.org\n"
        "PRIMARY INSURER:   MERIDIAN HEALTH PLAN   POLICY H7741920A\n"
        "ATTENDING:         BRIGHTWATER, DEVON\n"
        "UNIT/BED:          ICU-4B / 12\n",
        [
            ("NAME_PATIENT", "QUINTANILLA, ROSA M", 1),
            ("MRN", "00-4471-882", 1),
            ("ACCOUNT", "AR9930571", 1),
            ("DATE", "02/29/1932", 1),
            ("SSN", "412-88-7391", 1),
            ("PHONE", "(617) 555-0148", 1),
            ("EMAIL", "r.quintanilla@example.org", 1),
            ("ORG", "MERIDIAN HEALTH PLAN", 1),
            ("HEALTH_PLAN_ID", "H7741920A", 1),
            ("NAME_PROVIDER", "BRIGHTWATER, DEVON", 1),
        ],
    ),
    # ---- 5. age exactly at and over the boundary, plus an implied one ---------------------
    (
        "89-year-old woman seen in clinic; her 94-year-old brother is her caregiver.\n"
        "A nonagenarian roommate at the facility has the same first name.\n"
        "Born in 1929, the patient predates the hospital itself.\n"
        "Her 72 y/o daughter, Lenore Ashgrove, drove her in.\n",
        [
            ("AGE", "89-year-old", 1),
            ("AGE_OVER_89", "94-year-old", 1),
            ("AGE_OVER_89", "nonagenarian", 1),
            ("DATE", "1929", 1),
            ("AGE", "72 y/o", 1),
            ("NAME_PATIENT", "Lenore Ashgrove", 1),
        ],
    ),
    # ---- 6. clinical vocabulary that reads as PHI -- pure negatives except the two names --
    (
        "Temp 100.4 Fahrenheit. Foley in place, output 40 mL/hr. Braden score 14.\n"
        "APGAR 9 at five minutes. Glasgow Coma Scale 15. Bed ICU-4B, ward 7 East.\n"
        "Cultures sent to the lab at 0430. Crohn's disease, Graves' disease and Bell's palsy\n"
        "all appear in the family history. Reviewed with Bell, the charge nurse, and with\n"
        "the patient's wife, Odalys Kealoha.\n",
        [
            ("NAME_PROVIDER", "Bell", 2),
            ("NAME_PATIENT", "Odalys Kealoha", 1),
        ],
    ),
    # ---- 7. lowercase, no punctuation, run-on -- the OCR-ish worst case -------------------
    (
        "pt is farrukh tsevendorj mrn 7741902 seen 3/4/24 at harrowgate physical therapy\n"
        "group by ravindran for post op knee pain call back at 4135550172 or email\n"
        "f.tsevendorj@example.net address 18 saltmarsh lane wexford hollow ma 02139\n",
        [
            ("NAME_PATIENT", "farrukh tsevendorj", 1),
            ("MRN", "7741902", 1),
            ("DATE", "3/4/24", 1),
            ("ORG", "harrowgate physical therapy\ngroup", 1),
            ("NAME_PROVIDER", "ravindran", 1),
            ("PHONE", "4135550172", 1),
            ("EMAIL", "f.tsevendorj@example.net", 1),
            ("GEO_STREET", "18 saltmarsh lane", 1),
            ("GEO_CITY", "wexford hollow", 1),
            ("GEO_ZIP", "02139", 1),
        ],
    ),
    # ---- 8. header + footer PHI, the position the brief calls out -------------------------
    (
        "THORNBURY MEMORIAL HOSPITAL -- CARDIOLOGY\n"
        "Patient: Fitzsimmons, Clementine   Acct 8830192   Page 1 of 3\n"
        "\n"
        "Echocardiogram shows preserved ejection fraction. No further workup indicated.\n"
        "\n"
        "Printed 06/02/2024 by tech id T-4419 -- Fitzsimmons, Clementine -- do not copy\n",
        [
            ("ORG", "THORNBURY MEMORIAL HOSPITAL", 1),
            ("NAME_PATIENT", "Fitzsimmons, Clementine", 1),
            ("ACCOUNT", "8830192", 1),
            ("DATE", "06/02/2024", 1),
            ("ID_GENERIC", "T-4419", 1),
            ("NAME_PATIENT", "Fitzsimmons, Clementine", 2),
        ],
    ),
    # ---- 9. intervals that must survive the date shift -----------------------------------
    (
        "Admitted 03/14/2024. Taken to the OR 03/17/2024, three days after admission.\n"
        "Discharged 03/21/2024, post-operative day 4. Follow-up 04/04/2024, two weeks out.\n"
        "Patient Ilaria Baumgartner declined home health.\n",
        [
            ("DATE", "03/14/2024", 1),
            ("DATE", "03/17/2024", 1),
            ("DATE", "03/21/2024", 1),
            ("DATE", "04/04/2024", 1),
            ("NAME_PATIENT", "Ilaria Baumgartner", 1),
        ],
    ),
    # ---- 10. every structured-ID family in one signature block ---------------------------
    (
        "DISCHARGE MEDICATIONS -- controlled substances dispensed under DEA BW4471903.\n"
        "Prescriber NPI 1932455684, state licence MA-118-4471.\n"
        "Implanted device serial SN-4471-90A2 (pacemaker, model Aveir VR).\n"
        "Transport arranged, plate 4KLM882, VIN 1HGCM82633A004352.\n"
        "Records portal https://portal.example-health.org/p/rq88x   IP 10.14.22.8\n",
        [
            ("LICENCE", "BW4471903", 1),
            ("LICENCE", "1932455684", 1),
            ("LICENCE", "MA-118-4471", 1),
            ("DEVICE", "SN-4471-90A2", 1),
            ("VEHICLE", "4KLM882", 1),
            ("VEHICLE", "1HGCM82633A004352", 1),
            ("URL", "https://portal.example-health.org/p/rq88x", 1),
            ("IP", "10.14.22.8", 1),
        ],
    ),
    # ---- 11. initials, nicknames and possessives of the same person ----------------------
    (
        "R. Ashgrove was seen in follow-up. Ronnie reports the pain is improved.\n"
        "Ashgrove's wife confirms adherence. Mr. Ashgrove will return in six weeks.\n"
        "Note faxed to 617-555-0199 attention of the referring office.\n",
        [
            ("NAME_PATIENT", "R. Ashgrove", 1),
            ("NAME_PATIENT", "Ronnie", 1),
            ("NAME_PATIENT", "Ashgrove", 2),
            ("NAME_PATIENT", "Ashgrove", 3),
            ("FAX", "617-555-0199", 1),
        ],
    ),
    # ---- 12. drug and device names that look like surnames -- pure negative ---------------
    (
        "Started on Coumadin and Lasix. Zofran PRN. Allergic to Bactrim.\n"
        "Foley and Hickman both in place. Reviewed the Wells score and the Ottawa rules.\n"
        "No family history of note. Discussed with the patient at length.\n",
        [],
    ),
    # ---- 13. two patients in one note -- clustering must not merge them -------------------
    (
        "Twin study intake. Twin A: Adeyemi, Bola -- MRN 5510223.\n"
        "Twin B: Adeyemi, Femi -- MRN 5510224.\n"
        "Both seen 11/08/2023 by Larrabee. Bola's labs are pending; Femi's are resulted.\n",
        [
            ("NAME_PATIENT", "Adeyemi, Bola", 1),
            ("MRN", "5510223", 1),
            ("NAME_PATIENT", "Adeyemi, Femi", 1),
            ("MRN", "5510224", 1),
            ("DATE", "11/08/2023", 1),
            ("NAME_PROVIDER", "Larrabee", 1),
            ("NAME_PATIENT", "Bola", 2),
            ("NAME_PATIENT", "Femi", 2),
        ],
    ),
    # ---- 14. the trap that defeats a rules-only system: a name in prose, no cue at all ----
    (
        "The consult was requested because papadopoulos had failed two prior regimens.\n"
        "Hodgkin lymphoma is on the differential; Hodgkin himself described it in 1832.\n"
        "Case discussed at the tumour board held at Silverton Cardiology Specialists.\n",
        [
            ("NAME_PATIENT", "papadopoulos", 1),
            ("ORG", "Silverton Cardiology Specialists", 1),
        ],
    ),
]


def _resolve(text: str, label: str, surface: str, nth: int) -> dict:
    """Character offsets of the ``nth`` literal occurrence of ``surface``.

    Raises rather than skipping. A gold file that quietly drops an annotation reports a leaking
    system as clean, which is the one failure mode this whole test set exists to catch.
    """
    idx, found = -1, 0
    while found < nth:
        idx = text.find(surface, idx + 1)
        if idx < 0:
            raise ValueError(
                f"{label} {surface!r}: wanted occurrence {nth}, found {found} in note "
                f"starting {text[:40]!r}"
            )
        found += 1
    return {"start": idx, "end": idx + len(surface), "label": label}


def _valid_labels() -> set[str]:
    """The label names the pipeline can actually emit.

    First draft of this file used ``NPI``, ``DEA``, ``LICENSE``, ``DEVICE_ID`` and ``VEHICLE_ID``
    -- none of which are ``Category`` members (the enum spells it ``LICENCE`` and lumps NPI/DEA
    there, and uses ``DEVICE``/``VEHICLE``). The harness matches gold to prediction by label, so
    every one of those spans scored as a miss while the detectors were finding them correctly:
    a gold file that invents its own taxonomy reports a working system as leaking, which is the
    exact failure mode ``_resolve`` was written to prevent. Guarding here, at the one place all
    annotations pass through, is cheaper than trusting fourteen hand-typed literals.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from phi_gateway.types import Category

    return {c.value for c in Category}


def build() -> list[dict]:
    ok = _valid_labels()
    bad = sorted({g[0] for _, gold in NOTES for g in gold} - ok)
    if bad:
        raise ValueError(f"gold labels not in Category: {bad}\nvalid: {sorted(ok)}")
    rows = []
    for text, gold in NOTES:
        spans = sorted(
            (_resolve(text, *g) for g in gold), key=lambda s: (s["start"], s["end"])
        )
        # Overlap check: two gold spans covering the same characters would double-count a
        # recall miss and make the leak rate unreadable.
        for a, b in zip(spans, spans[1:]):
            if b["start"] < a["end"]:
                raise ValueError(f"overlapping gold spans {a} and {b}")
        rows.append(
            {"text": text, "spans": spans, "source": "handauthored", "split": "test"}
        )
    return rows


def main() -> None:
    rows = build()
    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_spans = sum(len(r["spans"]) for r in rows)
    labels: dict[str, int] = {}
    for r in rows:
        for s in r["spans"]:
            labels[s["label"]] = labels.get(s["label"], 0) + 1
    print(f"wrote {OUT}  notes={len(rows)}  spans={n_spans}  labels={len(labels)}")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(labels.items())))


if __name__ == "__main__":
    main()
