"""Lexicons that exist for one reason: the ambiguity problem (P2.5).

    "Dr. Parkinson diagnosed Parkinson's."
    "Mr. Wood was admitted to Wood Memorial."

One token is PHI, the identical token is not. Rules cannot resolve this in general --
which is exactly why a model is being trained. But two things still belong here:

1.  ``MEDICAL_EPONYMS`` + ``EPONYM_CONTEXT`` let the rule layer *suppress* a name
    detection when a capitalised surname is being used as a disease term. This stops
    the rules-only floor from producing high-confidence false positives that would
    shred clinical meaning.
2.  The same lists drive ``training/data/gen_adversarial.py``, which manufactures these
    collisions on purpose so the tagger sees hundreds of them during training. Hard
    cases do not appear in synthetic data by accident; they have to be built.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------------
# Surnames that are also clinical terms. Non-exhaustive by nature -- the point is to
# cover the collisions dense enough in real notes to matter.
# ---------------------------------------------------------------------------------
MEDICAL_EPONYMS: frozenset[str] = frozenset(
    {
        # neuro
        "parkinson", "alzheimer", "huntington", "guillain", "barre", "barré", "charcot",
        "duchenne", "becker", "friedreich", "wernicke", "korsakoff", "creutzfeldt",
        "jakob", "bell", "horner", "sequard", "tourette", "asperger", "rett",
        "babinski", "romberg", "brudzinski", "kernig", "lhermitte", "tinel", "phalen",
        "meniere", "ménière", "broca", "wallenberg", "todd", "uhthoff", "argyll",
        # gastro / hepatic
        "crohn", "barrett", "whipple", "hirschsprung", "gilbert", "meckel", "wilson",
        "budd", "chiari", "mallory", "weiss", "zollinger", "ellison", "boerhaave",
        "nissen", "billroth", "roux", "heller", "ladd", "murphy", "mcburney", "rovsing",
        "blumberg", "courvoisier", "ranson", "pugh", "banti", "caroli", "ogilvie",
        # onc / heme
        "hodgkin", "burkitt", "ewing", "wilms", "kaposi", "bowen", "paget", "gleason",
        "breslow", "waldenstrom", "waldenström", "sezary", "sézary", "reed",
        "sternberg", "richter", "binet", "durie", "karnofsky", "fuhrman", "auer",
        "howell", "jolly", "heinz", "bence", "krukenberg", "pancoast", "virchow",
        # endo / metabolic
        "addison", "cushing", "graves", "hashimoto", "conn", "sheehan", "nelson",
        "riedel", "quervain", "sipple", "wermer", "carney", "mccune", "albright",
        # rheum / immune
        "sjogren", "sjögren", "behcet", "behçet", "wegener", "takayasu", "buerger",
        "churg", "strauss", "henoch", "schonlein", "schönlein", "felty", "reiter",
        "raynaud", "dupuytren", "peyronie", "baker", "heberden", "bouchard", "gottron",
        # renal / uro
        "bright", "berger", "goodpasture", "alport", "fanconi", "bartter", "liddle",
        "gitelman", "potter", "peyer",
        # cardio / pulm
        "kawasaki", "marfan", "ehlers", "danlos", "osler", "rendu", "kussmaul",
        "cheyne", "stokes", "biot", "hamman", "brugada", "wolff", "wenckebach",
        "mobitz", "eisenmenger", "fallot", "ebstein", "blalock", "taussig", "fontan",
        "glenn", "norwood", "beck", "levine", "duroziez", "quincke", "corrigan",
        # ortho / derm / peds
        "osgood", "schlatter", "legg", "calve", "calvé", "perthes", "scheuermann",
        "colles", "pott", "bennett", "salter", "harris", "morton", "haglund", "sever",
        "kohler", "köhler", "freiberg", "kienbock", "klinefelter", "turner", "noonan",
        "prader", "willi", "angelman", "digeorge", "apert", "crouzon", "treacher",
        "robin", "apgar", "ballard", "dubowitz", "moro", "ortolani", "barlow",
        "galeazzi", "koplik", "nikolsky", "auspitz", "wickham", "darier",
        # eponymous procedures / positions / signs / scores
        "trendelenburg", "valsalva", "crede", "credé", "leopold", "mohs", "halsted",
        "kocher", "pfannenstiel", "lichtenstein", "bassini", "shouldice", "hartmann",
        "mcvay", "seldinger", "swan", "ganz", "foley", "yankauer", "penrose",
        "pratt", "allis", "adson", "homan", "trousseau", "chvostek", "finkelstein",
        "glasgow", "braden", "norton", "morse", "mallampati", "cormack", "lehane",
        "aldrete", "arbor", "framingham", "killip", "forrester",
    }
)

#: Words that, next to a capitalised surname, mean it is being used clinically.
EPONYM_CONTEXT: frozenset[str] = frozenset(
    {
        "disease", "diseases", "syndrome", "syndromes", "sign", "signs", "reflex",
        "reflexes", "palsy", "test", "tests", "maneuver", "manoeuvre",
        "procedure", "operation", "repair", "classification", "criteria", "criterion",
        "score", "scores", "scale", "index", "stage", "staging", "grade", "grading",
        "class", "type", "lymphoma", "sarcoma", "carcinoma", "tumor", "tumour",
        "fracture", "dislocation", "node", "nodes", "nodule", "ligament", "gland",
        "canal", "triangle", "space", "point", "line", "angle", "incision", "position",
        "anastomosis", "plasty", "dementia", "chorea", "ataxia", "atrophy", "dystrophy",
        "granulomatosis", "arteritis", "vasculitis", "thyroiditis", "esophagitis",
        "oesophagitis", "nephritis", "nephropathy", "neuropathy", "myopathy",
        "encephalopathy", "tetralogy", "anomaly", "deformity", "contracture", "cyst",
        "ulcer", "band", "law", "rule", "ratio", "formula", "diverticulum", "pouch",
        "tube", "catheter", "clamp", "forceps", "retractor", "flap", "graft", "shunt",
        "murmur", "breathing", "respiration", "triad", "phenomenon", "reaction",
        "variant", "lesion", "plaque", "body", "bodies", "cell", "cells", "crisis",
        "attack", "episode", "gait", "tremor", "rigidity", "aphasia", "amnesia",
        "fistula", "hernia", "prophylaxis", "index", "criteria",
    }
)

#: Words that, immediately before a capitalised surname, mean it is a diagnosis, not
#: a person. "diagnosed with Parkinson", "history of Crohn".
EPONYM_PRECEDING: frozenset[str] = frozenset(
    {
        "with", "of", "for", "diagnosed", "dx", "denies", "denied", "suspected",
        "possible", "probable", "known", "advanced", "early", "late", "stage", "grade",
        "class", "type", "positive", "negative", "consistent", "suggestive", "rule",
        "r/o", "ro", "treated", "treating", "managed", "worsening", "improving",
        "underlying", "concurrent", "comorbid", "secondary", "primary", "idiopathic",
        "bilateral", "unilateral", "chronic", "acute", "severe", "mild", "moderate",
    }
)

# ---------------------------------------------------------------------------------
# Name cues. The rules-only floor detects names ONLY via an explicit cue -- a title,
# a credential, or a form-field key. It never guesses from a gazetteer, because a
# surname gazetteer is precisely what turns "Parkinson's disease" into a false
# positive. Broad name recall is the neural tagger's job, not the rule layer's.
# ---------------------------------------------------------------------------------
TITLES: tuple[str, ...] = (
    "Dr", "Dr.", "Doctor", "Mr", "Mr.", "Mrs", "Mrs.", "Ms", "Ms.", "Miss", "Mx", "Mx.",
    "Prof", "Prof.", "Professor", "Rev", "Rev.", "Sr", "Sr.", "Fr", "Fr.", "Sgt", "Sgt.",
    "Capt", "Capt.", "Lt", "Lt.", "Col", "Col.", "Maj", "Maj.", "Sir", "Dame", "Nurse",
)

#: Titles that specifically imply a clinician rather than the patient.
PROVIDER_TITLES: frozenset[str] = frozenset({"dr", "dr.", "doctor", "prof", "prof.", "professor", "nurse"})

CREDENTIALS: tuple[str, ...] = (
    "MD", "M.D.", "DO", "D.O.", "MBBS", "MBChB", "RN", "R.N.", "LPN", "LVN", "NP",
    "ARNP", "CRNP", "FNP", "FNP-C", "PA", "PA-C", "PharmD", "RPh", "PhD", "Ph.D.",
    "DDS", "DMD", "DPM", "OD", "DVM", "CRNA", "CNM", "CNS", "LCSW", "MSW",
    "RRT", "DPT", "OTR", "SLP", "MSN", "BSN", "APRN", "FACS",
    "FACP", "FAAP", "FACC", "FACOG", "MPH", "MHA", "ScD",
)

#: The same credentials as a regex alternation, for the trailing-credential name cue
#: ("K. Petrosyan, PT, DPT"). Shared by ``detectors.rules`` and ``detectors.structural``,
#: which each used to carry their own copy and had already drifted apart.
#:
#: Order matters -- first alternative wins, so every multi-letter form must precede the
#: shorter form it starts with, or "DPT" is matched as a bare "D" + leftovers. Prehospital
#: and therapy credentials are included because a produced record's signature lines are full
#: of them: an EMT, a paramedic and a physical therapist are all identifiable providers.
#: Deliberately absent: ``DC`` (Washington DC, and "d/c" for discontinue), ``HR`` (heart
#: rate), ``PA`` without the ``-C`` (already covered, and "PA" is a chest-film view).
CREDENTIAL_ALT: str = (
    r"M\.?D\.?|D\.?O\.?|MBBS|MBChB|R\.?N\.?|L\.?P\.?N\.?|L\.?V\.?N\.?|A\.?R\.?N\.?P\.?|"
    r"C\.?R\.?N\.?P\.?|F\.?N\.?P\.?(?:-C)?|N\.?P\.?|P\.?A\.?-?C?|Pharm\.?D\.?|R\.?Ph\.?|"
    r"Ph\.?D\.?|D\.?D\.?S\.?|D\.?M\.?D\.?|D\.?P\.?M\.?|O\.?D\.?|D\.?V\.?M\.?|"
    r"C\.?R\.?N\.?A\.?|C\.?N\.?M\.?|L\.?C\.?S\.?W\.?|M\.?S\.?W\.?|R\.?R\.?T\.?|"
    r"D\.?P\.?T\.?|M\.?S\.?N\.?|B\.?S\.?N\.?|A\.?P\.?R\.?N\.?|F\.?A\.?C\.?S\.?|"
    r"F\.?A\.?C\.?P\.?|F\.?A\.?A\.?P\.?|F\.?A\.?C\.?C\.?|F\.?A\.?C\.?E\.?P\.?|"
    r"EMT(?:-[BPI])?|NRP|O\.?T\.?R\.?(?:/L)?|P\.?T\.?A\.?|P\.?T\.?"
)

#: Tokens that mark an organisation. Catches the "Mr. Wood -> Wood Memorial" collision
#: from the other direction: the capitalised token before one of these is an ORG, not
#: a person, even when it is spelled exactly like the patient's surname.
ORG_TOKENS: tuple[str, ...] = (
    "Hospital", "Hospitals", "Medical", "Center", "Centre", "Clinic", "Clinics",
    "Health", "Healthcare", "Memorial", "Regional", "General", "Infirmary",
    "Sanatorium", "Institute", "Institutes", "Laboratory", "Laboratories", "Labs",
    "Pharmacy", "Nursing", "Rehabilitation", "Rehab", "Surgery", "Surgical",
    "Practice", "Associates", "Physicians", "Partners",
    "University", "Foundation", "Presbyterian", "Baptist", "Methodist", "Lutheran",
    "Mercy", "Sinai", "Deaconess", "Hospice", "Pavilion", "Oncology", "Cardiology",
    "Orthopedic", "Orthopaedic", "Imaging", "Radiology", "Diagnostics", "Dialysis",
)

#: HIPAA 18 explicitly covers "any other unique identifying characteristic". A rare
#: occupation is exactly that. Kept short and high-signal; the model extends it.
PROFESSION_CUES: tuple[str, ...] = (
    "works as", "worked as", "employed as", "employed by", "occupation is",
    "self-employed as", "job title",
)

# ---------------------------------------------------------------------------------
# Form-field keys. Enormous recall win for the rules-only floor, because header,
# footer, signature-block and form-field PHI is highly structured -- and P2.7
# explicitly requires the test set to contain identifiers hiding in exactly those
# places. Order matters: first match wins, so specific keys precede generic ones.
# ---------------------------------------------------------------------------------
FIELD_KEY_CATEGORY: tuple[tuple[str, str], ...] = (
    (r"(?:social\s*security(?:\s*(?:no|number|#))?|ssn)", "SSN"),
    (r"(?:medical\s*record|mrn|chart|record)\s*(?:no\.?|number|#|id)?", "MRN"),
    (r"(?:health\s*plan|member|subscriber|policy|group|insurance|payer|beneficiary)"
     r"\s*(?:no\.?|number|#|id)?", "HEALTH_PLAN_ID"),
    (r"(?:account|acct|billing|invoice|claim)\s*(?:no\.?|number|#|id)?", "ACCOUNT"),
    (r"(?:npi|dea|license|licence|cert(?:ificate)?)\s*(?:no\.?|number|#|id)?", "LICENCE"),
    (r"(?:serial|s/n|device|implant|lot|udi)\s*(?:no\.?|number|#|id)?", "DEVICE"),
    (r"(?:vin|vehicle|(?:licen[cs]e\s*)?plate)\s*(?:no\.?|number|#)?", "VEHICLE"),
    (r"fax(?:\s*(?:no\.?|number|#))?", "FAX"),
    (r"(?:phone|telephone|tel|mobile|cell|pager|contact)\s*(?:no\.?|number|#)?", "PHONE"),
    (r"e-?mail(?:\s*address)?", "EMAIL"),
    (r"(?:url|website|web\s*site|portal)", "URL"),
    (r"ip\s*(?:address|addr)?", "IP"),
    (r"(?:date\s*of\s*birth|dob|birth\s*date|birthdate)", "DATE"),
    (r"(?:admission|admit|discharge|service|visit|procedure|surgery|collection|"
     r"report|dictation|transcription|encounter|onset|death|expiration)\s*date", "DATE"),
    (r"date\s*(?:of\s*(?:service|admission|discharge|procedure|visit|exam|death))?", "DATE"),
    (r"age", "AGE"),
    (r"(?:attending|referring|ordering|consulting|primary\s*care|pcp|provider|"
     r"physician|surgeon|doctor|clinician|dictated\s*by|signed\s*by|transcribed\s*by|"
     r"authenticated\s*by|reviewed\s*by|prepared\s*by|certified\s*by|co-?signed\s*by|"
     r"reported\s*by|resident|fellow|nurse|rn|therapist|proceduralist|radiologist|"
     r"pathologist|anesthesiologist|electromyographer|technologist)"
     r"(?:\s*(?:physician|provider|clinician|name|md|do))?", "NAME_PROVIDER"),
    # A bare "Patient" key labels the name on most produced forms, so the name suffix has to
    # be optional. "pt" stays mandatory-suffix: bare "PT" is physical therapy.
    (r"(?:patient|client|subject|resident)\s*(?:full\s*)?(?:name)?", "NAME_PATIENT"),
    (r"pt\s*(?:full\s*)?name", "NAME_PATIENT"),
    (r"(?:emergency\s*contact|next\s*of\s*kin|nok|guarantor|guardian|spouse|"
     r"mother|father|parent|relative|informant)\s*(?:name)?", "NAME_OTHER"),
    (r"(?:street|address|addr|residence|home\s*address|mailing\s*address)", "GEO_STREET"),
    (r"(?:city|town|municipality)", "GEO_CITY"),
    (r"(?:zip|zipcode|zip\s*code|postal\s*code|postcode)", "GEO_ZIP"),
    (r"(?:receiving|transferring|admitting|referring|discharge|performing)?\s*"
     r"(?:hospital|facility|clinic|site|institution|practice|pharmacy)(?:\s*name)?", "ORG"),
    (r"(?:occupation|profession|employer|job\s*title|position)", "PROFESSION"),
    (r"name", "NAME_PATIENT"),
)

#: Field keys whose values are clinical, never identifying. Prevents the field-cue
#: detector from masking "Diagnosis: Parkinson's disease" or "Allergies: penicillin",
#: which would be a catastrophic utility loss. Checked BEFORE FIELD_KEY_CATEGORY.
FIELD_KEY_ALLOWLIST: tuple[str, ...] = (
    r"(?:chief\s*)?complaint", r"diagnos[ei]s", r"impression", r"assessment", r"plan",
    r"history(?:\s*of\s*present\s*illness)?", r"hpi", r"ros", r"review\s*of\s*systems",
    r"allerg(?:y|ies)", r"medications?", r"meds", r"dosage", r"dose", r"route",
    r"frequency", r"sig", r"vitals?", r"temp(?:erature)?", r"pulse", r"bp",
    r"blood\s*pressure", r"hr", r"heart\s*rate", r"rr", r"resp(?:iration|iratory)?"
    r"(?:\s*rate)?", r"o2\s*sat", r"spo2", r"weight", r"wt", r"height", r"ht", r"bmi",
    r"pain(?:\s*score)?", r"findings?", r"technique", r"comparison", r"indication",
    r"procedure\s*performed", r"specimen", r"gross\s*description", r"microscopic",
    r"results?", r"reference\s*range", r"units?", r"flag", r"disposition", r"condition",
    r"prognosis", r"follow\s*-?\s*up", r"instructions?", r"labs?", r"imaging",
    r"pathology", r"micro(?:biology)?", r"sex", r"gender", r"race", r"ethnicity",
    r"marital\s*status", r"language", r"code\s*status", r"smoking\s*status", r"alcohol",
    r"substance", r"past\s*medical\s*history", r"pmh", r"family\s*history",
    r"social\s*history", r"surgical\s*history", r"immunizations?", r"problem\s*list",
)
