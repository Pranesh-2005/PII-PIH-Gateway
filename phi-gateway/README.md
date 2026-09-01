# PHI/PII De-identification Gateway

A gateway that sits between clinical documents and a foundation LLM. It strips every patient
identifier on the way in, sends the masked text to the model, and rehydrates the answer on the
way out — so the LLM never sees PHI and the caller still gets a readable reply.

The hard part is not entity tagging. It is the tension the assessment brief names directly:

> Redact too little and you leak patient data. Redact too much and the downstream LLM becomes
> useless. Both are failures.

Every design decision below is an answer to one side of that trade-off, and every claim in this
README has a number behind it in [`eval/REPORT.md`](eval/REPORT.md).

---

## What it does, in one command

```bash
pip install -e ".[ml]"
python demo/demo.py --file demo/sample_notes/note_02_adversarial.txt
```

That prints all five stages: raw note → masked text + mapping → LLM call → rehydrated response →
self-check verdict. It runs with **no API key** (the provider call degrades to a simulated
response and says so) and with **no GPU**.

Input can be `.pdf`, `.docx`, `.txt`, `.md`, or stdin — see [Document ingest](#document-ingest).

---

## Architecture

```
 PDF / DOCX / TXT  ──►  ingest.read()          layout-preserving text, headers + footers + tables
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│  A. structural pre-pass      headers, footers, signature blocks, tables, form fields  │
│                              → PHI density is high here, so thresholds tighten        │
│  B. rule detectors           SSN, phone, fax, email, URL, IP, MRN, account, NPI/DEA,  │
│                              VIN, device serial, ZIP — validated + context-cued       │
│  C. neural tagger            ModernBERT-base BIO, 149,634,855 params. CPU. Overlaps   │
│                              B on purpose — two independent chances at every span     │
│  D. merge + arbitration      recall-first union. Rules win structured IDs, the model   │
│                              wins names/geo. Low confidence → human-review queue      │
│  E. entity resolution        "Mr. Wood" / "Wood" / "Wood's" / "J. Wood" → one cluster │
│                              → one placeholder. Per-document only                     │
│  F. policy application       typed consistent pseudonyms; per-patient date shift;     │
│                              ages ≥ 90 → [AGE_OVER_89]                                │
│  G. LEAK SELF-CHECK          re-run B and a paranoid low-threshold C over our OWN     │
│                              masked output. Any hit = block, escalate, do not forward │
└───────────────────────────────────────────────────────────────────────────────────────┘
        │
        ├──► masked_text  ──►  foundation LLM (Groq → NVIDIA NIM failover)
        │                          │
        └──► mapping (AES-GCM, session-keyed, TTL, never on disk in plaintext, never
             returned over the wire)  ──►  rehydrate(response, mapping)
                                              │
                                              └──► only placeholders present in THIS
                                                   session's mapping are substituted.
                                                   Anything else → [UNKNOWN_PLACEHOLDER]
                                                   + a logged security event
```

**Stage G and the unknown-placeholder guard are the two pieces that answer "what if the LLM
echoes a token that was never in the input?"** The demo proves the second one live: it injects a
fabricated `[NAME_PATIENT_99]` into the response and you watch it come back as
`[UNKNOWN_PLACEHOLDER]` instead of somebody's name.

### Why typed consistent pseudonyms

`[NAME_PATIENT_1]`, not a fake name. Three reasons, in order of weight:

1. **A reviewer can eyeball the masked output for leaks.** Surrogate generation makes that
   impossible — a missed real name hides among the fakes, indistinguishable.
2. Co-reference survives, so the LLM can still reason about who did what.
3. Rehydration is deterministic and auditable.

The utility cost versus surrogates is measured, not assumed.

### Dates: shifted, intervals exact

Per-patient offset δ = `HMAC(secret_salt, patient_key)`, applied to every absolute date, so
**intervals are preserved exactly**. "Admitted 03/14, procedure 03/17" still reads as three days;
"3 days post-op" is left alone. Verified by arithmetic, not by an LLM: 152/152 intervals exact.

### Ages over 89

Anything ≥ 90 → `[AGE_OVER_89]`. Also catches `nonagenarian`/`centenarian`, and a **birth year
that implies an age over 89** — "born in 1929" is an age, and a system that only looks for
"N year old" leaks it.

---

## Document ingest

`src/phi_gateway/ingest.py` — one function, `read(path) -> str`, used by the CLI, the demo and
the API so a PDF can never be masked to a different standard than a pasted string.

| Format | Reader | What is deliberately preserved |
|---|---|---|
| `.pdf` | pdfplumber (`layout=True`), pypdf fallback | Column whitespace and line structure — stage A classifies form fields and table rows by exactly that. Pages joined with `\f`. |
| `.docx` | python-docx | **Headers and footers** (`section.header`, which `document.paragraphs` silently skips) and **table cells**, tab-joined. A running header "Fitzsimmons, Clementine — Page 1 of 3" is PHI on every page. |
| `.txt` `.md` `.rtf` | stdlib | As-is. |
| unknown suffix | stdlib, if it decodes as text | A note emailed as `.dat` still reads; a binary blob raises. |
| `.doc`, scanned PDF | — | **Raises.** No OCR, and an image-only PDF returns nothing, so it refuses rather than reporting an empty note as clean. |

```bash
python -m phi_gateway.ingest          # self-check: header/footer/table/binary cases
```

---

## How to run

### Install

```bash
cd phi-gateway
pip install -e ".[ml]"       # ml extra = the neural tagger; core alone still round-trips
```

The gateway runs **CPU-local**: no network, no GPU, full data residency — which is the whole
premise of the project. GPU is used only for training.

Optional, for the live LLM call — either one is enough, and neither is required:

```bash
export GROQ_API_KEY=...       # primary
export NVIDIA_API_KEY=...     # failover
```

Model IDs are discovered at runtime via `GET /models` rather than hardcoded, because both
providers rotate their catalogues and a dead model ID during a live demo is an avoidable crash.

### CLI

```bash
# mask one document, eyeball the mapping and the self-check
python -m phi_gateway.cli deidentify --file note.pdf
python -m phi_gateway.cli deidentify --file record.docx --json
echo "Mr. Wood was seen at Wood Memorial." | python -m phi_gateway.cli deidentify

# full round trip through the foundation LLM
python -m phi_gateway.cli roundtrip --file note.txt

# what the providers actually serve right now
python -m phi_gateway.cli models
```

### Web UI

```bash
python -m phi_gateway.cli serve      # then open http://127.0.0.1:8000/
```

`src/phi_gateway/service/static/index.html` — one file, plain HTML/CSS/JS, **no build step and no
CDN**. That last part is deliberate: the premise of this project is that a note never leaves the
machine, and a page fetching a font or a framework from a third party on every load contradicts
it. Served by the same uvicorn process that does the masking.

Panels: paste a note or upload a PDF/DOCX → side-by-side raw vs masked with every placeholder
highlighted → self-check verdict banner → per-category counts → the human-review queue → rehydrate
box → **Destroy session**. Two sample buttons, and the adversarial one is the honest one: it still
leaks `Bell` the charge nurse and a lowercase city, visibly, on the page.

The upload path shows no raw pane, because `POST /deidentify/file` does not return the extracted
text — the browser never holds the PHI-bearing version of an uploaded document. Note text is
HTML-escaped before it is written into the DOM; an uploaded document is untrusted input.

### API

```bash
python -m phi_gateway.cli serve            # http://127.0.0.1:8000/docs
```

| Endpoint | Purpose |
|---|---|
| `POST /deidentify` | `{text}` → `{session_id, masked_text, blocked, leak_findings}` |
| `POST /deidentify/file` | multipart PDF/DOCX/TXT upload, same response |
| `POST /rehydrate` | `{session_id, response}` → `{text}` |
| `POST /roundtrip` | mask → LLM → rehydrate in one call; **409 if stage G finds a leak** |
| `GET /review-queue/{session_id}` | low-confidence spans a human should look at |
| `GET /health` | policy, providers, live session count |

**The mapping is never returned over the wire.** Callers get a `session_id`; the mapping stays
encrypted in the vault and is only touched inside `/rehydrate`, so the re-identification key
never crosses a network boundary or lands in an access log.

### Tests and evaluation

```bash
pytest                                    # 59 tests, incl. the < 1B parameter assertion

python eval/harness.py --coarse           # 450 held-out synthetic notes
python eval/harness.py --coarse --data data/test_set/handauthored.jsonl
python eval/report.py                     # → eval/REPORT.md
python data/test_set/handauthored.py      # rebuild the hand-written gold
```

### Training (Modal)

```bash
modal run training/modal_app.py --step all       # build data → train → checkpoint
modal run training/modal_app.py --step ablate    # 4 encoders × 2 seeds, in parallel
```

Serving stays CPU-local; Modal is only for the A10G training runs.

---

## Results

Full tables, per-category breakdowns and every ablation run: [`eval/REPORT.md`](eval/REPORT.md).

**450 held-out synthetic notes:**

| system | recall | precision | F1 | leak rate | uncovered | p50 |
|---|---|---|---|---|---|---|
| gateway | **0.9807** | 0.9148 | 0.9466 | 0.4289 | **0.0000** | 1008 ms |
| regex only | 0.8334 | 0.9353 | 0.8814 | 0.9889 | 0.9844 | 14 ms |

**14 hand-authored adversarial notes** — no shared code with training:

| system | recall | precision | leak rate | uncovered |
|---|---|---|---|---|
| gateway | **0.9178** | 0.8072 | 0.2143 | 0.1429 |
| regex only | 0.6849 | 0.9091 | 0.7143 | 0.7143 |

Two leak metrics, because they answer different questions. **`leak rate`** counts a note as
leaking if any gold span went unmatched — *including one masked under a different category*.
**`uncovered`** counts only notes where a gold identifier has no overlapping prediction at all,
i.e. plaintext PHI actually reaching the LLM. The gap between them is taxonomy disagreement, not
breach. On the synthetic split `uncovered` is **0.0000**: every one of ~10k gold identifiers is
covered by a mask.

The hand-authored split does **not** hit the leak-rate-0 gate. What survives is diagnosed by name
in `eval/REPORT.md` — a lowercase unpunctuated address, a `LAST, FIRST` value in an ALLCAPS
column, and `Bell` the charge nurse in a note that also says Bell's palsy. Reporting the real
number, because a well-diagnosed honest result is worth more than an unverifiable good one.

**Ablation** (dev, A10G, all 8 runs in `eval/REPORT.md`): DeBERTa-v3-base wins on dev recall, but
dev is saturated (R ≈ 0.9999 for six of eight runs) so that ranking is noise, not signal — which
is why the shipped checkpoint is ModernBERT-base and the honest comparison is the held-out test
split, not dev.

---

## HIPAA 18 categories

Sixteen covered, two excluded on modality grounds, one partial — stated plainly rather than
claimed complete.

| # | Category | Approach |
|---|---|---|
| 1 | Names | Model. Split `NAME_PATIENT` / `NAME_PROVIDER` / `NAME_OTHER` |
| 2 | Geo < state | Model (street/city) + rules (ZIP → 3-digit, HIPAA restricted prefixes dropped) |
| 3 | Dates, DOB, ages > 89 | `policy/dates.py` + `policy/ages.py`, interval-preserving |
| 4–6 | Phone / fax / email | Rules, fax context-cued |
| 7 | SSN | Rules + validity checks (rejects 000/666/9xx area, 00 group, 0000 serial) |
| 8–10 | MRN / health plan / account | Rules, context-cued, + model `ID_GENERIC` |
| 11 | Certificate / licence | Rules + **NPI Luhn** and **DEA check-digit** validation |
| 12 | Vehicle / plate | Rules + **VIN check digit** |
| 13 | Device serial | Rules, context-cued |
| 14–15 | URL / IP | Rules + `ipaddress` stdlib validation |
| 16 | Biometric identifiers | **Excluded** — binary artefacts, not free-text tokens. Textual *references* are caught as ordinary IDs |
| 17 | Full-face photographs | **Excluded** — out of modality for a text gateway |
| 18 | Any other unique ID | **Partial, and said so.** High-entropy catch-all + model `ID_GENERIC`. Unbounded by definition; mitigated by stage G and re-identification risk scoring, not by claiming completeness |

---

## Data and leakage control

Training data is **synthetic or permissively licensed only**. MTSamples (already de-identified
real clinical prose) as the carrier, Faker-injected identifiers in realistic slots, plus an
adversarial generator for the hard cases: ~20 medical eponyms in *both* roles (Parkinson,
Hodgkin, Crohn, Bell, Graves, Addison …), surname==organisation collisions, providers with no
`Dr.` cue, ALLCAPS, lowercase, missing punctuation, and identifiers hidden in tables and
signature blocks. i2b2/n2c2 needs a DUA and was never on the critical path.

Scoring generator-derived test data against a model trained on the same generator inflates
results. So the hand-authored split shares no code with training, its gold spans are resolved
from literal surfaces at build time (an offset in a hand-edited literal rots the next time a word
is added, and a silently mis-offset gold span scores a correct system as leaking), and the two
splits are **reported separately and never averaged**.

---

## Repo layout

```
src/phi_gateway/
  ingest.py                    PDF / DOCX / TXT → text, layout preserved
  detectors/  rules.py  structural.py  fields.py  neural.py  merge.py
  resolve/cluster.py           surface-form coreference
  policy/     placeholders.py  dates.py  ages.py
  selfcheck/leak_scan.py       stage G
  vault/mapping_store.py       AES-GCM, session-keyed, TTL
  llm/client.py                Groq → NVIDIA NIM failover, runtime model discovery
  service/api.py  cli.py  pipeline.py
training/     modal_app.py, data generators
eval/         harness.py, baselines/, report.py, REPORT.md
data/test_set/handauthored.py  hand-written adversarial gold
demo/demo.py
tests/                         59 tests
```

---

## Known limits

* Lowercase, unpunctuated geography is the weakest surface — a training-distribution gap, fixable
  by casing augmentation rather than another word list.
* `ORG` recall 0.886 on the synthetic split; employer-style names (`Marlow Cartage LLP`) are
  detected but frequently labelled as something else, which costs leak rate and not coverage.
* Presidio baseline is wired but not installed here (`pip install ".[eval]"`); regex-only is the
  baseline actually reported.
* No OCR. A scanned PDF raises instead of returning empty text.
* The hand-authored split is 14 dense notes, not the 50 originally planned. Reported as 14.
