"""FastAPI surface for the gateway.

The mapping is **never** returned over the wire. Callers get a ``session_id``; the mapping
stays encrypted in the vault and is only touched inside ``/rehydrate``. That way the
re-identification key never crosses a network boundary or lands in an access log.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import ingest
from ..config import load_policy
from ..llm.client import LLMClient, LLMUnavailable
from ..pipeline import deidentify_full, rehydrate
from ..types import ReviewItem
from ..vault.mapping_store import MappingExpired, MappingStore

app = FastAPI(title="PHI/PII De-identification Gateway", version="0.1.0")

_policy = load_policy()
_vault = MappingStore(ttl_seconds=_policy.vault.ttl_seconds, key_env=_policy.vault.key_env)
_llm = LLMClient(_policy.llm)

#: ponytail: review items live in a plain dict beside the vault. They contain no PHI beyond
#: what the masked text already shows, so they do not need encrypting. Bound the size when
#: this stops being a demo.
_reviews: dict[str, list[dict]] = {}


class DeidRequest(BaseModel):
    text: str
    patient_key: str | None = None


class RehydrateRequest(BaseModel):
    session_id: str
    response: str


class RoundTripRequest(BaseModel):
    text: str
    prompt: str = Field(
        default="Summarise this clinical note in five bullet points. "
        "Keep every placeholder token exactly as written.",
    )
    patient_key: str | None = None


def _review_payload(items: list[ReviewItem]) -> list[dict]:
    return [
        {
            "placeholder": i.placeholder,
            "category": i.span.category.value,
            "text": i.span.text,
            "start": i.span.start,
            "end": i.span.end,
            "score": i.span.score,
            "reason": i.reason,
        }
        for i in items
    ]


#: ponytail: one file, so FileResponse rather than a StaticFiles mount. Served from the same
#: process that does the masking -- the page pulls no font, framework or script from a CDN,
#: because "the note never leaves this machine" has to be true of the UI too.
_INDEX = Path(__file__).with_name("static") / "index.html"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_INDEX, media_type="text/html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "date_mode": _policy.dates.mode,
        "neural_enabled": _policy.neural.enabled,
        "selfcheck": _policy.selfcheck.on_leak if _policy.selfcheck.enabled else "off",
        "llm_providers": [p.name for p in _llm.available_providers()],
        "sessions": len(_vault),
    }


@app.post("/deidentify")
def deidentify_endpoint(req: DeidRequest) -> dict:
    result = deidentify_full(req.text, patient_key=req.patient_key, policy=_policy)
    session_id = _vault.put(result.mapping)
    _reviews[session_id] = _review_payload(result.review_queue)
    return {
        "session_id": session_id,
        "masked_text": result.masked_text,
        "blocked": result.blocked,
        "summary": result.summary(),
        "leak_findings": [
            {"category": f.category.value, "text": f.text, "detector": f.detector}
            for f in (result.leak_report.findings if result.leak_report else [])
        ],
    }


@app.post("/deidentify/file")
async def deidentify_file(file: UploadFile = File(...),
                          patient_key: str | None = Form(default=None)) -> dict:
    """Same contract as ``/deidentify``, but the body is a PDF, DOCX or text file.

    Extraction goes through ``phi_gateway.ingest``, the same function the CLI and the demo use,
    so a PDF cannot be masked to a different standard than a pasted string. The upload is written
    to a temp file and deleted in a ``finally``: pdfplumber and python-docx both want a real
    path, and a PHI-bearing document left in the OS temp directory is the kind of leak this
    project exists to prevent.
    """
    suffix = Path(file.filename or "upload.txt").suffix or ".txt"
    tmp = Path(tempfile.mkdtemp()) / f"upload{suffix}"
    try:
        tmp.write_bytes(await file.read())
        try:
            text = ingest.read(tmp)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        return deidentify_endpoint(DeidRequest(text=text, patient_key=patient_key))
    finally:
        tmp.unlink(missing_ok=True)
        tmp.parent.rmdir()


@app.post("/rehydrate")
def rehydrate_endpoint(req: RehydrateRequest) -> dict:
    try:
        mapping = _vault.get(req.session_id)
    except MappingExpired as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    return {"text": rehydrate(req.response, mapping)}


@app.post("/roundtrip")
def roundtrip(req: RoundTripRequest) -> dict:
    result = deidentify_full(req.text, patient_key=req.patient_key, policy=_policy)
    if result.blocked:
        # The self-check found an identifier in our own output. Refuse to forward.
        raise HTTPException(
            status_code=409,
            detail={
                "error": "self-check leak; refusing to forward to the LLM",
                "findings": [
                    {"category": f.category.value, "text": f.text, "detector": f.detector}
                    for f in (result.leak_report.findings if result.leak_report else [])
                ],
            },
        )

    session_id = _vault.put(result.mapping)
    _reviews[session_id] = _review_payload(result.review_queue)
    try:
        llm = _llm.complete(f"{req.prompt}\n\n---\n{result.masked_text}\n---")
    except LLMUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "session_id": session_id,
        "masked_text": result.masked_text,
        "llm_provider": llm.provider,
        "llm_model": llm.model,
        "llm_response_masked": llm.text,
        "llm_response_rehydrated": rehydrate(llm.text, result.mapping),
        "summary": result.summary(),
    }


@app.get("/review-queue/{session_id}")
def review_queue(session_id: str) -> dict:
    if session_id not in _reviews:
        raise HTTPException(status_code=404, detail="unknown session")
    return {"session_id": session_id, "items": _reviews[session_id]}


@app.delete("/session/{session_id}")
def end_session(session_id: str) -> dict:
    """Explicit teardown. The TTL is the backstop, not the plan."""
    _vault.delete(session_id)
    _reviews.pop(session_id, None)
    return {"deleted": session_id}
