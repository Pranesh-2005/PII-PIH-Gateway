"""Stage C: the BIO tagger.

The half of the problem rules cannot do. A regex can validate an SSN checksum; it cannot
decide whether "Dr. Parkinson diagnosed Parkinson's" contains one identifier or none. That
decision is contextual, which is the whole reason a model is being trained.

Design constraints that come from the demo rather than from accuracy:

* **Import is lazy and failure is survivable.** ``pipeline._neural_spans`` catches
  ``ImportError`` and any runtime exception and continues rules-only. A missing checkpoint or
  an absent torch degrades the gateway; it never crashes it.
* **CPU by default.** No GPU at the interview, no network call, and full data residency --
  which is the premise of the whole project. The cost is latency: **~1.9s/note measured**, not
  the ~50ms this file used to claim. ``_load`` records what was tried and what the fix is.
* **Windowed, never truncated.** A 22-page record is 20k+ tokens. Truncating at 512 would
  silently drop every identifier past page one, and a silent drop is a leak.

Scores are the model's own probabilities, so ``configs/policy.yaml`` thresholds and the
per-class recall-floor tuning in Phase 4 apply to them unchanged.
"""

from __future__ import annotations

import functools
import logging

from ..config import Policy
from ..types import Category, Segment, Source, Span
from .structural import in_structured_segment

log = logging.getLogger("phi_gateway")

#: Below this the span is not even offered to Stage D. Deliberately far under every policy
#: threshold: arbitration and the per-class thresholds decide what survives, not this. It
#: exists only to keep the candidate list from filling with noise at 0.01.
_FLOOR = 0.05


@functools.lru_cache(maxsize=2)
def _load(model_path: str, device: str):
    """Load once per process. Cached because a per-request load is minutes, not milliseconds.

    Measured on this checkpoint (ModernBERT-base, 8 CPU threads, ~2.3k-char note): first
    forward pass ~7s of graph warmup, then **~1.9s/note steady state**. That is 40x the
    ~50ms/note this docstring used to claim, and the claim was wrong -- so it is now a number
    from ``ms_detect`` rather than an estimate.

    ``torch.quantization.quantize_dynamic`` on the Linear layers was tried and rejected: 25%
    faster (1.5s), but it fragmented spans and flipped categories -- one gold ``FAX`` span
    decoded as ``PHONE``/``FAX``/``PHONE`` pieces. Trading category accuracy on a recall-first
    tagger for 25% is the wrong direction. The real fix is ONNX Runtime or OpenVINO export,
    which quantises the graph rather than swapping modules at runtime; not on the critical
    path while 1.9s/note is survivable.
    """
    import os                                        # noqa: PLC0415
    import torch                                     # noqa: PLC0415  (optional dependency)
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    # Torch defaults to a conservative thread count under some launchers; a token classifier
    # is one long matmul chain. Worth the line, though this model turns out to be bandwidth-
    # bound rather than compute-bound, so the gain is small.
    torch.set_num_threads(max(1, os.cpu_count() or 4))
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval().to(device)
    return tok, model


def _decode(
    text: str,
    offsets: list[tuple[int, int]],
    tags: list[str],
    probs: list[float],
) -> list[tuple[int, int, str, float]]:
    """Strict BIO decode over one window, back to character offsets.

    An ``I-`` tag with no open entity opens one anyway. Recall-first: a tagger that emits
    ``I-NAME`` without the ``B-`` has still told us there is a name there, and dropping it
    to punish its formatting would be trading a leak for tidiness.
    """
    out: list[tuple[int, int, str, float]] = []
    start = end = -1
    label = ""
    score: list[float] = []

    def flush() -> None:
        if start >= 0 and label:
            out.append((start, end, label, sum(score) / len(score)))

    for (s, e), tag, p in zip(offsets, tags, probs):
        if s == e:                                   # special token
            continue
        if tag == "O":
            flush()
            start, label, score = -1, "", []
            continue
        prefix, _, cat = tag.partition("-")
        if prefix == "B" or cat != label:
            flush()
            start, end, label, score = s, e, cat, [p]
        else:
            end, score = e, score + [p]
    flush()
    return [(s, e, c, p) for s, e, c, p in out if text[s:e].strip()]


def detect(text: str, segments: list[Segment] | None = None, policy: Policy | None = None) -> list[Span]:
    import torch                                    # noqa: PLC0415

    from ..config import load_policy

    pol = policy or load_policy()
    tok, model = _load(pol.neural.model_path, pol.neural.device)
    id2label = model.config.id2label

    enc = tok(
        text,
        truncation=True,
        max_length=pol.neural.max_length,
        stride=pol.neural.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt",
        padding=True,
    )
    offsets = enc.pop("offset_mapping")
    enc.pop("overflow_to_sample_mapping", None)

    with torch.no_grad():
        logits = model(**{k: v.to(pol.neural.device) for k, v in enc.items()}).logits
    probs = torch.softmax(logits, dim=-1)
    conf, ids = probs.max(dim=-1)

    segs = segments or []
    # Windows overlap by ``stride``, so the same entity is decoded twice. Keyed by
    # (start, end, category) and kept at the higher confidence -- the copy nearer the middle
    # of a window generally scores better, and taking the max is the recall-first choice.
    best: dict[tuple[int, int, str], float] = {}
    for w in range(ids.size(0)):
        window_offsets = [(int(a), int(b)) for a, b in offsets[w].tolist()]
        tags = [id2label[int(i)] for i in ids[w].tolist()]
        for s, e, cat, p in _decode(text, window_offsets, tags, conf[w].tolist()):
            key = (s, e, cat)
            if p > best.get(key, 0.0):
                best[key] = p

    out: list[Span] = []
    for (s, e, cat), p in best.items():
        if p < _FLOOR:
            continue
        try:
            category = Category(cat)
        except ValueError:
            log.warning("tagger emitted unknown label %r; ignoring", cat)
            continue
        out.append(
            Span(
                start=s,
                end=e,
                category=category,
                text=text[s:e],
                score=float(p),
                source=Source.NEURAL,
                detector="tagger",
                in_structured_segment=in_structured_segment(s, segs),
            )
        )
    out.sort(key=lambda sp: (sp.start, sp.end))
    return out
