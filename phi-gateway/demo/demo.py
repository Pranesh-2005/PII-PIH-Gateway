"""One-command demo: raw note -> masked -> foundation LLM -> rehydrated.

Designed not to crash. If no provider key is set, or every provider is down, the LLM leg is
simulated by echoing the masked text back through a template -- the round trip and the
rehydration guard are still demonstrated, offline. Degradation is acceptable at a review;
a traceback is not.

    python demo/demo.py                                  # bundled note
    python demo/demo.py --file path/to/unseen_note.txt    # the acceptance test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from phi_gateway import ingest                                   # noqa: E402
from phi_gateway.config import load_policy                      # noqa: E402
from phi_gateway.llm.client import LLMClient, LLMUnavailable     # noqa: E402
from phi_gateway.pipeline import deidentify_full, rehydrate      # noqa: E402
from phi_gateway.vault.mapping_store import MappingStore         # noqa: E402

DEFAULT_NOTE = Path(__file__).resolve().parent / "sample_notes" / "note_01.txt"

PROMPT = (
    "You are a clinical assistant. Using only the note below, produce:\n"
    "1. A five-bullet summary.\n"
    "2. The number of days between admission and the start of new medication.\n"
    "3. The attending physician and the consulting specialist.\n"
    "Reproduce every [PLACEHOLDER] token exactly as written; do not invent any.\n"
)


def _banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _simulated_llm(masked: str) -> str:
    """Stand-in that behaves like a real model: it copies placeholders back, and -- on
    purpose -- invents one that was never in the input, so the rehydration guard is
    visible in the offline path too."""
    head = "\n".join(line for line in masked.splitlines() if line.strip())[:600]
    return (
        "SUMMARY (simulated -- no provider key set):\n"
        f"{head}\n\n"
        "Note: the patient identified as [NAME_PATIENT_1] was seen on [DATE_1].\n"
        "Fabricated reference for the guard test: [NAME_PATIENT_99]\n"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", default=str(DEFAULT_NOTE))
    ap.add_argument("--policy", default=None)
    ap.add_argument("--offline", action="store_true", help="skip the provider call entirely")
    args = ap.parse_args(argv)

    text = ingest.read(args.file)
    policy = load_policy(args.policy)

    _banner("STAGE 0 -- RAW NOTE (contains PHI)")
    print(text)

    result = deidentify_full(text, policy=policy)

    _banner("STAGE A-F -- MASKED NOTE (this is what leaves the building)")
    print(result.masked_text)

    _banner("MAPPING (stays in the vault, encrypted -- shown here for the demo only)")
    for placeholder, entry in result.mapping.entries.items():
        print(f"  {placeholder:<26} {entry.category.value:<15} {', '.join(entry.surfaces[:4])}")

    # The mapping is round-tripped through the encrypted vault, as it would be in service.
    vault = MappingStore(ttl_seconds=policy.vault.ttl_seconds, key_env=policy.vault.key_env)
    session_id = vault.put(result.mapping)
    print(f"\n  vault session: {session_id}  (AES-GCM, TTL {policy.vault.ttl_seconds}s)")

    _banner("STAGE G -- LEAK SELF-CHECK (we are our own adversary)")
    report = result.leak_report
    if report is None or report.clean:
        print("  clean: no identifier detected in our own masked output")
    else:
        for f in report.findings:
            print(f"  ! LEAK {f.category.value:<15} {f.text!r}  [{f.detector}]")
        if result.blocked:
            print("\n  BLOCKED -- refusing to forward to the foundation LLM.")
            print("  This is the intended behaviour: a blocked request is not a breach.")
            return 1

    if result.review_queue:
        _banner("HUMAN REVIEW QUEUE (masked anyway -- recall asymmetry)")
        for i in result.review_queue:
            print(f"  ? {i.placeholder:<24} {i.span.text!r} -- {i.reason}")

    provider_name = "simulated"
    if args.offline:
        llm_text = _simulated_llm(result.masked_text)
    else:
        try:
            client = LLMClient(policy.llm)
            out = client.complete(f"{PROMPT}\n---\n{result.masked_text}\n---")
            llm_text, provider_name = out.text, f"{out.provider}/{out.model}"
        except LLMUnavailable as exc:
            print(f"\n[llm unavailable, simulating] {exc}")
            llm_text = _simulated_llm(result.masked_text)

    _banner(f"FOUNDATION LLM RESPONSE ({provider_name}) -- still de-identified")
    print(llm_text)

    _banner("REHYDRATED RESPONSE (mapping from the vault)")
    print(rehydrate(llm_text, vault.get(session_id)))

    _banner("STATS")
    for k, v in result.summary().items():
        print(f"  {k}: {v}")

    vault.delete(session_id)
    print("\n  vault session destroyed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
