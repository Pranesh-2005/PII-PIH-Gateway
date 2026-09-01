"""Command-line surface. ``deidentify`` for eyeballing a single note, ``roundtrip`` for the
full raw -> masked -> LLM -> rehydrated chain, ``models`` for checking what the providers
actually serve before a live demo, ``serve`` for the API.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import ingest
from .config import load_policy
from .llm.client import LLMClient, LLMUnavailable
from .pipeline import deidentify_full, rehydrate

_DEFAULT_PROMPT = (
    "Summarise this clinical note in five bullet points, then state the interval in days "
    "between the first and last dated event. Keep every placeholder token exactly as written."
)


def _read_input(args) -> str:
    if args.file:
        return ingest.read(args.file)
    if args.text:
        return args.text
    return sys.stdin.read()


def _print_mapping(result) -> None:
    print("\n=== MAPPING ===")
    width = max((len(p) for p in result.mapping.entries), default=12)
    for placeholder, entry in result.mapping.entries.items():
        surfaces = ", ".join(entry.surfaces[:4])
        extra = f" (+{len(entry.surfaces) - 4} more)" if len(entry.surfaces) > 4 else ""
        print(f"  {placeholder:<{width}}  {entry.category.value:<15} {surfaces}{extra}")


def _print_leaks(result) -> None:
    report = result.leak_report
    if report is None:
        return
    if report.clean:
        print("\n=== SELF-CHECK === clean")
        return
    print(f"\n=== SELF-CHECK === {len(report.findings)} LEAK(S) FOUND")
    for f in report.findings:
        print(f"  ! {f.category.value:<15} {f.text!r}  [{f.detector}]")
    if result.blocked:
        print("  BLOCKED: not forwarding to the foundation LLM.")


def cmd_deidentify(args) -> int:
    text = _read_input(args)
    policy = load_policy(args.policy)
    result = deidentify_full(text, patient_key=args.patient_key, policy=policy)

    if args.json:
        print(
            json.dumps(
                {
                    "masked_text": result.masked_text,
                    "mapping": json.loads(result.mapping.to_json()),
                    "summary": result.summary(),
                    "review_queue": [
                        {"placeholder": i.placeholder, "text": i.span.text, "reason": i.reason}
                        for i in result.review_queue
                    ],
                },
                indent=2,
            )
        )
        return 1 if result.blocked else 0

    print("=== MASKED ===")
    print(result.masked_text)
    _print_mapping(result)
    if result.review_queue:
        print("\n=== REVIEW QUEUE ===")
        for i in result.review_queue:
            print(f"  ? {i.placeholder:<20} {i.span.text!r} -- {i.reason}")
    _print_leaks(result)
    print("\n=== SUMMARY ===")
    print(json.dumps(result.summary(), indent=2))
    return 1 if result.blocked else 0


def cmd_roundtrip(args) -> int:
    text = _read_input(args)
    policy = load_policy(args.policy)
    result = deidentify_full(text, patient_key=args.patient_key, policy=policy)

    print("=== MASKED (sent to the LLM) ===")
    print(result.masked_text)
    _print_leaks(result)
    if result.blocked:
        return 1

    client = LLMClient(policy.llm)
    try:
        llm = client.complete(f"{args.prompt}\n\n---\n{result.masked_text}\n---")
    except LLMUnavailable as exc:
        print(f"\n=== LLM UNAVAILABLE ===\n{exc}", file=sys.stderr)
        print("Masking succeeded; only the LLM leg failed.", file=sys.stderr)
        return 2

    print(f"\n=== LLM RESPONSE ({llm.provider}/{llm.model}, still masked) ===")
    print(llm.text)
    print("\n=== REHYDRATED ===")
    print(rehydrate(llm.text, result.mapping))
    return 0


def cmd_models(args) -> int:
    policy = load_policy(args.policy)
    client = LLMClient(policy.llm)
    for provider in policy.llm.providers:
        status = "key set" if provider.available else f"NO KEY ({provider.api_key_env})"
        print(f"\n{provider.name}  [{status}]  {provider.base_url}")
        if not provider.available:
            continue
        served = client.list_models(provider)
        print(f"  serves {len(served)} model(s); resolved -> {client.resolve_model(provider)}")
        for m in served[: args.limit]:
            print(f"    {m}")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    print(f"  web UI   http://{args.host}:{args.port}/")
    print(f"  API docs http://{args.host}:{args.port}/docs")
    uvicorn.run("phi_gateway.service.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phi-gateway", description=__doc__)
    parser.add_argument("--policy", default=None, help="path to policy.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    def _io(p):
        p.add_argument("--file", help="path to a clinical note")
        p.add_argument("--text", help="inline text")
        p.add_argument("--patient-key", default=None, help="share a date offset across notes")

    p = sub.add_parser("deidentify", help="mask a note and show the mapping")
    _io(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_deidentify)

    p = sub.add_parser("roundtrip", help="mask -> LLM -> rehydrate")
    _io(p)
    p.add_argument("--prompt", default=_DEFAULT_PROMPT)
    p.set_defaults(func=cmd_roundtrip)

    p = sub.add_parser("models", help="what the configured providers actually serve")
    p.add_argument("--limit", type=int, default=15)
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("serve", help="run the FastAPI service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
