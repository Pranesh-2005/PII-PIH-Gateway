"""Turn a PDF record production into one .txt per page.

Not part of the gateway -- the gateway's contract is text in, text out. This exists only so
a PDF handed over at review time can be fed in without retyping it.

``extraction_mode="layout"`` is the point of this script rather than a one-liner: it keeps
the horizontal whitespace, so a two-column table row stays "Key<spaces>Value" instead of
collapsing to "Key Value". The structural pre-pass segments cells on runs of two-plus
spaces, so without layout mode every table row parses as prose and the form-field cue
detector -- the largest recall win in the whole pipeline -- goes blind on exactly the
header/table PHI it exists to catch.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pypdf import PdfReader


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: pdf_to_text.py <in.pdf> [outdir]", file=sys.stderr)
        return 2
    src = Path(argv[1])
    outdir = Path(argv[2]) if len(argv) > 2 else src.parent / (src.stem + "_pages")
    outdir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(src))
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text(extraction_mode="layout") or ""
        dest = outdir / f"page_{i:02d}.txt"
        dest.write_text(text, encoding="utf-8")
        print(f"{dest}  {len(text)} chars")

    joined = outdir / "all_pages.txt"
    joined.write_text(
        "\n\n".join(
            (outdir / f"page_{i:02d}.txt").read_text(encoding="utf-8")
            for i in range(1, len(reader.pages) + 1)
        ),
        encoding="utf-8",
    )
    print(f"{joined}  (all {len(reader.pages)} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
