"""Structural pre-pass (Stage A).

Clinical notes are not flat prose. Headers, footers, signature blocks, tables and form
fields carry a disproportionate share of the identifiers -- P2.7 requires the test set to
contain "identifiers hiding in tables, headers and signature blocks", and P2.8 lists
structured-segment PHI as a stretch goal.

Two things come out of this stage:

* ``segments`` -- regions tagged by kind. Spans landing in one get their detection
  threshold scaled *down* (``NeuralPolicy.structured_segment_threshold_scale``), i.e. we
  become more suspicious where PHI is dense, not less.
* ``form_fields`` -- ``key: value`` pairs with offsets. The key tells us what the value
  is, which is the single largest recall win available without a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..lexicons import CREDENTIAL_ALT
from ..types import Segment

#: A cell separator: two-plus spaces, a tab, or a pipe. Splitting on these is what lets
#: "Name: John Wood      DOB: 01/02/1950" parse as two fields rather than one.
_CELL_SEP = re.compile(r"\s{2,}|\t+|\|")

_FIELD_IN_CELL = re.compile(
    r"^\s*(?P<key>[A-Za-z][A-Za-z0-9 /_\-.'&#()]{0,44}?)\s*[:=]\s*(?P<val>.*?)\s*$"
)

_SIGNATURE_LINE = re.compile(
    r"(?im)^\s*(?:(?:electronically\s+)?(?:signed|e-signed|dictated|transcribed|"
    r"authenticated|verified|reviewed|entered|approved)\s*(?:by)?\b|/s/|"
    r"attending\s*(?:physician)?\s*[:\-]|physician\s*signature|signature\s*[:\-])"
)

#: Trailing credential, e.g. "Alan Reyes, M.D." -- a strong signature-block marker.
_CREDENTIAL_TAIL = re.compile(rf",\s*(?:{CREDENTIAL_ALT})\s*$", re.IGNORECASE)

#: A cell that is a bare form-field key: no colon, no comma, so a two-column row parses as
#: key/value while a name ("WHITFIELD, MARCUS D.") or a timestamp ("17:56") never can.
_KEY_ONLY = re.compile(r"^[A-Za-z][A-Za-z0-9 /_\-.'&#()]{0,44}$")

_TABLE_LINE = re.compile(r"\S(?:\s{2,}|\t|\s*\|\s*)\S")

#: A form-field value is a field, not prose. Cells are bounded by two-plus spaces, a tab or
#: a pipe -- so on a run-on single-line note ("DOB: 03/14/1952. Admitted 03/14/2024.") the
#: whole line is one cell and the value swallows the entire note, which the field detector
#: then masks as one date. Cut at the first sentence break instead.
#:
#: The lookbehind requires a lowercase letter or digit before the period, so "E. Smith" and
#: "SMITH, JOHN A. MD" are never cut at an initial; ``_ABBREV_TAIL`` covers the lowercase
#: abbreviations that would otherwise truncate an address ("12 St. Mary Rd").
_SENTENCE_BREAK = re.compile(r"(?<=[a-z0-9)\]])\.\s+(?=[A-Z])")
_ABBREV_TAIL = re.compile(
    r"(?i)\b(?:st|mr|mrs|ms|dr|jr|sr|prof|rev|ave|rd|blvd|apt|ste|no|vs|approx|est|dept)$"
)


def _value_end(value: str) -> int:
    """Offset within ``value`` where the field ends. See ``_SENTENCE_BREAK``."""
    # A parenthetical in a form value is an annotation, not part of the value:
    # "03/14/1987 (Age 36)". Swallowing it masks the age along with the date, and an age
    # under 90 is retained by policy on purpose.
    paren = value.find(" (")
    for m in _SENTENCE_BREAK.finditer(value):
        if _ABBREV_TAIL.search(value[: m.start()]):
            continue
        return m.start() if paren < 0 else min(m.start(), paren)
    return len(value) if paren < 0 else paren

#: Lines that look like a report banner / letterhead.
_BANNER = re.compile(
    r"(?i)^\s*[-=_*#]{4,}\s*$|^\s*\**\s*(?:confidential|page\s+\d+\s+of\s+\d+|"
    r"patient\s+information|demographics|face\s*sheet|discharge\s+summary|"
    r"radiology\s+report|laboratory\s+report|operative\s+report|"
    r"history\s+and\s+physical|consultation\s+note|progress\s+note)\b"
)

_HEADER_MAX_LINES = 18
_FOOTER_MAX_LINES = 12


@dataclass(frozen=True)
class FormField:
    """A ``key: value`` pair with absolute offsets into the original text."""

    key: str
    key_start: int
    key_end: int
    value: str
    value_start: int
    value_end: int
    line_index: int
    in_table: bool = False


@dataclass(frozen=True)
class Line:
    text: str
    start: int
    end: int
    index: int

    @property
    def stripped(self) -> str:
        return self.text.strip()

    @property
    def blank(self) -> bool:
        return not self.stripped


def split_lines(text: str) -> list[Line]:
    lines: list[Line] = []
    pos = 0
    for i, raw in enumerate(text.splitlines(keepends=True)):
        body = raw.rstrip("\r\n")
        lines.append(Line(text=body, start=pos, end=pos + len(body), index=i))
        pos += len(raw)
    return lines


def _cell_spans(line: Line) -> list[tuple[int, int]]:
    """Absolute [start, end) spans of the cells on a line."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _CELL_SEP.finditer(line.text):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(line.text):
        spans.append((pos, len(line.text)))
    return [(line.start + s, line.start + e) for s, e in spans]


def form_fields(text: str) -> list[FormField]:
    """Extract ``key: value`` pairs, including several per line.

    Offsets are absolute so downstream stages can mask the value without re-searching.
    """
    out: list[FormField] = []
    for line in split_lines(text):
        if line.blank:
            continue
        in_table = bool(_TABLE_LINE.search(line.text))
        cells = _cell_spans(line)
        # A line with no cell separators is still a candidate single field.
        if not cells:
            cells = [(line.start, line.end)]
        before = len(out)
        for cs, ce in cells:
            cell = text[cs:ce]
            m = _FIELD_IN_CELL.match(cell)
            if not m:
                continue
            value = m.group("val")[: _value_end(m.group("val"))].rstrip()
            if not value:
                continue
            key_start = cs + m.start("key")
            key_end = cs + m.end("key")
            val_start = cs + m.start("val")
            val_end = val_start + len(value)
            out.append(
                FormField(
                    key=m.group("key").strip(),
                    key_start=key_start,
                    key_end=key_end,
                    value=value,
                    value_start=val_start,
                    value_end=val_end,
                    line_index=line.index,
                    in_table=in_table,
                )
            )
        if len(out) > before or len(cells) < 2 or len(cells) % 2:
            continue
        # Colonless two-column form. Layout-extracted PDF text writes
        # "Patient Name<spaces>Whitfield, Marcus D." with no colon at all, so a colon-only
        # detector is blind to nearly every identifier in a produced record -- the field cue
        # is the largest recall win in the rule layer and it was firing on none of them.
        # Pairing even cells with the next one covers 2- and 4-column forms; requiring an
        # even count and a bare-key left cell is what keeps prose rows (bullets, timestamped
        # vitals, "17:56  BP 148/88") from pairing up by accident. Whether the key means
        # anything is decided downstream by FIELD_KEY_CATEGORY, not here.
        if not all(_KEY_ONLY.match(text[s:e].strip()) for s, e in cells[::2]):
            continue
        for (ks, ke), (vs, ve) in zip(cells[::2], cells[1::2]):
            raw = text[vs:ve]
            value = raw.strip()
            value = value[: _value_end(value)].rstrip()
            if not value:
                continue
            val_start = vs + (len(raw) - len(raw.lstrip()))
            out.append(
                FormField(
                    key=text[ks:ke].strip(),
                    key_start=ks,
                    key_end=ke,
                    value=value,
                    value_start=val_start,
                    value_end=val_start + len(value),
                    line_index=line.index,
                    in_table=in_table,
                )
            )
    return out


def _classify_line(line: Line) -> str:
    if line.blank:
        return "blank"
    if _BANNER.match(line.text):
        return "banner"
    if _SIGNATURE_LINE.search(line.text) or _CREDENTIAL_TAIL.search(line.text):
        return "signature"
    if _TABLE_LINE.search(line.text):
        return "table"
    if _FIELD_IN_CELL.match(line.text) and len(line.stripped) < 120:
        return "field"
    return "body"


def segment(text: str) -> list[Segment]:
    """Tag structural regions of the note.

    Deliberately conservative: a region is only claimed as header/footer when the lines
    really are field/table/banner shaped. Over-claiming would drag the threshold down
    across the whole note and drown the output in false positives.
    """
    lines = split_lines(text)
    if not lines:
        return []

    kinds = [_classify_line(ln) for ln in lines]
    segments: list[Segment] = []

    structured = {"field", "table", "banner", "blank"}

    # --- header: leading run of structured lines -----------------------------------
    i = 0
    while i < min(len(lines), _HEADER_MAX_LINES) and kinds[i] in structured:
        i += 1
    # Trim trailing blanks off the claimed header.
    while i > 0 and kinds[i - 1] == "blank":
        i -= 1
    if i > 0:
        segments.append(Segment(start=lines[0].start, end=lines[i - 1].end, kind="header"))
    header_end_line = i

    # --- signature block: from the first signature line at/after the body ----------
    sig_start_line: int | None = None
    for j in range(header_end_line, len(lines)):
        if kinds[j] == "signature":
            sig_start_line = j
            break
    if sig_start_line is not None:
        segments.append(
            Segment(start=lines[sig_start_line].start, end=lines[-1].end, kind="signature")
        )

    # --- footer: trailing run of structured lines ----------------------------------
    k = len(lines) - 1
    limit = max(header_end_line, len(lines) - _FOOTER_MAX_LINES)
    while k >= limit and kinds[k] in structured:
        k -= 1
    if k < len(lines) - 1:
        first = k + 1
        while first < len(lines) - 1 and kinds[first] == "blank":
            first += 1
        if first <= len(lines) - 1:
            segments.append(Segment(start=lines[first].start, end=lines[-1].end, kind="footer"))

    # --- tables and standalone field lines in the body -----------------------------
    j = header_end_line
    while j < len(lines):
        if kinds[j] in ("table", "field"):
            kind = kinds[j]
            start_line = j
            while j + 1 < len(lines) and kinds[j + 1] == kind:
                j += 1
            segments.append(
                Segment(start=lines[start_line].start, end=lines[j].end, kind=kind)
            )
        j += 1

    return segments


def in_structured_segment(pos: int, segments: list[Segment]) -> bool:
    return any(s.contains_pos(pos) for s in segments)
