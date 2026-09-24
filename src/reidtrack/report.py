"""Plain-text tables for command-line reports, laid out like a paper table."""

from __future__ import annotations

from collections.abc import Sequence


def format_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    caption: str | None = None,
    footer: Sequence[Sequence[object]] = (),
    note: str | None = None,
) -> str:
    """Top rule, header, mid rule, rows, optional footer rows, bottom rule.

    The first column is left-aligned and the rest right-aligned.
    """
    cells = [[str(c) for c in row] for row in (headers, *rows, *footer)]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    rule = "-" * (sum(widths) + 2 * (len(widths) - 1))

    def line(row: list[str]) -> str:
        return "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(row, widths)))

    body = [line(row) for row in cells[1 : 1 + len(rows)]]
    out = [caption, ""] if caption else []
    out += [rule, line(cells[0]), rule, *body]
    if footer:
        out += [rule, *(line(row) for row in cells[1 + len(rows) :])]
    out.append(rule)
    if note:
        out.append(note)
    return "\n".join(out)
