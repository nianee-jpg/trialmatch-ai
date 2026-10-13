"""Small text helpers shared by the Protocol Agent and its tools."""

from __future__ import annotations

import re
from typing import List

# ClinicalTrials.gov stores eligibility text as Markdown, so literal symbols are
# escaped with a backslash: "\<1500", "\>5", "mm\^3", "\[AST...\]".
_MARKDOWN_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!<>^|~=])")

# A number such as 5, 1.73, 1500 or 50,000 (thousands separators allowed).
_NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])\d+(?:\.\d+)?")


def unescape_markdown(text: str) -> str:
    """Remove Markdown backslash escapes: '\\<1500/mm\\^3' -> '<1500/mm^3'.

    Used only for *reading* the text. The exact original (escaped) text is
    always kept separately for traceability.
    """
    return _MARKDOWN_ESCAPE.sub(r"\1", text)


def parse_number(token: str) -> float:
    """'50,000' -> 50000.0"""
    return float(token.replace(",", ""))


def numbers_in(text: str) -> List[float]:
    """Every number that appears in `text`, in order."""
    return [parse_number(m.group(0)) for m in _NUMBER.finditer(text)]
