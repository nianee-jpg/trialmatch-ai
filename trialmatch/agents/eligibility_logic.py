"""Pure helper logic for the Eligibility Agent (no I/O, easy to test).

Decisions use four internal values:
  TRUE     the criterion's condition holds   (inclusion -> MET,     exclusion -> PRESENT)
  FALSE    it does not hold                  (inclusion -> NOT_MET, exclusion -> ABSENT)
  UNKNOWN  information needed is missing
  REVIEW   a person must decide (conflicting evidence, unclear wording, missing threshold ...)
"""

from __future__ import annotations

import calendar
import datetime as dt
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

from trialmatch.protocol_models import Comparator


class D(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"
    REVIEW = "REVIEW"


@dataclass
class Finding:
    """Result of checking one condition, with everything needed to explain it."""

    decision: D
    reason: str
    comparisons: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    review_reasons: List[str] = field(default_factory=list)
    method: str = "deterministic"


def merge(decision: D, reason: str, parts: Iterable[Finding], method: Optional[str] = None) -> Finding:
    """Combine several findings into one, keeping all their evidence and notes."""
    parts = list(parts)
    out = Finding(decision, reason, method=method or (parts[0].method if len(parts) == 1 else "combined"))
    for p in parts:
        for attr in ("comparisons", "evidence", "assumptions", "missing", "review_reasons"):
            for item in getattr(p, attr):
                if item not in getattr(out, attr):
                    getattr(out, attr).append(item)
    return out


# --------------------------------------------------------------------------- #
# Combining decisions
# --------------------------------------------------------------------------- #

def combine_any(decisions: Iterable[D]) -> D:
    """OR: one TRUE is enough; FALSE only if every part is FALSE."""
    ds = list(decisions)
    if D.TRUE in ds:
        return D.TRUE
    if ds and all(d == D.FALSE for d in ds):
        return D.FALSE
    return D.REVIEW if D.REVIEW in ds else D.UNKNOWN


def combine_all(decisions: Iterable[D]) -> D:
    """AND: one FALSE is enough; TRUE only if every part is TRUE."""
    ds = list(decisions)
    if D.FALSE in ds:
        return D.FALSE
    if ds and all(d == D.TRUE for d in ds):
        return D.TRUE
    return D.REVIEW if D.REVIEW in ds else D.UNKNOWN


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #

CMP_WORDS = {Comparator.LT: "below", Comparator.LE: "at or below",
             Comparator.GT: "above", Comparator.GE: "at or above"}


def compare(value: float, comparator: Comparator, limit: float) -> bool:
    """Exact comparison. Strict comparators exclude equality."""
    return {
        Comparator.LT: value < limit,
        Comparator.LE: value <= limit,
        Comparator.GT: value > limit,
        Comparator.GE: value >= limit,
    }[comparator]


def fmt_num(x: float) -> str:
    return f"{x:,.0f}" if float(x) == int(x) else f"{x:,g}"


def describe_comparison(label: str, value: float, unit: str, comparator: Comparator, limit: float,
                        limit_text: Optional[str] = None) -> str:
    """'ANC 1,500 /mm^3 is not below 1,500 /mm^3 (a value exactly at the limit does not count).'"""
    holds = compare(value, comparator, limit)
    words = CMP_WORDS[comparator]
    target = limit_text or f"{fmt_num(limit)} {unit}".strip()
    text = f"{label} {fmt_num(value)} {unit}".strip() + f" is {'' if holds else 'not '}{words} {target}"
    if value == limit:
        text += " (exactly at the limit, which " + ("counts" if not comparator.is_strict else "does not count") + ")"
    return text + "."


def as_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #

def parse_date(value: Any) -> Optional[dt.date]:
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def subtract_months(d: dt.date, months: int) -> dt.date:
    """Calendar months: 2015-03-31 minus 1 month -> 2015-02-28."""
    total = d.year * 12 + (d.month - 1) - months
    year, month = divmod(total, 12)
    month += 1
    return dt.date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def window_start(anchor: dt.date, amount: int, unit: str) -> dt.date:
    unit = unit.rstrip("s")
    if unit == "day":
        return anchor - dt.timedelta(days=amount)
    if unit == "week":
        return anchor - dt.timedelta(weeks=amount)
    if unit == "month":
        return subtract_months(anchor, amount)
    if unit == "year":
        return subtract_months(anchor, 12 * amount)
    raise ValueError(f"Unknown time unit: {unit}")


def in_window(event: dt.date, anchor: dt.date, amount: int, unit: str) -> D:
    """Is `event` within `amount unit` before `anchor`?

    Anything on or after the start of the window counts as inside, including
    events after the anchor (e.g. ongoing treatment). An event exactly on the
    first day of the window is REVIEW: the protocol does not say whether that
    boundary day counts.
    """
    start = window_start(anchor, amount, unit)
    if event == start:
        return D.REVIEW
    return D.TRUE if event > start else D.FALSE


def is_ongoing(observation: Dict[str, Any]) -> bool:
    note = (observation.get("note") or "").lower()
    return "ongoing" in note and not observation.get("end_date")


def exposure_date(observation: Dict[str, Any], anchor: dt.date) -> Optional[dt.date]:
    """Last date of exposure for a treatment entry, or None if the record does not say."""
    if observation.get("end_date"):
        return parse_date(observation["end_date"])
    if is_ongoing(observation):
        return anchor  # still being taken at the reference date
    return None


# --------------------------------------------------------------------------- #
# Text results
# --------------------------------------------------------------------------- #

def classify_result(value: Any, note: Optional[str] = None) -> Optional[str]:
    """'positive' / 'negative' / 'indeterminate' from a documented result, else None."""
    for text in (value, note):
        if not isinstance(text, str):
            continue
        t = text.strip().lower()
        if "indeterminate" in t:
            return "indeterminate"
        has_pos, has_neg = "positive" in t, "negative" in t
        if has_pos != has_neg:  # exactly one of the two words appears
            return "positive" if has_pos else "negative"
    return None


def says_none(value: Any) -> Optional[bool]:
    """True if a documented answer is a clear 'no' (False, 'None ...', 'No ...')."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        t = value.strip().lower()
        if t.startswith(("none", "no ", "no.", "not ")) or t == "no":
            return True
    return None


_BIOPSY_CLASS = re.compile(r"\b(VI|IV|V|III|II|I)\b")


def biopsy_classes(text: str) -> List[str]:
    """Roman-numeral ISN/RPS classes mentioned in a biopsy result."""
    return _BIOPSY_CLASS.findall(text.replace("-", " "))


def biopsy_is_proliferative(text: str) -> Optional[bool]:
    """Protocol: 'Class III, Class IV, or Class V in combination with Class III or IV'.

    True if class III or IV is present (alone or with V); False if classes are
    stated but none is III/IV; None if no class can be read.
    """
    classes = biopsy_classes(text)
    if not classes:
        return None
    return "III" in classes or "IV" in classes
