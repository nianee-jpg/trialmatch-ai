"""Deterministic tools used by the Protocol Agent.

Every function here is pure (same input -> same output), works offline, and
never invents values: when the text does not state something, the result says
so with a Flag instead of filling the gap.

Tools:
  segment_criteria      split eligibility text into criteria with exact character offsets
  extract_thresholds    find numeric comparisons ("<1500/mm^3", ">=2 times the ULN", "three or more")
  extract_time_windows  find "within N days/months/years [before <event>]"
  detect_flags          find wording that needs human judgement
  map_required_data     list the patient data fields a criterion depends on
  build_interpretation  write a plain, rule-based statement of the criterion
  verify_extraction     check every criterion against the original text
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from pydantic import BaseModel

from trialmatch.data_dictionary import (
    DATA_FIELDS,
    field_label,
    REQUIRED_DATA_RULES,
    THRESHOLD_PARAMETER_RULES,
    ULN_PARAMETER_RULES,
)
from trialmatch.protocol_models import (
    Comparator,
    Criterion,
    CriterionType,
    Flag,
    FlagCode,
    Logic,
    RequiredData,
    Threshold,
    TimeWindow,
    VerificationIssue,
)
from trialmatch.text_utils import numbers_in, parse_number, unescape_markdown

# --------------------------------------------------------------------------- #
# 1. Segmentation
# --------------------------------------------------------------------------- #


class Segment(BaseModel):
    """One criterion's location in the eligibility text (before analysis)."""

    criterion_id: str
    type: CriterionType
    parent_id: Optional[str] = None
    char_start: int
    char_end: int
    line_number: int
    logic: Optional[Logic] = None
    structure_note: Optional[str] = None


_HEADING = re.compile(r"^[ \t]*(inclusion|exclusion) criteria\s*:?\s*$", re.IGNORECASE)
_TOP_ITEM = re.compile(r"^(\d+)\.[ \t]+")          # "12. text"
_SUB_ITEM = re.compile(r"^[ \t]+[*\-][ \t]+")      # "   * text"
_GROUP_INTRO = re.compile(r"(either|any one|any) of the following:\s*$", re.IGNORECASE)

_PREFIX = {CriterionType.INCLUSION: "INC", CriterionType.EXCLUSION: "EXC"}


def segment_criteria(full_text: str) -> List[Segment]:
    """Split the eligibility text into criteria, keeping exact character offsets.

    Rules:
      * "N. text" at the start of a line is a top-level criterion.
      * An indented "* text" line is a sub-criterion of the last top-level one.
      * A non-blank line that is neither continues the previous criterion.
      * If a sub-criterion ends with "...any one of the following:", the
        sub-items AFTER it are nested under it. This is flagged as
        STRUCTURE_INFERRED because the source shows them at the same indent.
    """
    section: Optional[CriterionType] = None
    raw: List[dict] = []  # flat list before nesting is resolved

    offset = 0
    for line_no, line in enumerate(full_text.split("\n"), start=1):
        line_start, offset = offset, offset + len(line) + 1
        stripped_end = line_start + len(line.rstrip())

        heading = _HEADING.match(line)
        if heading:
            section = CriterionType.INCLUSION if heading.group(1).lower() == "inclusion" else CriterionType.EXCLUSION
            continue
        if not line.strip() or section is None:
            continue

        top, sub = _TOP_ITEM.match(line), _SUB_ITEM.match(line)
        if top:
            raw.append(dict(kind="top", number=int(top.group(1)), type=section,
                            start=line_start + top.end(), end=stripped_end, line=line_no, subs=[]))
        elif sub and raw and raw[-1]["type"] == section:
            raw[-1]["subs"].append(dict(start=line_start + sub.end(), end=stripped_end, line=line_no, children=[]))
        elif raw:
            # Continuation line: extend the most recent criterion or sub-criterion.
            target = raw[-1]["subs"][-1] if raw[-1]["subs"] else raw[-1]
            target["end"] = stripped_end

    segments: List[Segment] = []
    for item in raw:
        top_id = f"{_PREFIX[item['type']]}-{item['number']:02d}"
        top_text = full_text[item["start"]:item["end"]]
        segments.append(Segment(
            criterion_id=top_id, type=item["type"], char_start=item["start"], char_end=item["end"],
            line_number=item["line"], logic=Logic.ANY_OF if item["subs"] and _GROUP_INTRO.search(top_text) else None,
        ))

        # Resolve nesting: sub-items after a "...of the following:" sub-item become its children.
        subs: List[dict] = []
        for sub in item["subs"]:
            if subs and _GROUP_INTRO.search(full_text[subs[-1]["start"]:subs[-1]["end"]]) or (
                subs and subs[-1]["children"]
            ):
                subs[-1]["children"].append(sub)
            else:
                subs.append(sub)

        for i, sub in enumerate(subs):
            sub_id = f"{top_id}.{chr(ord('a') + i)}"
            segments.append(Segment(
                criterion_id=sub_id, type=item["type"], parent_id=top_id,
                char_start=sub["start"], char_end=sub["end"], line_number=sub["line"],
                logic=Logic.ANY_OF if sub["children"] else None,
            ))
            for j, child in enumerate(sub["children"], start=1):
                segments.append(Segment(
                    criterion_id=f"{sub_id}.{j}", type=item["type"], parent_id=sub_id,
                    char_start=child["start"], char_end=child["end"], line_number=child["line"],
                    structure_note=(
                        f"In the published text this item sits at the same level as {top_id}.a and {top_id}.b. "
                        f"The app placed it under {sub_id} because {sub_id} ends with '...any one of the following:'. "
                        "Please confirm against the full protocol."
                    ),
                ))
    return segments


# --------------------------------------------------------------------------- #
# 2. Thresholds
# --------------------------------------------------------------------------- #

# Comparators as they appear in the raw (Markdown-escaped) text: \< \> \>= <= ≥ ≤
_COMPARATOR = re.compile(r"(?P<cmp>\\?[<>](?:\\?=)?|≤|≥)[ \t]*(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)?")
_CMP_MAP = {"<": Comparator.LT, "<=": Comparator.LE, ">": Comparator.GT, ">=": Comparator.GE,
            "≤": Comparator.LE, "≥": Comparator.GE}

# Units recognised right after a number (or right after a comparator with no number).
_UNITS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"/mm\\?\^3"), "/mm^3"),
    (re.compile(r"mL/min/1\.73 ?m\\?\^2"), "mL/min/1.73m^2"),
    (re.compile(r"mg/dL"), "mg/dL"),
    (re.compile(r"g/dL"), "g/dL"),
    (re.compile(r"RBC/hpf"), "/hpf"),
    (re.compile(r"White blood cell per high powered field", re.IGNORECASE), "/hpf"),
    (re.compile(r"times the upper limit of normal", re.IGNORECASE), "x ULN"),
]

_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                 "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12}
_OR_MORE = re.compile(r"\b(?P<num>\d+|" + "|".join(_WORD_NUMBERS) + r") or more (?P<what>courses)\b", re.IGNORECASE)


def _match_unit(text: str, pos: int) -> Tuple[Optional[str], int]:
    """Return (unit, end_position) if a known unit starts at `pos` (after optional spaces)."""
    gap = len(text[pos:]) - len(text[pos:].lstrip(" \t"))
    for pattern, unit in _UNITS:
        m = pattern.match(text, pos + gap)
        if m:
            return unit, m.end()
    return None, pos


def _nearest_parameter(text: str, cmp_start: int, num_end: int) -> Optional[str]:
    """Pick the measurement named closest to the comparator.

    Prefers a keyword just BEFORE the comparator ("platelets <50,000"); falls
    back to one just AFTER the number ("> 5 RBC/hpf").
    """
    before: List[Tuple[int, str]] = []
    after: List[Tuple[int, str]] = []
    for pattern, field in THRESHOLD_PARAMETER_RULES:
        for m in pattern.finditer(text):
            if m.end() <= cmp_start:
                before.append((cmp_start - m.end(), field))
            elif m.start() >= num_end and m.start() - num_end <= 60:
                after.append((m.start() - num_end, field))
    if before:
        return min(before)[1]
    if after:
        return min(after)[1]
    return None


def extract_thresholds(original_text: str) -> Tuple[List[Threshold], List[Flag]]:
    """Find every numeric comparison in a criterion's ORIGINAL text."""
    thresholds: List[Threshold] = []
    flags: List[Flag] = []

    for m in _COMPARATOR.finditer(original_text):
        comparator = _CMP_MAP[m.group("cmp").replace("\\", "")]
        num = m.group("num")
        value = parse_number(num) if num else None
        unit, end = _match_unit(original_text, m.end())
        end = max(end, m.end())
        source_text = original_text[m.start():end]

        if unit == "x ULN":
            # One relative threshold per lab named in the criterion.
            params = [field for pattern, field in ULN_PARAMETER_RULES if pattern.search(original_text)]
            for field in params or [None]:
                thresholds.append(Threshold(parameter=field, comparator=comparator, value=value, unit=None,
                                            relative_to="upper limit of normal", source_text=source_text))
            flags.append(Flag(code=FlagCode.RELATIVE_THRESHOLD, message=(
                f"'{unescape_markdown(source_text)}' depends on each lab's own upper limit of normal, "
                "so the patient's lab reference ranges are needed.")))
        else:
            parameter = _nearest_parameter(original_text, m.start(), m.end())
            thresholds.append(Threshold(parameter=parameter, comparator=comparator, value=value,
                                        unit=unit, source_text=source_text))
            if parameter is None:
                flags.append(Flag(code=FlagCode.THRESHOLD_PARAMETER_UNKNOWN, message=(
                    f"The app found '{unescape_markdown(source_text)}' but could not tell what it measures.")))
            if unit is None and value is not None:
                flags.append(Flag(code=FlagCode.UNIT_NOT_STATED, message=(
                    f"'{unescape_markdown(source_text)}' has no unit. The app will not assume one.")))

        if value is None:
            flags.append(Flag(code=FlagCode.MISSING_THRESHOLD, message=(
                f"The protocol says '{unescape_markdown(source_text)}' but leaves out the number. "
                "The app will not guess it; a person must find the intended value.")))

    for m in _OR_MORE.finditer(original_text):
        token = m.group("num").lower()
        value = float(_WORD_NUMBERS[token]) if token in _WORD_NUMBERS else parse_number(token)
        parameter = _nearest_parameter(original_text, m.start(), m.end())
        thresholds.append(Threshold(parameter=parameter, comparator=Comparator.GE, value=value,
                                    unit=m.group("what").lower(), source_text=m.group(0)))
    return thresholds, flags


# --------------------------------------------------------------------------- #
# 3. Time windows
# --------------------------------------------------------------------------- #

_WINDOW = re.compile(
    r"within\s+(?:the\s+)?(?:(?:last|past|previous)\s+)?"
    r"(?P<num>\d+|" + "|".join(_WORD_NUMBERS) + r")?\s*"
    r"(?P<unit>days?|weeks?|months?|years?)\b"
    r"(?:\s+(?:before|of|prior to)\s+(?P<anchor>visit -1|study enrollment|screening))?",
    re.IGNORECASE,
)


def extract_time_windows(original_text: str) -> Tuple[List[TimeWindow], List[Flag]]:
    windows: List[TimeWindow] = []
    flags: List[Flag] = []
    for m in _WINDOW.finditer(original_text):
        unit = m.group("unit").lower()
        token = m.group("num")
        if token is None:
            if unit.endswith("s"):
                continue  # "within months" has no amount: not a usable window
            amount = 1  # "within the past year" = one year, as written
        else:
            amount = _WORD_NUMBERS.get(token.lower()) or int(token)
        windows.append(TimeWindow(amount=amount, unit=unit if unit.endswith("s") else unit + "s",
                                  anchor=m.group("anchor"), source_text=m.group(0)))
        if m.group("anchor") is None:
            flags.append(Flag(code=FlagCode.TIME_ANCHOR_NOT_STATED, message=(
                f"'{m.group(0)}' does not say which date to count from "
                "(for example screening or first dose). A person must confirm it.")))
    return windows, flags


# --------------------------------------------------------------------------- #
# 4. Wording that needs judgement
# --------------------------------------------------------------------------- #

_SUBJECTIVE = re.compile(
    r"\b(significant|moderately severe|severe|uncontrolled|progressive|consistent with|adequately treated|"
    r"active)\b", re.IGNORECASE)


def detect_flags(original_text: str) -> List[Flag]:
    text = unescape_markdown(original_text)
    flags: List[Flag] = []
    if re.search(r"investigator's opinion", text, re.IGNORECASE):
        flags.append(Flag(code=FlagCode.INVESTIGATOR_JUDGEMENT,
                          message="Depends on the study doctor's opinion, so it cannot be decided from the record alone."))
    terms = sorted({m.group(1).lower() for m in _SUBJECTIVE.finditer(text)})
    if terms:
        flags.append(Flag(code=FlagCode.SUBJECTIVE_TERM,
                          message="Uses words that are not defined here: " + ", ".join(terms) + ". A person must judge them."))
    if re.search(r"including but not limited to", text, re.IGNORECASE):
        flags.append(Flag(code=FlagCode.NON_EXHAUSTIVE_LIST,
                          message="Gives examples ('including but not limited to'), so things not listed may also count."))
    if re.search(r"half-lives", text, re.IGNORECASE):
        flags.append(Flag(code=FlagCode.EXTERNAL_INFO_NEEDED, message=(
            "The time limit depends on each drug's half-life ('5 half-lives ... whichever is greater'), "
            "which the protocol does not give.")))
    return flags


# --------------------------------------------------------------------------- #
# 5. Required data
# --------------------------------------------------------------------------- #

def map_required_data(original_text: str) -> Tuple[List[RequiredData], List[Flag]]:
    text = unescape_markdown(original_text).strip()
    found: List[RequiredData] = []
    for pattern, field, detail in REQUIRED_DATA_RULES:
        if pattern.search(text):
            item = RequiredData(field=field, detail=detail)
            if item not in found:
                found.append(item)
    flags = [] if found else [Flag(code=FlagCode.REQUIRED_DATA_UNMAPPED, message=(
        "The app could not tell which patient information this needs. A person must decide."))]
    return found, flags


# --------------------------------------------------------------------------- #
# 6. Rule-based interpretation
# --------------------------------------------------------------------------- #

_CMP_WORDS = {Comparator.LT: "below", Comparator.LE: "at or below",
              Comparator.GT: "above", Comparator.GE: "at or above"}


def _fmt(value: float) -> str:
    return f"{value:,.0f}" if value == int(value) else f"{value:g}"


def build_interpretation(
    ctype: CriterionType,
    display_text: str,
    thresholds: List[Threshold],
    windows: List[TimeWindow],
    child_ids: List[str],
) -> str:
    """Plain statement of what must be checked, built ONLY from extracted pieces.

    Example: "Excluded if: Absolute neutrophil count (ANC) is below 1,500 /mm^3.
    A value of exactly 1,500 does not count."
    """
    parts: List[str] = []
    lead = "Required" if ctype == CriterionType.INCLUSION else "Excluded if"

    if child_ids:
        parts.append(f"{lead}: any one of {', '.join(child_ids)}.")

    # Group thresholds that share one comparison (e.g. AST, ALT, ALP >= 2 x ULN).
    groups: List[Tuple[Threshold, List[str]]] = []
    for t in thresholds:
        name = field_label(t.parameter) if t.parameter else "[unclear measurement]"
        for first, names in groups:
            if (first.source_text, first.comparator, first.value) == (t.source_text, t.comparator, t.value):
                names.append(name)
                break
        else:
            groups.append((t, [name]))
    for t, names in groups:
        value = _fmt(t.value) if t.value is not None else "[number missing from the protocol]"
        if t.relative_to:
            target = f"{value} × the {t.relative_to}"
        else:
            target = f"{value} {t.unit}" if t.unit else f"{value} (no unit given in the protocol)"
        subject = names[0] if len(names) == 1 else "any of " + ", ".join(names[:-1]) + " or " + names[-1]
        strict = (f" A value of exactly {value} does not count."
                  if t.comparator.is_strict and t.value is not None else "")
        parts.append(f"{lead}: {subject} is {_CMP_WORDS[t.comparator]} {target}.{strict}")

    if not thresholds and not child_ids:
        parts.append(f"{lead}: {display_text}")
    for w in windows:
        # Quote the source wording rather than re-stating the amount ("the past year", not "1 years").
        when = f"counted from {w.anchor}" if w.anchor else "the protocol does not say counted from which date"
        parts.append(f"Time limit: '{unescape_markdown(w.source_text)}' ({when}).")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# 7. Verification
# --------------------------------------------------------------------------- #

_CRITERION_ID = re.compile(r"\b(?:INC|EXC)-\d+(?:\.[a-z](?:\.\d+)?)?")


def source_numbers(original_text: str) -> List[float]:
    """Numbers stated in the source, as digits ("1500") or words ("three")."""
    text = unescape_markdown(original_text)
    words = [float(_WORD_NUMBERS[w.lower()]) for w in re.findall(r"[A-Za-z]+", text) if w.lower() in _WORD_NUMBERS]
    return numbers_in(text) + words


def unsupported_numbers(candidate: str, original_text: str) -> List[float]:
    """Numbers in `candidate` that do not appear in `original_text` (criterion IDs ignored)."""
    allowed = set(source_numbers(original_text))
    return [n for n in numbers_in(_CRITERION_ID.sub("", candidate)) if n not in allowed]


def verify_extraction(criteria: List[Criterion], full_text: str) -> List[VerificationIssue]:
    """Check the extraction against the original eligibility text.

    1. Exact text: original_text == full_text[char_start:char_end]
    2. Coverage: no criterion text in the source was skipped
    3. No invented numbers: every threshold value and every number in the
       interpretation appears in the criterion's original text
    4. Missing thresholds are flagged; required data uses known fields
    5. IDs are unique and every parent exists
    """
    issues: List[VerificationIssue] = []
    ids = [c.criterion_id for c in criteria]

    for c in criteria:
        span = full_text[c.source.char_start:c.source.char_end]
        if span != c.original_text:
            issues.append(VerificationIssue(criterion_id=c.criterion_id, check="exact_text",
                                            message="original_text does not match the source at the recorded offsets."))
        for t in c.thresholds:
            if t.source_text not in c.original_text:
                issues.append(VerificationIssue(criterion_id=c.criterion_id, check="threshold_source",
                                                message=f"Threshold source '{t.source_text}' not found in original text."))
            if t.value is not None and t.value not in source_numbers(c.original_text):
                issues.append(VerificationIssue(criterion_id=c.criterion_id, check="invented_number",
                                                message=f"Threshold value {t.value} does not appear in the original text."))
            if t.value is None and not c.has_flag(FlagCode.MISSING_THRESHOLD):
                issues.append(VerificationIssue(criterion_id=c.criterion_id, check="missing_threshold_flag",
                                                message="A threshold has no value but is not flagged MISSING_THRESHOLD."))
        bad = unsupported_numbers(c.normalized_interpretation, c.original_text)
        if bad:
            issues.append(VerificationIssue(criterion_id=c.criterion_id, check="invented_number",
                                            message=f"Interpretation contains numbers not in the source: {bad}."))
        for rd in c.required_data:
            if rd.field not in DATA_FIELDS:
                issues.append(VerificationIssue(criterion_id=c.criterion_id, check="unknown_field",
                                                message=f"Required data field '{rd.field}' is not in the data dictionary."))
        if c.parent_id and c.parent_id not in ids:
            issues.append(VerificationIssue(criterion_id=c.criterion_id, check="parent_exists",
                                            message=f"Parent {c.parent_id} does not exist."))

    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        issues.append(VerificationIssue(criterion_id=dup, check="unique_id", message="Duplicate criterion ID."))

    # Coverage: mark every character that belongs to a criterion, heading or list marker.
    covered = [False] * len(full_text)
    for c in criteria:
        for i in range(c.source.char_start, c.source.char_end):
            covered[i] = True
    offset = 0
    for line in full_text.split("\n"):
        marker = _HEADING.match(line) or _TOP_ITEM.match(line) or _SUB_ITEM.match(line)
        if marker:
            for i in range(offset, offset + marker.end()):
                covered[i] = True
        offset += len(line) + 1
    uncovered = "".join(ch if not covered[i] else "\n" for i, ch in enumerate(full_text))
    for fragment in (f.strip() for f in uncovered.split("\n")):
        if fragment:
            issues.append(VerificationIssue(criterion_id=None, check="coverage",
                                            message=f"Source text not assigned to any criterion: '{fragment[:80]}'"))
    return issues
