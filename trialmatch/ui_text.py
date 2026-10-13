"""Plain-language wording for the dashboard.

Internal code names (flag codes, tool names, field names) stay in the data so
results remain traceable; this module translates them for people.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Tuple

from trialmatch.data_dictionary import DATA_FIELDS, field_label

# flag code -> (short title, what it means for the reviewer)
FLAG_INFO: Dict[str, Tuple[str, str]] = {
    "MISSING_THRESHOLD": ("Number missing from the protocol",
                          "The protocol compares against a value but never states it."),
    "UNIT_NOT_STATED": ("No unit given", "The protocol gives a number without a unit."),
    "RELATIVE_THRESHOLD": ("Depends on the lab's normal range",
                           "The limit is a multiple of each lab's own upper limit of normal."),
    "EXTERNAL_INFO_NEEDED": ("Needs information not in the protocol",
                             "For example, a drug's half-life."),
    "TIME_ANCHOR_NOT_STATED": ("Start date of time limit not stated",
                               "The protocol says 'within N months' but not counted from which date."),
    "INVESTIGATOR_JUDGEMENT": ("Study doctor's judgement", "Depends on the investigator's opinion."),
    "SUBJECTIVE_TERM": ("Wording open to interpretation",
                        "Uses words such as 'severe' or 'significant' without a definition."),
    "NON_EXHAUSTIVE_LIST": ("Examples, not a complete list",
                            "Says 'including but not limited to', so other items may also count."),
    "STRUCTURE_INFERRED": ("Grouping worked out by the app",
                           "The published layout was ambiguous; the app grouped these items by a rule."),
    "THRESHOLD_PARAMETER_UNKNOWN": ("Unclear what is being measured",
                                    "A number was found but not what it applies to."),
    "REQUIRED_DATA_UNMAPPED": ("Unclear what information is needed",
                               "The app could not match this criterion to patient information."),
    "AMBIGUITY_NOTED_BY_CLAUDE": ("Point raised by Claude", "Claude noted something a reviewer should resolve."),
}


def flag_title(code: str) -> str:
    return FLAG_INFO.get(code, (code, ""))[0]


# tool name -> plain description
TOOL_LABELS: Dict[str, str] = {
    "segment_criteria": "Split the text into criteria",
    "extract_thresholds": "Find numbers to check",
    "extract_time_windows": "Find time limits",
    "detect_flags": "Find wording that needs a person",
    "map_required_data": "Work out what information is needed",
    "verify_extraction": "Accuracy check",
    "list_criteria": "Read the list of criteria",
    "get_criterion_details": "Read one criterion in detail",
    "submit_interpretation": "Suggest plain-language wording",
    "list_patients": "List practice patients",
    "get_patient_overview": "Get patient summary",
    "list_recorded_fields": "List what the record contains",
    "get_field": "Look up a value",
    "get_medication_history": "Search medication history",
}

EVENT_LABELS: Dict[str, str] = {
    "tool_call": "Step started",
    "tool_result": "Step result",
    "info": "Note",
    "warning": "Warning",
    "verification": "Accuracy check",
    "llm_request": "Sent to Claude",
    "llm_response": "Claude replied",
}

STATUS_WORDS = {
    "FOUND": "found",
    "NOT_RECORDED": "not in the record",
    "CONFLICTING": "sources disagree",
    "MULTIPLE_DATES": "several dates",
    "UNKNOWN_FIELD": "unknown item",
    "UNKNOWN_PATIENT": "unknown patient",
    "ERROR": "error",
}

_FIELD_PATTERN = re.compile("|".join(re.escape(f) for f in sorted(DATA_FIELDS, key=len, reverse=True)))


def plain_fields(text: str) -> str:
    """Replace every internal field name in `text` with its readable label."""
    return _FIELD_PATTERN.sub(lambda m: field_label(m.group(0)), text)


def plain_log_detail(detail: str) -> str:
    """Make an activity-log detail line readable."""
    try:
        data = json.loads(detail)
    except ValueError:
        data = None
    if isinstance(data, dict):
        parts = []
        for key, value in data.items():
            if key == "field":
                parts.append(field_label(str(value)))
            elif key == "drug":
                parts.append(f"medicine: {value or 'all'}")
            elif key == "text":
                parts.append(f"“{value}”")
            else:
                parts.append(str(value))
        return " · ".join(parts) if parts else "(no input)"
    text = plain_fields(detail)
    for code, word in STATUS_WORDS.items():
        text = re.sub(rf"\b{code}\b", word, text)
    for code, (title, _) in FLAG_INFO.items():
        text = re.sub(rf"\b{code}\b", title.lower(), text)
    for old, new in _LOG_PHRASES:
        text = text.replace(old, new)
    text = re.sub(r"(\d)\.0\b", r"\1", text)  # 1500.0 -> 1500
    text = " | ".join(part for part in (p.strip() for p in text.split("|")) if part != "[]") or "nothing found"
    return text


_LOG_PHRASES = [
    ("observation(s)", "entries"),
    ("thresholds: ", "numbers found: "),
    ("windows: ", "time limits: "),
    ("flags: ", "needs a person: "),
    ("fields: ", "information needed: "),
    ("no anchor", "start date not stated"),
    (" < ", " below "), (" <= ", " at or below "), (" > ", " above "), (" >= ", " at or above "),
    ("MISSING", "[missing]"),
]


# Pre-screening results ------------------------------------------------------

RESULT_LABELS: Dict[str, str] = {
    "MET": "✅ MET",
    "NOT_MET": "❌ NOT MET",
    "PRESENT": "⛔ PRESENT",
    "ABSENT": "✅ ABSENT",
    "UNKNOWN": "❔ UNKNOWN",
    "NEEDS_HUMAN_REVIEW": "👤 NEEDS REVIEW",
}

RESULT_MEANINGS: Dict[str, str] = {
    "MET": "The inclusion requirement is satisfied by the record.",
    "NOT_MET": "The record shows the inclusion requirement is not satisfied.",
    "PRESENT": "The record shows this exclusion applies.",
    "ABSENT": "The record shows this exclusion does not apply.",
    "UNKNOWN": "Information needed to check this is missing from the record.",
    "NEEDS_HUMAN_REVIEW": "The app will not decide this: the wording needs judgement, sources disagree, "
                          "or the protocol leaves something out.",
}

METHOD_LABELS: Dict[str, str] = {
    "deterministic": "Calculated by the app",
    "record lookup": "Read from the record",
    "combined": "Combined from its parts",
    "needs a person": "Left for a person",
}

TOOL_LABELS["submit_suggestion"] = "Suggest a reading (advisory only)"


# Verification ---------------------------------------------------------------

OUTCOME_LABELS: Dict[str, str] = {
    "VERIFIED": "✅ Verified",
    "VERIFIED_WITH_WARNINGS": "⚠️ Verified, with warnings",
    "FAILED": "❌ Failed verification",
}

CHECK_LABELS: Dict[str, str] = {
    "coverage": "Every criterion checked once",
    "protocol_extraction": "Criteria extraction passed its accuracy check",
    "synthetic": "Record is labelled SYNTHETIC",
    "criterion_text": "Criterion matches the protocol text",
    "evidence_matches": "Evidence matches the record",
    "evidence_present": "Result is backed by evidence",
    "missing_threshold": "Protocol gaps were not decided",
    "contradiction": "Conflicting entries were not decided",
    "recomputed": "Comparison gives the same answer when redone",
    "combination": "Parts and whole agree",
    "dates_supported": "Dates come from the record",
    "numbers_supported": "Numbers come from the protocol or record",
    "missing_confirmed": "Missing information is really missing",
    "decision_language": "No final-decision wording",
    "assumption": "Assumption made",
    "ai_suggestion": "Claude suggestion is advisory and cites real evidence",
}

