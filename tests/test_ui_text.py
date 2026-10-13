"""Every internal code shown in the dashboard has a plain-language translation."""

from trialmatch.agents.eligibility_agent import SUBMIT_SUGGESTION_TOOL
from trialmatch.agents.patient_agent import PATIENT_TOOLS
from trialmatch.agents.protocol_agent import CLAUDE_TOOLS
from trialmatch.protocol_models import FlagCode
from trialmatch.ui_text import FLAG_INFO, TOOL_LABELS, plain_log_detail

PROTOCOL_TOOLS = ["segment_criteria", "extract_thresholds", "extract_time_windows", "detect_flags",
                  "map_required_data", "verify_extraction"]


def test_every_flag_has_a_title():
    assert {c.value for c in FlagCode} == set(FLAG_INFO)


def test_every_tool_has_a_label():
    names = (set(PROTOCOL_TOOLS) | {t["name"] for t in PATIENT_TOOLS} | {t["name"] for t in CLAUDE_TOOLS}
             | {SUBMIT_SUGGESTION_TOOL["name"]})
    assert names <= set(TOOL_LABELS)


def test_log_details_are_translated():
    assert plain_log_detail('{"patient_id": "SYN-003", "field": "labs.hemoglobin"}') == "SYN-003 · Hemoglobin"
    assert plain_log_detail("NOT_RECORDED · labs.iga · 0 observation(s)") == (
        "not in the record · IgA (immunoglobulin A) · 0 entries")
    assert plain_log_detail("thresholds: labs.anc < 1500.0 | flags: MISSING_THRESHOLD") == (
        "numbers found: Absolute neutrophil count (ANC) below 1500 | needs a person: number missing from the protocol")
    assert plain_log_detail("[] | []") == "nothing found"
