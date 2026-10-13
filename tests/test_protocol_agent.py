"""End-to-end tests of the Protocol Agent on the real CALIBRATE eligibility text."""

from types import SimpleNamespace

import pytest

from trialmatch.activity_log import ActivityLog
from trialmatch.agents.protocol_agent import ProtocolAgent
from trialmatch.agents.protocol_tools import verify_extraction
from trialmatch.protocol_models import Comparator, FlagCode


@pytest.fixture(scope="module")
def extraction(calibrate_eligibility_text):
    return ProtocolAgent(ActivityLog()).run("NCT02260934", calibrate_eligibility_text)


def test_offline_mode_and_verification_pass(extraction):
    assert extraction.mode == "offline (rule-based)"
    assert extraction.verification_issues == []


def test_every_numbered_criterion_is_extracted(extraction):
    top = [c for c in extraction.criteria if c.parent_id is None]
    assert [c.criterion_id for c in top if c.type.value == "INCLUSION"] == [f"INC-{i:02d}" for i in range(1, 6)]
    assert [c.criterion_id for c in top if c.type.value == "EXCLUSION"] == [f"EXC-{i:02d}" for i in range(1, 36)]


def test_original_text_is_exact(extraction, calibrate_eligibility_text):
    for c in extraction.criteria:
        assert calibrate_eligibility_text[c.source.char_start:c.source.char_end] == c.original_text
    assert extraction.by_id("EXC-02").original_text == r"Neutropenia (absolute neutrophil count \<1500/mm\^3)."


def test_urinary_sediment_nesting_is_flagged(extraction):
    b = extraction.by_id("INC-03.b")
    children = [c for c in extraction.criteria if c.parent_id == "INC-03.b"]
    assert b.logic is not None and len(children) == 3
    assert all(c.has_flag(FlagCode.STRUCTURE_INFERRED) for c in children)


def test_known_thresholds(extraction):
    anc = extraction.by_id("EXC-02").thresholds[0]
    assert (anc.parameter, anc.comparator, anc.value) == ("labs.anc", Comparator.LT, 1500.0)
    egfr = extraction.by_id("EXC-14").thresholds[0]
    assert (egfr.parameter, egfr.value, egfr.unit) == ("labs.egfr", 20.0, "mL/min/1.73m^2")


def test_anemia_threshold_missing_in_source(extraction):
    c = extraction.by_id("EXC-04")
    assert c.thresholds[0].value is None
    assert c.has_flag(FlagCode.MISSING_THRESHOLD)
    assert "[number missing from the protocol]" in c.normalized_interpretation


def test_interpretations_state_strictness(extraction):
    text = extraction.by_id("EXC-02").normalized_interpretation
    assert "Absolute neutrophil count (ANC) is below 1,500 /mm^3" in text
    assert "exactly 1,500 does not count" in text


def test_activity_log_records_tool_calls(calibrate_eligibility_text):
    log = ActivityLog()
    ProtocolAgent(log).run("NCT02260934", calibrate_eligibility_text)
    tools_called = {e.tool for e in log.entries if e.event == "tool_call"}
    assert {"segment_criteria", "extract_thresholds", "extract_time_windows", "detect_flags",
            "map_required_data", "verify_extraction"} <= tools_called
    assert any(e.event == "verification" and e.detail.startswith("Passed") for e in log.entries)


# --- Verification catches problems ---------------------------------------- #

def test_verification_detects_altered_text(extraction, calibrate_eligibility_text):
    c = extraction.by_id("EXC-03").model_copy(deep=True)
    c.original_text = c.original_text.replace("50,000", "5,000")
    issues = verify_extraction([c], calibrate_eligibility_text)
    assert any(i.check == "exact_text" for i in issues)


def test_verification_detects_invented_threshold(extraction, calibrate_eligibility_text):
    c = extraction.by_id("EXC-04").model_copy(deep=True)
    c.thresholds[0].value = 10.0  # a plausible-looking but invented number
    checks = {i.check for i in verify_extraction([c], calibrate_eligibility_text)}
    assert "invented_number" in checks


def test_verification_detects_skipped_criteria(extraction, calibrate_eligibility_text):
    issues = verify_extraction(extraction.criteria[:-1], calibrate_eligibility_text)
    assert any(i.check == "coverage" and "Inability to comply" in i.message for i in issues)


# --- Optional Claude mode, with a fake client (no network, no key) -------- #

def _tool_use(id_, name, inp):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


class FakeClaude:
    """Replays scripted responses and records the requests it receives."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def test_claude_mode_rejects_invented_numbers(calibrate_eligibility_text):
    client = FakeClaude([
        SimpleNamespace(stop_reason="tool_use", content=[
            _tool_use("t1", "submit_interpretation",
                      {"criterion_id": "EXC-04", "interpretation": "Exclude if Hgb < 8 g/dL.", "ambiguity_notes": []}),
        ]),
        SimpleNamespace(stop_reason="tool_use", content=[
            _tool_use("t2", "submit_interpretation",
                      {"criterion_id": "EXC-04",
                       "interpretation": "Exclude moderately severe anemia; the protocol gives no hemoglobin cutoff.",
                       "ambiguity_notes": ["Hemoglobin threshold is blank in the registry text."]}),
        ]),
        SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="Done.")]),
    ])
    log = ActivityLog()
    result = ProtocolAgent(log, client=client).run("NCT02260934", calibrate_eligibility_text)

    # First submission (invented "8") was rejected and fed back to Claude as a tool result.
    tool_results = [block for m in client.requests[-1]["messages"] if m["role"] == "user"
                    and isinstance(m["content"], list) for block in m["content"]]
    first_result = next(b["content"] for b in tool_results if b["tool_use_id"] == "t1")
    assert "rejected" in first_result and "8" in first_result

    c = result.by_id("EXC-04")
    assert result.mode == "claude"
    assert c.interpretation_source == "claude"
    assert "8" not in c.normalized_interpretation
    assert c.has_flag(FlagCode.AMBIGUITY_NOTED_BY_CLAUDE)
    assert c.thresholds[0].value is None  # Claude cannot change thresholds
    assert result.verification_issues == []
    assert client.requests[0]["tools"] and client.requests[0]["model"]


def test_claude_refusal_falls_back_to_rule_based(calibrate_eligibility_text):
    client = FakeClaude([SimpleNamespace(stop_reason="refusal", content=[])])
    log = ActivityLog()
    result = ProtocolAgent(log, client=client).run("NCT02260934", calibrate_eligibility_text)
    assert result.mode == "offline (rule-based)"
    assert all(c.interpretation_source == "rule-based" for c in result.criteria)
    assert any(e.event == "warning" for e in log.entries)
