"""End-to-end pre-screening of the five SYNTHETIC patients against CALIBRATE."""

from types import SimpleNamespace

import pytest

from trialmatch.activity_log import ActivityLog
from trialmatch.agents.eligibility_agent import EligibilityAgent
from trialmatch.agents.patient_agent import PatientAgent, load_patients
from trialmatch.agents.protocol_agent import ProtocolAgent
from trialmatch.eligibility_models import CriterionEvaluation, ScreeningResult
from trialmatch.study_parser import parse_study


@pytest.fixture(scope="module")
def setup(calibrate_eligibility_text):
    import json
    from tests.conftest import RAW_DIR

    raw = sorted(p for p in RAW_DIR.glob("NCT02260934_*.json") if not p.name.endswith(".meta.json"))[-1]
    study = parse_study(json.loads(raw.read_bytes()))
    extraction = ProtocolAgent(ActivityLog()).run("NCT02260934", calibrate_eligibility_text)
    return study, extraction, load_patients()


@pytest.fixture(scope="module")
def results(setup):
    study, extraction, patients = setup
    log = ActivityLog()
    agent = EligibilityAgent(log, PatientAgent(log, patients))
    return {pid: agent.screen(pid, extraction, study) for pid in patients}


def status(results, pid, cid):
    return results[pid].by_id(cid).status


# --- General safety properties --------------------------------------------- #

def test_no_overall_eligibility_decision(results):
    assert "eligible" not in ScreeningResult.model_fields
    for r in results.values():
        assert r.record_label == "SYNTHETIC"
        assert "not an eligibility decision" in r.disclaimer


def test_every_criterion_is_evaluated(results, setup):
    _, extraction, _ = setup
    for r in results.values():
        ids = [e.criterion_id for e in r.evaluations]
        assert ids == ["REG-AGE"] + [c.criterion_id for c in extraction.criteria]


def test_decided_results_cite_real_evidence(results, setup):
    _, _, patients = setup
    for pid, r in results.items():
        for e in r.evaluations:
            if e.status in ("MET", "NOT_MET", "PRESENT", "ABSENT") and e.method.value != "combined":
                assert e.evidence, f"{pid} {e.criterion_id} has no evidence"
            for ev in e.evidence:
                p, field, index = ev.evidence_id.split("/")
                assert p == pid
                assert patients[pid].observations[field][int(index)].value == ev.value


def test_unknown_results_name_the_missing_information(results):
    for r in results.values():
        for e in r.evaluations:
            if e.status == "UNKNOWN" and e.method.value != "combined":
                assert e.missing_information, e.criterion_id


def test_status_must_match_criterion_type():
    with pytest.raises(ValueError):
        CriterionEvaluation(criterion_id="X", type="INCLUSION", criterion_text="x", status="PRESENT",
                            method="deterministic", reason="x")


def test_patient_data_comes_through_logged_tools(setup):
    study, extraction, patients = setup
    log = ActivityLog()
    EligibilityAgent(log, PatientAgent(log, patients)).screen("SYN-001", extraction, study)
    assert any(e.agent == "Patient Agent" and e.tool == "get_field" for e in log.entries)
    assert any(e.agent == "Eligibility Agent" and "No overall eligibility decision" in e.detail for e in log.entries)


# --- Protocol gaps always go to a person ------------------------------------ #

def test_missing_protocol_threshold_never_decided(results):
    for pid in results:
        assert status(results, pid, "EXC-04") == "NEEDS_HUMAN_REVIEW"


def test_missing_protocol_unit_never_decided(results):
    for pid in ["SYN-001", "SYN-002", "SYN-004", "SYN-005"]:
        e = results[pid].by_id("INC-04")
        assert e.status == "NEEDS_HUMAN_REVIEW"
        assert any("if the units match" in c for c in e.comparisons)


def test_investigator_judgement_always_needs_person(results):
    for pid in results:
        for cid in ("EXC-12", "EXC-27", "EXC-34"):
            assert status(results, pid, cid) == "NEEDS_HUMAN_REVIEW"


# --- SYN-001: complete record ----------------------------------------------- #

def test_syn001(results):
    for cid in ("REG-AGE", "INC-01", "INC-02", "INC-03", "INC-03.a", "INC-05"):
        assert status(results, "SYN-001", cid) == "MET", cid
    for cid in ("EXC-01", "EXC-02", "EXC-03", "EXC-05", "EXC-14", "EXC-21", "EXC-25", "EXC-26"):
        assert status(results, "SYN-001", cid) == "ABSENT", cid


# --- SYN-002: strict inequalities at the boundary --------------------------- #

def test_syn002_boundaries(results):
    r = results["SYN-002"]
    for cid in ("EXC-02", "EXC-03", "EXC-05", "EXC-14"):  # "below X" with value exactly X
        assert r.by_id(cid).status == "ABSENT", cid
        assert "exactly at the limit, which does not count" in r.by_id(cid).reason
    assert r.by_id("EXC-26").status == "PRESENT"  # ">= 2 x ULN" with exactly 2 x ULN
    assert "which counts" in r.by_id("EXC-26").reason
    assert r.by_id("INC-03.b.2").status == "NOT_MET"  # WBC exactly 5 is not > 5
    assert r.by_id("REG-AGE").status == "MET"  # age exactly 18


# --- SYN-003: missing data and date windows --------------------------------- #

def test_syn003_missing_data(results):
    r = results["SYN-003"]
    assert r.by_id("EXC-05").status == "UNKNOWN" and "labs.iga" in r.by_id("EXC-05").missing_information
    assert r.by_id("EXC-06").status == "UNKNOWN"
    assert r.by_id("EXC-09").status == "UNKNOWN"
    assert r.by_id("EXC-20").status == "UNKNOWN"
    assert r.by_id("INC-03.b.3").status == "UNKNOWN"


def test_syn003_spot_urine_not_accepted(results):
    e = results["SYN-003"].by_id("INC-04")
    assert e.status == "UNKNOWN"
    assert "24-hour" in e.reason


def test_syn003_date_windows(results):
    r = results["SYN-003"]
    assert r.by_id("INC-03.a").status == "NOT_MET"   # biopsy 138 days old: outside 3 months
    assert r.by_id("INC-03").status == "UNKNOWN"     # 12-month option depends on missing urine data
    assert r.by_id("INC-02").status == "UNKNOWN"     # ANA outside 14 days, anti-dsDNA not recorded
    assert r.by_id("EXC-21").status == "ABSENT"      # last cyclophosphamide 189 days ago: outside 6 months


# --- SYN-004: prior treatment and start-date sensitivity -------------------- #

def test_syn004_prior_treatment(results):
    r = results["SYN-004"]
    assert r.by_id("EXC-21").status == "PRESENT"     # cyclophosphamide within 6 months
    assert r.by_id("EXC-25").status == "PRESENT"     # rituximab in 2012 (any time)
    assert r.by_id("EXC-01").status == "ABSENT"      # previously treated: not new onset
    assert r.by_id("EXC-13").status == "ABSENT"      # live vaccine 116 days before enrollment
    assert r.by_id("EXC-28").status == "NEEDS_HUMAN_REVIEW"


def test_syn004_window_depends_on_start_date(results):
    e = results["SYN-004"].by_id("EXC-23")          # IVIG: 77 days before screening, 91 before enrollment
    assert e.status == "NEEDS_HUMAN_REVIEW"
    assert "depends on which date" in e.reason


# --- SYN-005: contradictory evidence ---------------------------------------- #

def test_syn005_conflicts_go_to_a_person(results):
    r = results["SYN-005"]
    for cid in ("EXC-02", "EXC-18", "EXC-25"):
        assert r.by_id(cid).status == "NEEDS_HUMAN_REVIEW", cid
        assert "Sources disagree" in r.by_id(cid).review_reasons
    assert len(r.by_id("EXC-02").evidence) == 2      # both ANC values are shown


def test_syn005_biologic_statement_does_not_affect_non_biologics(results):
    assert results["SYN-005"].by_id("EXC-01").status == "ABSENT"


# --- Optional Claude suggestions, with a fake client ------------------------ #

class FakeClaude:
    def __init__(self, responses):
        self.responses = list(responses)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: self.responses.pop(0)))


def _use(i, name, inp):
    return SimpleNamespace(type="tool_use", id=i, name=name, input=inp)


def test_claude_suggestion_must_cite_retrieved_evidence(setup):
    study, extraction, patients = setup
    client = FakeClaude([
        SimpleNamespace(stop_reason="tool_use", content=[_use("a", "submit_suggestion", {
            "criterion_id": "EXC-12", "suggested_status": "ABSENT", "rationale": "No infections.",
            "evidence_ids": ["SYN-001/infections.history/0"]})]),          # not retrieved yet -> rejected
        SimpleNamespace(stop_reason="tool_use", content=[_use("b", "get_field", {
            "patient_id": "SYN-001", "field": "infections.history"})]),
        SimpleNamespace(stop_reason="tool_use", content=[_use("c", "submit_suggestion", {
            "criterion_id": "EXC-12", "suggested_status": "ABSENT", "rationale": "No infections documented.",
            "evidence_ids": ["SYN-001/infections.history/0"]})]),
        SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="Done.")]),
    ])
    log = ActivityLog()
    r = EligibilityAgent(log, PatientAgent(log, patients), client=client).screen("SYN-001", extraction, study)
    e = r.by_id("EXC-12")
    assert e.status == "NEEDS_HUMAN_REVIEW"           # a suggestion never changes the status
    assert e.ai_suggestion is not None and e.ai_suggestion.suggested_status == "ABSENT"
    assert any("rejected" in x.detail for x in log.entries if x.tool == "submit_suggestion")


def test_claude_cannot_query_another_patient(setup):
    study, extraction, patients = setup
    client = FakeClaude([
        SimpleNamespace(stop_reason="tool_use", content=[_use("a", "get_field", {
            "patient_id": "SYN-002", "field": "labs.anc"})]),
        SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="Done.")]),
    ])
    log = ActivityLog()
    EligibilityAgent(log, PatientAgent(log, patients), client=client).screen("SYN-001", extraction, study)
    assert any("Only SYN-001" in x.detail for x in log.entries if x.event == "tool_result")
