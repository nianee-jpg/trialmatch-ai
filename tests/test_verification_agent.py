"""The Verification Agent passes honest results and catches every kind of tampering."""

import json

import pytest

from trialmatch.activity_log import ActivityLog
from trialmatch.agents.eligibility_agent import EligibilityAgent
from trialmatch.agents.patient_agent import PatientAgent, load_patients
from trialmatch.agents.protocol_agent import ProtocolAgent
from trialmatch.agents.verification_agent import VerificationAgent
from trialmatch.eligibility_models import Method
from trialmatch.report import build_html_report
from trialmatch.study_parser import parse_study
from trialmatch.verification_models import Outcome, Severity
from tests.conftest import RAW_DIR


@pytest.fixture(scope="module")
def env(calibrate_eligibility_text):
    raw = sorted(p for p in RAW_DIR.glob("NCT02260934_*.json") if not p.name.endswith(".meta.json"))[-1]
    study = parse_study(json.loads(raw.read_bytes()))
    log = ActivityLog()
    extraction = ProtocolAgent(log).run("NCT02260934", calibrate_eligibility_text)
    patient_agent = PatientAgent(log, load_patients())
    return study, extraction, patient_agent, log


def screen(env, pid):
    study, extraction, pa, log = env
    return EligibilityAgent(log, pa).screen(pid, extraction, study)


def verify(env, result):
    study, extraction, pa, log = env
    return VerificationAgent(log, pa).verify(result, extraction, study)


@pytest.mark.parametrize("pid", ["SYN-001", "SYN-002", "SYN-003", "SYN-004", "SYN-005"])
def test_honest_results_pass(env, pid):
    report = verify(env, screen(env, pid))
    assert report.passed
    failures = [(r.criterion_id, c.message) for r in report.results for c in r.checks if c.severity == Severity.ERROR]
    assert failures == []


def test_protocol_gaps_are_noted(env):
    report = verify(env, screen(env, "SYN-001"))
    assert any(c.check == "missing_threshold" and c.severity == Severity.NOTE for c in report.by_id("EXC-04").checks)


def test_conflicts_are_noted(env):
    report = verify(env, screen(env, "SYN-005"))
    assert any(c.check == "contradiction" for c in report.by_id("EXC-02").checks)


def test_assumptions_are_surfaced(env):
    report = verify(env, screen(env, "SYN-001"))
    assert any(c.check == "assumption" for c in report.by_id("EXC-21").checks)


TAMPERING = [
    ("SYN-001", "EXC-02", lambda e: setattr(e, "reason", e.reason + " The limit is 2,000."), "numbers_supported"),
    ("SYN-001", "EXC-02", lambda e: setattr(e, "status", "PRESENT"), "recomputed"),
    ("SYN-001", "EXC-02", lambda e: setattr(e.evidence[0], "value", 900), "evidence_matches"),
    ("SYN-001", "EXC-02", lambda e: setattr(e, "evidence", []), "evidence_present"),
    ("SYN-001", "EXC-04", lambda e: (setattr(e, "status", "ABSENT"), setattr(e, "method", Method.DETERMINISTIC)),
     "missing_threshold"),
    ("SYN-005", "EXC-18", lambda e: setattr(e, "status", "ABSENT"), "contradiction"),
    ("SYN-003", "EXC-06", lambda e: setattr(e, "missing_information", ["labs.anc"]), "missing_confirmed"),
    ("SYN-001", "INC-01", lambda e: setattr(e, "reason", "The patient is eligible."), "decision_language"),
    ("SYN-001", "EXC-21", lambda e: setattr(e, "reason", e.reason + " Last dose 2014-12-01."), "dates_supported"),
    ("SYN-003", "INC-03", lambda e: setattr(e, "status", "MET"), "combination"),
    ("SYN-001", "INC-02", lambda e: setattr(e, "criterion_text", "Positive ANA."), "criterion_text"),
    ("SYN-001", "EXC-25", lambda e: setattr(e.evidence[0], "evidence_id", "SYN-001/medications.history/99"),
     "evidence_matches"),
]


@pytest.mark.parametrize("pid, cid, tamper, check", TAMPERING, ids=[t[3] + ":" + t[1] for t in TAMPERING])
def test_tampering_is_caught(env, pid, cid, tamper, check):
    result = screen(env, pid)
    tamper(result.by_id(cid))
    report = verify(env, result)
    v = report.by_id(cid)
    assert v.outcome == Outcome.FAILED
    assert check in {c.check for c in v.problems}
    assert not report.passed


def test_missing_result_is_caught(env):
    result = screen(env, "SYN-001")
    result.evaluations = result.evaluations[:-1]
    report = verify(env, result)
    assert any(g.check == "coverage" for g in report.general) and not report.passed


def test_outcomes_are_logged(env):
    study, extraction, pa, _ = env
    log = ActivityLog()
    VerificationAgent(log, pa).verify(screen(env, "SYN-002"), extraction, study)
    lines = [e.detail for e in log.entries if e.agent == "Verification Agent" and e.event == "verification"]
    assert len(lines) == 47  # 46 results + summary
    assert lines[-1].startswith("Summary for SYN-002: 46 verified")


# --- Report ---------------------------------------------------------------- #

def test_report_contents(env):
    study, extraction, pa, log = env
    result = screen(env, "SYN-004")
    report = verify(env, result)
    html = build_html_report(study=study, source_url="https://clinicaltrials.gov/api/v2/studies/NCT02260934",
                             retrieved_at="2026-10-13T16:46:16Z", raw_file="data/raw/x.json", raw_sha256="abc123",
                             extraction=extraction, record=pa.patients["SYN-004"], screening=result,
                             verification=report, log=log)
    assert html.count("SYNTHETIC RECORD") == 2                 # banner at top and bottom
    assert "not an eligibility decision" in html
    assert "abc123" in html and "NCT02260934" in html
    for e in result.evaluations:                               # every criterion and its evidence appear
        assert f"<td>{e.criterion_id}</td>" in html
        for x in e.evidence:
            assert x.evidence_id in html
    assert "EXC-23" in html and "depends on" in html


def test_report_escapes_text(env):
    study, extraction, pa, log = env
    result = screen(env, "SYN-001")
    result.by_id("INC-01").reason = "<script>alert(1)</script>"
    html = build_html_report(study=study, source_url="u", retrieved_at="t", raw_file="f", raw_sha256="s",
                             extraction=extraction, record=pa.patients["SYN-001"], screening=result,
                             verification=verify(env, result))
    assert "<script>" not in html and "&lt;script&gt;" in html
