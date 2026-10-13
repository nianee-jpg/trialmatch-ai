"""The pre-screening status and option explanations shown to the user."""

import json
import re

import pytest

from trialmatch.activity_log import ActivityLog
from trialmatch.agents.eligibility_agent import EligibilityAgent
from trialmatch.agents.patient_agent import PatientAgent, load_patients
from trialmatch.agents.protocol_agent import ProtocolAgent
from trialmatch.agents.verification_agent import VerificationAgent
from trialmatch.report import build_html_report
from trialmatch.screening_summary import follow_up, headline, option_label, part_context, status_counts
from trialmatch.study_parser import parse_study
from tests.conftest import RAW_DIR


@pytest.fixture(scope="module")
def env(calibrate_eligibility_text):
    raw = sorted(p for p in RAW_DIR.glob("NCT02260934_*.json") if not p.name.endswith(".meta.json"))[-1]
    study = parse_study(json.loads(raw.read_bytes()))
    log = ActivityLog()
    extraction = ProtocolAgent(log).run("NCT02260934", calibrate_eligibility_text)
    pa = PatientAgent(log, load_patients())
    results = {pid: EligibilityAgent(log, pa).screen(pid, extraction, study) for pid in pa.patients}
    return study, extraction, pa, results


@pytest.mark.parametrize("pid, level", [
    ("SYN-001", "review"),    # nothing rules them out, but 6 criteria need the study team
    ("SYN-002", "rule_out"),  # AST exactly 2 x ULN
    ("SYN-003", "review"),
    ("SYN-004", "rule_out"),  # recent cyclophosphamide, prior rituximab
    ("SYN-005", "review"),
])
def test_headline_level(env, pid, level):
    assert headline(env[3][pid]).level == level


def test_patient_a_is_shown_as_needing_review(env):
    h = headline(env[3]["SYN-001"])
    assert h.title == "Needs review by the study team"
    assert any("6 criteria need a decision" in d for d in h.details)


def test_headline_never_claims_eligibility(env):
    for r in env[3].values():
        h = headline(r)
        text = " ".join([h.title] + h.details)
        assert not re.search(r"\beligible\b", text, re.IGNORECASE)


def test_missing_information_is_listed(env):
    h = headline(env[3]["SYN-003"])
    assert any("information is missing" in d and "INC-04" in d for d in h.details)


def test_unneeded_option_is_explained(env):
    r = env[3]["SYN-001"]
    e = r.by_id("INC-03.b.2")
    assert e.status == "NOT_MET"                        # the option really is not met ...
    assert r.by_id("INC-03.b").status == "MET"          # ... but the criterion is met through other options
    assert option_label(r, e) == "option · not needed"
    assert part_context(r, e) == ("Option not needed: only one option is required, and INC-03.b is already "
                                  "met through INC-03.b.1 and INC-03.b.3.")
    assert option_label(r, r.by_id("INC-03.b.1")) == "option · counts"


def test_unneeded_option_is_not_counted_against_the_patient(env):
    r = env[3]["SYN-001"]
    assert status_counts(r)["INCLUSION"].get("NOT_MET", 0) == 0
    assert not follow_up(r).rule_out


def test_options_when_criterion_not_met(env):
    r = env[3]["SYN-003"]
    assert option_label(r, r.by_id("INC-03.a")) == "option"
    assert part_context(r, r.by_id("INC-03.a")) == (
        "INC-03 cannot be decided yet: no option is met so far, and some options are missing information. "
        "This option is not met.")


def test_main_criteria_have_no_option_context(env):
    r = env[3]["SYN-001"]
    assert part_context(r, r.by_id("INC-03")) is None and option_label(r, r.by_id("INC-03")) is None


def test_report_shows_status_and_options(env):
    study, extraction, pa, results = env
    r = results["SYN-001"]
    html = build_html_report(study=study, source_url="u", retrieved_at="t", raw_file="f", raw_sha256="s",
                             extraction=extraction, record=pa.patients["SYN-001"], screening=r,
                             verification=VerificationAgent(ActivityLog(), pa).verify(r, extraction, study))
    assert "Pre-screening status: Needs review by the study team" in html
    assert "option · not needed" in html
    assert "already met through INC-03.b.1 and INC-03.b.3" in html


@pytest.mark.parametrize("pid", ["SYN-001", "SYN-002", "SYN-003", "SYN-004", "SYN-005"])
def test_every_patient_display_is_consistent(env, pid):
    """For every patient: options agree with their criterion, every option explains itself,
    and every main criterion that needs action is named in the headline."""
    r = env[3][pid]
    listed = " ".join(headline(r).details)
    for e in r.evaluations:
        if e.parent_id is None:
            if e.status in ("NOT_MET", "PRESENT", "UNKNOWN", "NEEDS_HUMAN_REVIEW"):
                assert e.criterion_id in listed, f"{pid}: {e.criterion_id} missing from the headline"
            continue
        assert part_context(r, e), f"{pid}: {e.criterion_id} has no explanation"
        parent = r.by_id(e.parent_id)
        options = [k for k in r.evaluations if k.parent_id == parent.criterion_id]
        if parent.status == "MET":
            assert any(k.status == "MET" for k in options), f"{pid}: {parent.criterion_id} met without a met option"
            if e.status != "MET":
                assert option_label(r, e) == "option · not needed"
        if all(k.status == "NOT_MET" for k in options):
            assert parent.status == "NOT_MET"
