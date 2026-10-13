"""Tests for SYNTHETIC patient records and the Patient Agent."""

import datetime as dt
import json

import pytest

from trialmatch.activity_log import ActivityLog
from trialmatch.agents.patient_agent import (
    PATIENT_TOOLS,
    PatientAgent,
    PatientDataError,
    load_patient_file,
    load_patients,
)
from trialmatch.data_dictionary import DATA_FIELDS


@pytest.fixture(scope="module")
def patients():
    return load_patients()


@pytest.fixture
def agent(patients):
    return PatientAgent(ActivityLog(), patients)


def value(agent, pid, field, i=0):
    return agent.get_field(pid, field)["observations"][i]["value"]


def date(agent, pid, field, key="date", i=0):
    return dt.date.fromisoformat(agent.get_field(pid, field)["observations"][i][key])


# --- Records are synthetic, valid and identifier-free ---------------------- #

def test_five_synthetic_patients(patients):
    assert list(patients) == ["SYN-001", "SYN-002", "SYN-003", "SYN-004", "SYN-005"]
    for p in patients.values():
        assert p.synthetic is True and p.record_label == "SYNTHETIC"
        assert p.display_name.startswith("Synthetic Patient")


def _write(tmp_path, record):
    path = tmp_path / "SYN-999.json"
    path.write_text(json.dumps(record))
    return path


def _minimal(**overrides):
    record = {"record_label": "SYNTHETIC", "synthetic": True, "patient_id": "SYN-999",
              "display_name": "Synthetic Patient Z", "scenario": "test", "observations": {}}
    record.update(overrides)
    return record


def test_minimal_record_loads(tmp_path):
    assert load_patient_file(_write(tmp_path, _minimal())).patient_id == "SYN-999"


@pytest.mark.parametrize("overrides", [
    {"synthetic": False},
    {"record_label": "REAL"},
    {"patient_id": "12345"},
    {"display_name": "Jane Doe"},
    {"observations": {"labs.weight": [{"value": 70, "source": "x"}]}},  # not in the data dictionary
])
def test_unsafe_or_malformed_records_are_rejected(tmp_path, overrides):
    with pytest.raises(PatientDataError):
        load_patient_file(_write(tmp_path, _minimal(**overrides)))


@pytest.mark.parametrize("phi_key", ["name", "mrn", "date_of_birth", "address"])
def test_identifying_keys_are_rejected_anywhere(tmp_path, phi_key):
    record = _minimal(observations={"labs.anc": [{"value": 2000, "source": "x", "attributes": {phi_key: "x"}}]})
    with pytest.raises(PatientDataError, match="identifying"):
        load_patient_file(_write(tmp_path, record))


# --- Patient Agent tools --------------------------------------------------- #

def test_get_field_returns_exact_value_with_evidence_id(agent):
    result = agent.get_field("SYN-002", "labs.anc")
    assert result["status"] == "FOUND" and result["consistency"] == "SINGLE"
    obs = result["observations"][0]
    assert obs["value"] == 1500 and obs["unit"] == "/mm^3"
    assert obs["evidence_id"] == "SYN-002/labs.anc/0"
    assert obs["source"]


def test_missing_data_is_not_recorded_never_inferred(agent):
    for field in ["labs.hemoglobin", "labs.iga", "infections.quantiferon_tb", "labs.anti_dsdna",
                  "urinalysis.wbc_per_hpf", "reproductive.menses_at_collection"]:
        result = agent.get_field("SYN-003", field)
        assert result["status"] == "NOT_RECORDED", field
        assert result["observations"] == []


def test_investigator_judgement_is_never_in_a_record(agent, patients):
    for pid in patients:
        assert agent.get_field(pid, "assessment.investigator_judgement")["status"] == "NOT_RECORDED"


def test_unknown_field_and_patient(agent):
    assert agent.get_field("SYN-001", "labs.weight")["status"] == "UNKNOWN_FIELD"
    assert agent.get_field("SYN-404", "labs.anc")["status"] == "UNKNOWN_PATIENT"


def test_same_day_contradictions_are_flagged(agent):
    for field in ["labs.anc", "reproductive.pregnancy_status"]:
        result = agent.get_field("SYN-005", field)
        assert result["consistency"] == "CONFLICTING", field
        assert len(result["observations"]) == 2


def test_medication_list_is_not_a_contradiction(agent):
    assert agent.get_field("SYN-004", "medications.history")["consistency"] == "LIST"


def test_medication_search_surfaces_conflicting_statement(agent):
    result = agent.get_medication_history("SYN-005", "belimumab")
    assert result["status"] == "FOUND" and result["consistency"] == "CONFLICTING"
    assert result["negative_statements"][0]["value"] == "no prior biologic therapy"


def test_medication_absence_is_not_proof(agent):
    result = agent.get_medication_history("SYN-001", "rituximab")
    assert result["status"] == "NOT_RECORDED"
    assert "not proof" in result["message"]


def test_list_recorded_fields_partitions_dictionary(agent):
    result = agent.list_recorded_fields("SYN-003")
    assert set(result["recorded_fields"]) | set(result["not_recorded_fields"]) == set(DATA_FIELDS)
    assert "labs.hemoglobin" in result["not_recorded_fields"]


def test_tool_calls_are_logged(patients):
    log = ActivityLog()
    a = PatientAgent(log, patients)
    a.call("get_field", {"patient_id": "SYN-001", "field": "labs.egfr"})
    a.call("no_such_tool", {})
    assert [(e.event, e.tool) for e in log.entries] == [
        ("tool_call", "get_field"), ("tool_result", "get_field"),
        ("tool_call", "no_such_tool"), ("tool_result", "no_such_tool")]
    assert "ERROR" in log.entries[-1].detail


def test_tool_schemas_match_handlers(agent):
    assert {t["name"] for t in PATIENT_TOOLS} == set(agent._handlers)
    for t in PATIENT_TOOLS:
        assert t["input_schema"]["additionalProperties"] is False


# --- Each synthetic scenario really contains what it claims ---------------- #

def test_syn002_boundary_values(agent):
    assert value(agent, "SYN-002", "demographics.age") == 18
    assert value(agent, "SYN-002", "labs.platelets") == 50000
    assert value(agent, "SYN-002", "labs.egfr") == 20
    assert value(agent, "SYN-002", "labs.upcr") == 1.0
    ast = agent.get_field("SYN-002", "labs.ast")["observations"][0]
    assert ast["value"] == 2 * ast["attributes"]["uln"]


def test_syn003_date_windows_and_collection_method(agent):
    visit = date(agent, "SYN-003", "labs.ana")
    v1 = dt.date.fromisoformat(value(agent, "SYN-003", "visits.visit_minus_1_date"))
    assert (v1 - visit).days == 20  # outside the 14-day serology window
    screening = dt.date.fromisoformat(value(agent, "SYN-003", "visits.screening_date"))
    assert (screening - date(agent, "SYN-003", "renal.biopsy_class")).days == 138
    assert "spot" in value(agent, "SYN-003", "labs.upcr_collection_method")
    cyc = agent.get_medication_history("SYN-003", "cyclophosphamide")["observations"][0]
    assert (screening - dt.date.fromisoformat(cyc["end_date"])).days == 189


def test_syn004_prior_treatment_and_windows(agent):
    screening = dt.date.fromisoformat(value(agent, "SYN-004", "visits.screening_date"))
    enrollment = dt.date.fromisoformat(value(agent, "SYN-004", "visits.enrollment_date"))
    cyc = agent.get_medication_history("SYN-004", "cyclophosphamide")["observations"][0]
    assert (screening - dt.date.fromisoformat(cyc["end_date"])).days == 143
    assert agent.get_medication_history("SYN-004", "rituximab")["status"] == "FOUND"
    assert (screening - date(agent, "SYN-004", "procedures.ivig_plasmapheresis_leukopheresis")).days == 77
    assert (enrollment - date(agent, "SYN-004", "vaccinations.live_attenuated")).days == 116


def test_every_field_has_a_plain_language_label():
    from trialmatch.data_dictionary import FIELD_GROUPS, FIELD_LABELS

    assert set(FIELD_LABELS) == set(DATA_FIELDS)
    assert {f.split(".")[0] for f in DATA_FIELDS} <= set(FIELD_GROUPS)
