"""Patient Agent: answers questions about ONE synthetic patient record via tools.

It is the only component allowed to read patient data. Other agents ask it for
specific fields, and every answer:
  * returns the documented observations exactly as recorded (no merging or averaging)
  * gives each observation an evidence ID (e.g. "SYN-003/labs.upcr/0") for citation
  * says NOT_RECORDED when the record has nothing; it never fills a gap
  * says CONFLICTING when same-day observations disagree

Tools (also exposed as JSON schemas in PATIENT_TOOLS for LLM tool calling):
  list_patients, get_patient_overview, list_recorded_fields,
  get_field, get_medication_history
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from trialmatch import config
from trialmatch.activity_log import ActivityLog, summarize
from trialmatch.data_dictionary import DATA_FIELDS
from trialmatch.patient_models import PatientRecord, find_phi_keys

AGENT_NAME = "Patient Agent"

# Fields where several entries are expected (a medication list), so differing
# values on the same date are NOT a contradiction.
LIST_FIELDS = {"medications.history"}


class PatientDataError(ValueError):
    """Raised when a patient file is unsafe or malformed."""


def load_patient_file(path: Path) -> PatientRecord:
    """Load and validate one SYNTHETIC record; refuse anything that looks identifying."""
    data = json.loads(path.read_text(encoding="utf-8"))
    phi = find_phi_keys(data)
    if phi:
        raise PatientDataError(f"{path.name}: identifying keys are not allowed: {phi}")
    try:
        return PatientRecord.model_validate(data)
    except ValueError as exc:
        raise PatientDataError(f"{path.name}: {exc}") from exc


def load_patients(patients_dir: Path = config.PATIENT_DATA_DIR) -> Dict[str, PatientRecord]:
    records = [load_patient_file(p) for p in sorted(patients_dir.glob("SYN-*.json"))]
    return {r.patient_id: r for r in records}


class PatientAgent:
    def __init__(self, log: ActivityLog, patients: Optional[Dict[str, PatientRecord]] = None) -> None:
        self.log = log
        self.patients = patients if patients is not None else load_patients()
        self._handlers = {
            "list_patients": self.list_patients,
            "get_patient_overview": self.get_patient_overview,
            "list_recorded_fields": self.list_recorded_fields,
            "get_field": self.get_field,
            "get_medication_history": self.get_medication_history,
        }

    # ------------------------------------------------------------------ #
    # Dispatcher: every tool call goes through here and is logged
    # ------------------------------------------------------------------ #

    def call(self, tool: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        args = args or {}
        self.log.add(AGENT_NAME, "tool_call", summarize(args), tool=tool)
        if tool not in self._handlers:
            result: Dict[str, Any] = {"status": "ERROR", "error": f"Unknown tool '{tool}'"}
        else:
            try:
                result = self._handlers[tool](**args)
            except TypeError as exc:
                result = {"status": "ERROR", "error": f"Bad arguments: {exc}"}
        self.log.add(AGENT_NAME, "tool_result", _short(result), tool=tool)
        return result

    # ------------------------------------------------------------------ #
    # Tools
    # ------------------------------------------------------------------ #

    def list_patients(self) -> Dict[str, Any]:
        return {"status": "OK", "record_label": "SYNTHETIC", "patients": [
            {"patient_id": p.patient_id, "display_name": p.display_name, "scenario": p.scenario}
            for p in self.patients.values()
        ]}

    def get_patient_overview(self, patient_id: str) -> Dict[str, Any]:
        p = self._patient(patient_id)
        if p is None:
            return _unknown_patient(patient_id)
        key = ["demographics.age", "demographics.sex", "visits.screening_date",
               "visits.visit_minus_1_date", "visits.enrollment_date"]
        return {"status": "OK", "record_label": "SYNTHETIC", "patient_id": p.patient_id,
                "display_name": p.display_name, "scenario": p.scenario,
                "key_facts": {f: self.get_field(patient_id, f) for f in key}}

    def list_recorded_fields(self, patient_id: str) -> Dict[str, Any]:
        p = self._patient(patient_id)
        if p is None:
            return _unknown_patient(patient_id)
        recorded = sorted(f for f, obs in p.observations.items() if obs)
        return {"status": "OK", "record_label": "SYNTHETIC", "patient_id": patient_id,
                "recorded_fields": recorded,
                "not_recorded_fields": sorted(set(DATA_FIELDS) - set(recorded))}

    def get_field(self, patient_id: str, field: str) -> Dict[str, Any]:
        """Return every observation for one data-dictionary field."""
        p = self._patient(patient_id)
        if p is None:
            return _unknown_patient(patient_id)
        if field not in DATA_FIELDS:
            return {"status": "UNKNOWN_FIELD", "field": field,
                    "message": "Not a data-dictionary field. Use list_recorded_fields to see valid names."}
        observations = p.observations.get(field, [])
        if not observations:
            return {"status": "NOT_RECORDED", "record_label": "SYNTHETIC", "patient_id": patient_id,
                    "field": field, "observations": [],
                    "message": "The record contains no value for this field. Do not infer one."}
        evidence = [_evidence(patient_id, field, i, o) for i, o in enumerate(observations)]
        consistency, conflict_note = _consistency(field, evidence)
        result = {"status": "FOUND", "record_label": "SYNTHETIC", "patient_id": patient_id, "field": field,
                  "consistency": consistency, "observations": evidence}
        if conflict_note:
            result["message"] = conflict_note
        return result

    def get_medication_history(self, patient_id: str, drug: Optional[str] = None) -> Dict[str, Any]:
        """Medication entries, optionally filtered by drug name (case-insensitive substring).

        Negative statements such as "no prior biologic therapy" are ALWAYS
        returned alongside matches, so a reviewer can see when sources disagree.
        """
        base = self.get_field(patient_id, "medications.history")
        if base["status"] != "FOUND" or not drug:
            return base
        needle = drug.strip().lower()
        matches = [e for e in base["observations"] if needle in str(e["value"]).lower()]
        statements = [e for e in base["observations"]
                      if e["attributes"].get("kind") == "negative_statement" and e not in matches]
        result = {"status": "FOUND" if matches else "NOT_RECORDED", "record_label": "SYNTHETIC",
                  "patient_id": patient_id, "field": "medications.history", "drug_query": drug,
                  "observations": matches, "negative_statements": statements}
        if not matches:
            result["message"] = (f"No '{drug}' entry in the medication history. Absence from this record is "
                                 "not proof the drug was never given.")
        if matches and statements:
            result["consistency"] = "CONFLICTING"
            result["message"] = "A negative statement in the record conflicts with a documented administration."
        return result

    # ------------------------------------------------------------------ #

    def _patient(self, patient_id: str) -> Optional[PatientRecord]:
        return self.patients.get(patient_id)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #

def _evidence(patient_id: str, field: str, index: int, o: Any) -> Dict[str, Any]:
    return {
        "evidence_id": f"{patient_id}/{field}/{index}",
        "value": o.value,
        "unit": o.unit,
        "date": o.date.isoformat() if o.date else None,
        "end_date": o.end_date.isoformat() if o.end_date else None,
        "source": o.source,
        "note": o.note,
        "attributes": o.attributes,
    }


def _consistency(field: str, evidence: List[Dict[str, Any]]):
    if len(evidence) == 1:
        return "SINGLE", None
    if field in LIST_FIELDS:
        return "LIST", None
    by_date: Dict[Optional[str], set] = {}
    for e in evidence:
        by_date.setdefault(e["date"], set()).add(json.dumps([e["value"], e["unit"]], sort_keys=True, default=str))
    clashes = [d for d, values in by_date.items() if len(values) > 1]
    if clashes:
        return "CONFLICTING", (f"Observations dated {', '.join(str(d) for d in clashes)} disagree. "
                               "Do not pick one; a human must resolve the conflict.")
    return "MULTIPLE_DATES", None


def _unknown_patient(patient_id: str) -> Dict[str, Any]:
    return {"status": "UNKNOWN_PATIENT", "patient_id": patient_id, "message": "No synthetic record with this ID."}


def _short(result: Dict[str, Any]) -> str:
    """One-line summary of a tool result for the activity log."""
    parts = [str(result.get("status"))]
    if "field" in result:
        parts.append(result["field"])
    if "observations" in result:
        parts.append(f"{len(result['observations'])} observation(s)")
    if result.get("consistency") and result["consistency"] != "SINGLE":
        parts.append(result["consistency"])
    if "patients" in result:
        parts.append(f"{len(result['patients'])} patients")
    if "recorded_fields" in result:
        parts.append(f"{len(result['recorded_fields'])} recorded / {len(result['not_recorded_fields'])} not recorded")
    return " · ".join(parts)


# ---------------------------------------------------------------------- #
# JSON schemas for LLM tool calling (used by the Eligibility Agent later)
# ---------------------------------------------------------------------- #

_PID = {"type": "string", "description": "Synthetic patient ID, e.g. SYN-001"}

PATIENT_TOOLS = [
    {"name": "list_patients", "description": "List the SYNTHETIC patient records available.",
     "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
     "strict": True},
    {"name": "get_patient_overview",
     "description": "Age, sex, screening/visit -1/enrollment dates and scenario for one synthetic patient.",
     "input_schema": {"type": "object", "properties": {"patient_id": _PID}, "required": ["patient_id"],
                      "additionalProperties": False}, "strict": True},
    {"name": "list_recorded_fields",
     "description": "Which data-dictionary fields have at least one observation, and which are not recorded.",
     "input_schema": {"type": "object", "properties": {"patient_id": _PID}, "required": ["patient_id"],
                      "additionalProperties": False}, "strict": True},
    {"name": "get_field",
     "description": ("All observations for one data-dictionary field, with evidence IDs, dates and sources. "
                     "Returns NOT_RECORDED if absent and CONFLICTING if same-day values disagree."),
     "input_schema": {"type": "object", "properties": {
         "patient_id": _PID,
         "field": {"type": "string", "enum": sorted(DATA_FIELDS)}},
         "required": ["patient_id", "field"], "additionalProperties": False}, "strict": True},
    {"name": "get_medication_history",
     "description": ("Medication entries, optionally filtered by drug name. Negative statements "
                     "(e.g. 'no prior biologic therapy') are always included for comparison."),
     "input_schema": {"type": "object", "properties": {
         "patient_id": _PID,
         "drug": {"type": "string", "description": "Drug name or fragment; empty string for all"}},
         "required": ["patient_id", "drug"], "additionalProperties": False}, "strict": True},
]
