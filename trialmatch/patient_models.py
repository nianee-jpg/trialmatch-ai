"""Pydantic models for SYNTHETIC patient records (Milestone 3).

Every record is fictional. The model refuses to load a file unless it is
explicitly labelled SYNTHETIC, uses an ID like SYN-001, and contains no
identifying keys (name, MRN, address, date of birth, ...).

Record layout: each data-dictionary field holds a LIST of observations, because
real charts often contain several values for one thing (repeat labs, a lab
result and a clinic note that disagree). Nothing is merged or averaged here.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from trialmatch.data_dictionary import DATA_FIELDS

SYNTHETIC_LABEL = "SYNTHETIC"

# Keys that could hold identifying information. A synthetic record never needs them.
PHI_KEYS = {
    "name", "first_name", "last_name", "full_name", "mrn", "medical_record_number", "ssn",
    "address", "street", "zip", "postcode", "phone", "email", "date_of_birth", "dob", "birth_date",
}


class Observation(BaseModel):
    """One piece of evidence in the record, with where it came from."""

    model_config = ConfigDict(extra="forbid")

    value: Any = Field(..., description="As documented: number, text, true/false, or a small object")
    unit: Optional[str] = None
    date: Optional[dt.date] = Field(None, description="Collection/administration/start date")
    end_date: Optional[dt.date] = Field(None, description="End date for treatments, if any")
    source: str = Field(..., description="Fictional source document, e.g. 'Central lab report'")
    note: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict, description="e.g. route, upper limit of normal")


class PatientRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_label: Literal["SYNTHETIC"]
    synthetic: Literal[True]
    patient_id: str = Field(..., pattern=r"^SYN-\d{3}$")
    display_name: str = Field(..., pattern=r"^Synthetic Patient [A-Z]$")
    scenario: str = Field(..., description="What this fictional case is designed to test")
    observations: Dict[str, List[Observation]]

    @field_validator("observations")
    @classmethod
    def _known_fields_only(cls, value: Dict[str, List[Observation]]) -> Dict[str, List[Observation]]:
        unknown = sorted(set(value) - set(DATA_FIELDS))
        if unknown:
            raise ValueError(f"Fields not in the data dictionary: {unknown}")
        return value


def find_phi_keys(data: Any, path: str = "") -> List[str]:
    """Return paths of any identifying keys found anywhere in a JSON object."""
    hits: List[str] = []
    if isinstance(data, dict):
        for key, child in data.items():
            here = f"{path}.{key}" if path else key
            if key.lower() in PHI_KEYS:
                hits.append(here)
            hits += find_phi_keys(child, here)
    elif isinstance(data, list):
        for i, child in enumerate(data):
            hits += find_phi_keys(child, f"{path}[{i}]")
    return hits
