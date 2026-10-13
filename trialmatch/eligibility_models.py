"""Pydantic models for pre-screening results (Milestone 4).

There is deliberately NO overall "eligible" field. The app reports each
criterion separately and lists what needs follow-up; the decision belongs to
the study team.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class InclusionStatus(str, Enum):
    MET = "MET"
    NOT_MET = "NOT_MET"
    UNKNOWN = "UNKNOWN"
    NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"


class ExclusionStatus(str, Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    UNKNOWN = "UNKNOWN"
    NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"


class Method(str, Enum):
    DETERMINISTIC = "deterministic"      # numeric comparison or date arithmetic by code
    RECORD_LOOKUP = "record lookup"      # a documented yes/no or positive/negative answer
    COMBINED = "combined"                # result of combining sub-criteria
    NEEDS_PERSON = "needs a person"      # wording or evidence the app will not decide


class EvidenceItem(BaseModel):
    """One observation from the synthetic record that a conclusion relies on."""

    evidence_id: str
    field: str
    value: Any
    unit: Optional[str] = None
    date: Optional[str] = None
    end_date: Optional[str] = None
    source: Optional[str] = None
    note: Optional[str] = None


class AISuggestion(BaseModel):
    """Optional Claude suggestion for a criterion that needs a person. Advisory only."""

    suggested_status: str
    rationale: str
    evidence_ids: List[str] = Field(default_factory=list)


class CriterionEvaluation(BaseModel):
    criterion_id: str
    type: str = Field(..., description="INCLUSION or EXCLUSION")
    criterion_text: str
    status: str
    method: Method
    reason: str = Field(..., description="Plain-language explanation of the status")
    comparisons: List[str] = Field(default_factory=list, description="Each check performed, in words")
    evidence: List[EvidenceItem] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list, description="Data-dictionary fields")
    review_reasons: List[str] = Field(default_factory=list)
    parent_id: Optional[str] = None
    ai_suggestion: Optional[AISuggestion] = None

    @model_validator(mode="after")
    def _status_matches_type(self) -> "CriterionEvaluation":
        allowed = InclusionStatus if self.type == "INCLUSION" else ExclusionStatus
        if self.status not in {s.value for s in allowed}:
            raise ValueError(f"{self.status} is not a valid status for an {self.type} criterion")
        return self


class ScreeningResult(BaseModel):
    patient_id: str
    record_label: str = "SYNTHETIC"
    nct_id: str
    screened_at_utc: str
    mode: str
    evaluations: List[CriterionEvaluation]
    disclaimer: str = (
        "Educational simulation using a SYNTHETIC record. This is a pre-screening aid, not an eligibility "
        "decision. Every result must be confirmed by the study team."
    )

    def by_id(self, criterion_id: str) -> CriterionEvaluation:
        for e in self.evaluations:
            if e.criterion_id == criterion_id:
                return e
        raise KeyError(criterion_id)

    def counts(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {"INCLUSION": {}, "EXCLUSION": {}}
        for e in self.evaluations:
            out[e.type][e.status] = out[e.type].get(e.status, 0) + 1
        return out
