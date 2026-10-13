"""Pydantic models for the Verification Agent's findings (Milestone 5)."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel


class Severity(str, Enum):
    ERROR = "error"      # the conclusion is not supported; do not rely on it
    WARNING = "warning"  # supported, but something deserves a second look
    NOTE = "note"        # information for the reviewer (assumptions, protocol gaps handled)


class Outcome(str, Enum):
    VERIFIED = "VERIFIED"
    VERIFIED_WITH_WARNINGS = "VERIFIED_WITH_WARNINGS"
    FAILED = "FAILED"


class VerificationCheck(BaseModel):
    criterion_id: Optional[str]
    check: str
    severity: Severity
    message: str


class CriterionVerification(BaseModel):
    criterion_id: str
    outcome: Outcome
    checks: List[VerificationCheck]

    @property
    def problems(self) -> List[VerificationCheck]:
        return [c for c in self.checks if c.severity != Severity.NOTE]


class VerificationReport(BaseModel):
    patient_id: str
    record_label: str = "SYNTHETIC"
    verified_at_utc: str
    results: List[CriterionVerification]
    general: List[VerificationCheck]

    @property
    def passed(self) -> bool:
        return all(r.outcome != Outcome.FAILED for r in self.results) and not any(
            g.severity == Severity.ERROR for g in self.general)

    def by_id(self, criterion_id: str) -> CriterionVerification:
        for r in self.results:
            if r.criterion_id == criterion_id:
                return r
        raise KeyError(criterion_id)

    def counts(self) -> Dict[str, int]:
        out = {o.value: 0 for o in Outcome}
        for r in self.results:
            out[r.outcome.value] += 1
        return out
