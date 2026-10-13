"""Pydantic models for structured eligibility criteria (Milestone 2).

Each Criterion keeps three versions of its wording:
  * original_text            exact characters from ClinicalTrials.gov (Markdown escapes included)
  * display_text             the same text with Markdown escapes removed, for reading
  * normalized_interpretation a plain statement of what would need to be checked

Only `original_text` is authoritative. The other two are derived from it and
are checked against it by `verify_extraction`.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class CriterionType(str, Enum):
    INCLUSION = "INCLUSION"
    EXCLUSION = "EXCLUSION"


class Comparator(str, Enum):
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="

    @property
    def is_strict(self) -> bool:
        return self in (Comparator.LT, Comparator.GT)


class Logic(str, Enum):
    """How a parent criterion combines its sub-criteria."""

    ANY_OF = "ANY_OF"


class FlagCode(str, Enum):
    MISSING_THRESHOLD = "MISSING_THRESHOLD"          # comparator with no number, e.g. "Hgb \< mg/dL"
    UNIT_NOT_STATED = "UNIT_NOT_STATED"              # number with no unit, e.g. "UPCR \>1"
    RELATIVE_THRESHOLD = "RELATIVE_THRESHOLD"        # e.g. ">=2 times the upper limit of normal"
    EXTERNAL_INFO_NEEDED = "EXTERNAL_INFO_NEEDED"    # e.g. drug half-lives
    TIME_ANCHOR_NOT_STATED = "TIME_ANCHOR_NOT_STATED"  # "within the past 60 days" of what?
    INVESTIGATOR_JUDGEMENT = "INVESTIGATOR_JUDGEMENT"  # "in the investigator's opinion"
    SUBJECTIVE_TERM = "SUBJECTIVE_TERM"              # "significant", "severe", "consistent with"
    NON_EXHAUSTIVE_LIST = "NON_EXHAUSTIVE_LIST"      # "including but not limited to"
    STRUCTURE_INFERRED = "STRUCTURE_INFERRED"        # nesting decided by a rule, not by indentation
    THRESHOLD_PARAMETER_UNKNOWN = "THRESHOLD_PARAMETER_UNKNOWN"
    REQUIRED_DATA_UNMAPPED = "REQUIRED_DATA_UNMAPPED"
    AMBIGUITY_NOTED_BY_CLAUDE = "AMBIGUITY_NOTED_BY_CLAUDE"  # optional Claude mode only


class Flag(BaseModel):
    code: FlagCode
    message: str


class Threshold(BaseModel):
    """A numeric comparison found in the criterion text. Never filled in by guessing."""

    parameter: Optional[str] = Field(None, description="Data-dictionary field, e.g. 'labs.anc'")
    comparator: Comparator
    value: Optional[float] = Field(None, description="None when the source text has no number")
    unit: Optional[str] = Field(None, description="Unit as written, or None if not stated")
    relative_to: Optional[str] = Field(None, description="e.g. 'upper limit of normal'")
    source_text: str = Field(..., description="Exact span of original_text this came from")

    @property
    def complete(self) -> bool:
        return self.value is not None and self.parameter is not None


class TimeWindow(BaseModel):
    amount: int
    unit: str = Field(..., description="days, weeks, months or years")
    anchor: Optional[str] = Field(None, description="Reference event, e.g. 'visit -1'; None if not stated")
    source_text: str


class RequiredData(BaseModel):
    field: str = Field(..., description="Data-dictionary field name, e.g. 'labs.platelets'")
    detail: Optional[str] = None


class SourceReference(BaseModel):
    nct_id: str
    json_path: str = "protocolSection.eligibilityModule.eligibilityCriteria"
    char_start: int = Field(..., description="Offset of original_text within the eligibility text")
    char_end: int
    line_number: int = Field(..., description="1-based line in the eligibility text")
    raw_file: Optional[str] = None
    raw_sha256: Optional[str] = None


class Criterion(BaseModel):
    criterion_id: str = Field(..., description="e.g. INC-03.b.1 or EXC-14")
    type: CriterionType
    parent_id: Optional[str] = None
    logic: Optional[Logic] = Field(None, description="Set when this criterion groups sub-criteria")
    original_text: str
    display_text: str
    normalized_interpretation: str
    interpretation_source: str = Field("rule-based", description="'rule-based' or 'claude'")
    thresholds: List[Threshold] = Field(default_factory=list)
    time_windows: List[TimeWindow] = Field(default_factory=list)
    required_data: List[RequiredData] = Field(default_factory=list)
    flags: List[Flag] = Field(default_factory=list)
    source: SourceReference

    def has_flag(self, code: FlagCode) -> bool:
        return any(f.code == code for f in self.flags)


class VerificationIssue(BaseModel):
    criterion_id: Optional[str]
    check: str
    message: str


class ProtocolExtraction(BaseModel):
    nct_id: str
    mode: str = Field(..., description="'offline (rule-based)' or 'claude'")
    model: Optional[str] = None
    criteria: List[Criterion]
    verification_issues: List[VerificationIssue] = Field(default_factory=list)

    @property
    def passed_verification(self) -> bool:
        return not self.verification_issues

    def by_id(self, criterion_id: str) -> Criterion:
        for c in self.criteria:
            if c.criterion_id == criterion_id:
                return c
        raise KeyError(criterion_id)
