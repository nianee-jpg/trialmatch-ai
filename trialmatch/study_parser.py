"""Parse a ClinicalTrials.gov API v2 study record into Pydantic models.

Design rule: we NEVER rewrite or paraphrase protocol text here. Every string is
copied exactly as it appears in the API response, and each field records the
JSON path it came from so later milestones can cite their sources.

Structured, criterion-by-criterion extraction is the Protocol Agent's job
(Milestone 2). This module only separates the original eligibility text into
its "Inclusion" and "Exclusion" sections, verbatim, for display.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


class Intervention(BaseModel):
    type: str | None = None
    name: str | None = None
    description: str | None = None


class ArmGroup(BaseModel):
    label: str | None = None
    type: str | None = None
    description: str | None = None
    intervention_names: list[str] = Field(default_factory=list)


class Outcome(BaseModel):
    measure: str | None = None
    time_frame: str | None = None
    description: str | None = None


class EligibilitySections(BaseModel):
    """The original eligibility text split into its two headed sections.

    Both sections are exact substrings of `full_text`. If the headings cannot be
    found, the sections are left as None and only `full_text` should be used.
    """

    inclusion_text: str | None = None
    exclusion_text: str | None = None
    split_succeeded: bool = False


class Eligibility(BaseModel):
    full_text: str | None = Field(
        default=None,
        description="Complete, unmodified text from "
        "protocolSection.eligibilityModule.eligibilityCriteria",
    )
    sections: EligibilitySections = Field(default_factory=EligibilitySections)
    sex: str | None = None
    minimum_age: str | None = None
    maximum_age: str | None = None
    healthy_volunteers: bool | None = None
    std_ages: list[str] = Field(default_factory=list)
    source_path: str = "protocolSection.eligibilityModule"


class StudyMetadata(BaseModel):
    """Human-readable summary of one study record."""

    nct_id: str
    brief_title: str | None = None
    official_title: str | None = None
    acronym: str | None = None
    overall_status: str | None = None
    start_date: str | None = None
    primary_completion_date: str | None = None
    completion_date: str | None = None
    last_update_posted: str | None = None
    lead_sponsor: str | None = None
    collaborators: list[str] = Field(default_factory=list)
    study_type: str | None = None
    phases: list[str] = Field(default_factory=list)
    allocation: str | None = None
    intervention_model: str | None = None
    primary_purpose: str | None = None
    masking: str | None = None
    enrollment_count: int | None = None
    enrollment_type: str | None = None
    conditions: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    brief_summary: str | None = None
    detailed_description: str | None = None
    arm_groups: list[ArmGroup] = Field(default_factory=list)
    interventions: list[Intervention] = Field(default_factory=list)
    primary_outcomes: list[Outcome] = Field(default_factory=list)
    secondary_outcomes: list[Outcome] = Field(default_factory=list)
    eligibility: Eligibility = Field(default_factory=Eligibility)
    has_results: bool | None = None


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def _get(data: Any, *path: str) -> Any:
    """Safely walk nested dicts: _get(d, "a", "b") == d["a"]["b"] or None."""
    for key in path:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def parse_study(data: dict[str, Any]) -> StudyMetadata:
    """Convert a raw API v2 study dict into a StudyMetadata object."""
    ps = data.get("protocolSection", {})
    ident = ps.get("identificationModule", {})
    status = ps.get("statusModule", {})
    sponsors = ps.get("sponsorCollaboratorsModule", {})
    design = ps.get("designModule", {})
    desc = ps.get("descriptionModule", {})
    conds = ps.get("conditionsModule", {})
    arms = ps.get("armsInterventionsModule", {})
    outcomes = ps.get("outcomesModule", {})
    elig = ps.get("eligibilityModule", {})

    full_text = elig.get("eligibilityCriteria")

    return StudyMetadata(
        nct_id=ident.get("nctId", ""),
        brief_title=ident.get("briefTitle"),
        official_title=ident.get("officialTitle"),
        acronym=ident.get("acronym"),
        overall_status=status.get("overallStatus"),
        start_date=_get(status, "startDateStruct", "date"),
        primary_completion_date=_get(status, "primaryCompletionDateStruct", "date"),
        completion_date=_get(status, "completionDateStruct", "date"),
        last_update_posted=_get(status, "lastUpdatePostDateStruct", "date"),
        lead_sponsor=_get(sponsors, "leadSponsor", "name"),
        collaborators=[c.get("name") for c in sponsors.get("collaborators", []) if c.get("name")],
        study_type=design.get("studyType"),
        phases=design.get("phases", []),
        allocation=_get(design, "designInfo", "allocation"),
        intervention_model=_get(design, "designInfo", "interventionModel"),
        primary_purpose=_get(design, "designInfo", "primaryPurpose"),
        masking=_get(design, "designInfo", "maskingInfo", "masking"),
        enrollment_count=_get(design, "enrollmentInfo", "count"),
        enrollment_type=_get(design, "enrollmentInfo", "type"),
        conditions=conds.get("conditions", []),
        keywords=conds.get("keywords", []),
        brief_summary=desc.get("briefSummary"),
        detailed_description=desc.get("detailedDescription"),
        arm_groups=[
            ArmGroup(
                label=a.get("label"),
                type=a.get("type"),
                description=a.get("description"),
                intervention_names=a.get("interventionNames", []),
            )
            for a in arms.get("armGroups", [])
        ],
        interventions=[
            Intervention(type=i.get("type"), name=i.get("name"), description=i.get("description"))
            for i in arms.get("interventions", [])
        ],
        primary_outcomes=[_outcome(o) for o in outcomes.get("primaryOutcomes", [])],
        secondary_outcomes=[_outcome(o) for o in outcomes.get("secondaryOutcomes", [])],
        eligibility=Eligibility(
            full_text=full_text,
            sections=split_eligibility_sections(full_text),
            sex=elig.get("sex"),
            minimum_age=elig.get("minimumAge"),
            maximum_age=elig.get("maximumAge"),
            healthy_volunteers=elig.get("healthyVolunteers"),
            std_ages=elig.get("stdAges", []),
        ),
        has_results=data.get("hasResults"),
    )


def _outcome(o: dict[str, Any]) -> Outcome:
    return Outcome(measure=o.get("measure"), time_frame=o.get("timeFrame"), description=o.get("description"))


# Headings as they typically appear, e.g. "Inclusion Criteria:" / "Exclusion Criteria:".
# Matched case-insensitively at the start of a line.
_INCLUSION_HEADING = re.compile(r"^[ \t]*inclusion criteria\s*:?", re.IGNORECASE | re.MULTILINE)
_EXCLUSION_HEADING = re.compile(r"^[ \t]*exclusion criteria\s*:?", re.IGNORECASE | re.MULTILINE)


def split_eligibility_sections(full_text: str | None) -> EligibilitySections:
    """Split eligibility text at its 'Inclusion Criteria' / 'Exclusion Criteria' headings.

    The returned sections are exact slices of `full_text` (only leading and
    trailing whitespace is trimmed). We do not attempt to split individual
    criteria here — that is deferred to the Protocol Agent.

    If exactly one inclusion heading and one exclusion heading are not found,
    in that order, we report `split_succeeded=False` rather than guessing.
    """
    if not full_text:
        return EligibilitySections()

    inc = list(_INCLUSION_HEADING.finditer(full_text))
    exc = list(_EXCLUSION_HEADING.finditer(full_text))
    if len(inc) != 1 or len(exc) != 1 or inc[0].start() > exc[0].start():
        return EligibilitySections(split_succeeded=False)

    inclusion = full_text[inc[0].end() : exc[0].start()].strip()
    exclusion = full_text[exc[0].end() :].strip()
    return EligibilitySections(inclusion_text=inclusion, exclusion_text=exclusion, split_succeeded=True)
