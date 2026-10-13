"""Plain-language summary of a pre-screening result, shared by the app and the report.

Two things are worked out here:

  * The pre-screening STATUS of the patient: what the study team has to do
    next (confirm a possible rule-out, request information, make decisions).
    This is a workflow status, never an eligibility decision.
  * For each sub-criterion ("option"), what it means for its main criterion,
    e.g. "Option not needed: INC-03.b is met through INC-03.b.1 and INC-03.b.3".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from trialmatch.eligibility_models import CriterionEvaluation, ScreeningResult

POSITIVE = {"INCLUSION": "MET", "EXCLUSION": "PRESENT"}


@dataclass
class FollowUp:
    rule_out: List[CriterionEvaluation]   # NOT_MET inclusion or PRESENT exclusion
    unknown: List[CriterionEvaluation]    # information missing
    review: List[CriterionEvaluation]     # a person must decide


@dataclass
class Headline:
    level: str      # "rule_out", "review", "unknown" or "none"
    title: str
    details: List[str]


def main_criteria(result: ScreeningResult) -> List[CriterionEvaluation]:
    return [e for e in result.evaluations if e.parent_id is None]


def follow_up(result: ScreeningResult) -> FollowUp:
    top = main_criteria(result)
    return FollowUp(
        rule_out=[e for e in top if e.status in ("NOT_MET", "PRESENT")],
        unknown=[e for e in top if e.status == "UNKNOWN"],
        review=[e for e in top if e.status == "NEEDS_HUMAN_REVIEW"],
    )


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def headline(result: ScreeningResult) -> Headline:
    """The patient's pre-screening status, most serious first. Never says 'eligible'."""
    f = follow_up(result)
    details = []
    if f.rule_out:
        details.append(f"⛔ {_plural(len(f.rule_out), 'criterion', 'criteria')} may rule the patient out: "
                       + ", ".join(e.criterion_id for e in f.rule_out) + ".")
    if f.review:
        details.append(f"👤 {_plural(len(f.review), 'criterion needs', 'criteria need')} a decision by the "
                       "study team: " + ", ".join(e.criterion_id for e in f.review) + ".")
    if f.unknown:
        details.append(f"❔ {_plural(len(f.unknown), 'criterion cannot', 'criteria cannot')} be checked because "
                       "information is missing: " + ", ".join(e.criterion_id for e in f.unknown) + ".")

    if f.rule_out:
        return Headline("rule_out", "Possible rule-out found: confirm with the study team", details)
    if f.review:
        return Headline("review", "Needs review by the study team", details)
    if f.unknown:
        return Headline("unknown", "Information missing: request it before deciding", details)
    return Headline("none", "No problems found by the app",
                    ["The study team must still confirm eligibility. This is not an eligibility decision."])


def _children(result: ScreeningResult, parent_id: str) -> List[CriterionEvaluation]:
    return [e for e in result.evaluations if e.parent_id == parent_id]


def part_context(result: ScreeningResult, e: CriterionEvaluation) -> Optional[str]:
    """What a sub-criterion's result means for its main criterion. None for main criteria."""
    if e.parent_id is None:
        return None
    parent = result.by_id(e.parent_id)
    siblings = _children(result, e.parent_id)
    positive = POSITIVE[e.type]
    satisfied_by = [s.criterion_id for s in siblings if s.status == positive]
    others = [i for i in satisfied_by if i != e.criterion_id]
    parent_ok = parent.status == positive

    if e.status == positive:
        if parent_ok:
            return f"This option is satisfied, so it counts toward {parent.criterion_id}."
        return (f"This option is satisfied, but {parent.criterion_id} also needs its other conditions "
                f"(see {parent.criterion_id}).")
    if parent_ok and others:
        return (f"Option not needed: only one option is required, and {parent.criterion_id} is already met "
                f"through {_join(others)}.")
    if parent_ok:
        return f"Option not needed: {parent.criterion_id} is already met."
    this = f"This option is {e.status.replace('_', ' ').lower()}."
    if parent.status == "UNKNOWN":
        return (f"{parent.criterion_id} cannot be decided yet: no option is met so far, and some options are "
                f"missing information. {this}")
    if parent.status == "NEEDS_HUMAN_REVIEW":
        return f"{parent.criterion_id} needs a person to decide: no option is clearly met. {this}"
    return f"No option of {parent.criterion_id} is met, so {parent.criterion_id} is not met. {this}"


def option_label(result: ScreeningResult, e: CriterionEvaluation) -> Optional[str]:
    """Short tag for the results table: 'option · counts' or 'option · not needed'."""
    if e.parent_id is None:
        return None
    parent = result.by_id(e.parent_id)
    positive = POSITIVE[e.type]
    if e.status == positive:
        return "option · counts" if parent.status == positive else "option"
    return "option · not needed" if parent.status == positive else "option"


def _join(ids: List[str]) -> str:
    return ids[0] if len(ids) == 1 else ", ".join(ids[:-1]) + " and " + ids[-1]


def status_counts(result: ScreeningResult) -> Dict[str, Dict[str, int]]:
    """Counts for main criteria only."""
    out: Dict[str, Dict[str, int]] = {"INCLUSION": {}, "EXCLUSION": {}}
    for e in main_criteria(result):
        out[e.type][e.status] = out[e.type].get(e.status, 0) + 1
    return out
