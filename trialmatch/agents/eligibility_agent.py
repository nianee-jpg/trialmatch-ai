"""Eligibility Agent: evaluates each criterion for one SYNTHETIC patient.

How a criterion is evaluated:
  1. If a rule for this criterion exists in RULES (written for NCT02260934),
     that rule runs. Rules only combine the small checks below.
  2. Otherwise, a criterion with sub-criteria combines them ("any one of").
  3. Otherwise, a criterion with numeric thresholds is compared by code.
  4. Otherwise it goes to a person (NEEDS_HUMAN_REVIEW).

Safety rules built in:
  * All patient data comes from the Patient Agent's tools (logged).
  * Missing information -> UNKNOWN. Conflicting sources -> NEEDS_HUMAN_REVIEW.
  * A threshold missing from the protocol, or a unit that is not stated or does
    not match the record, -> NEEDS_HUMAN_REVIEW. Units are never converted.
  * Time limits that do not say which date they count from are checked from
    both screening and planned enrollment; if the answers differ -> NEEDS_HUMAN_REVIEW.
  * No overall eligibility decision is produced.

Optional: with an API key, Claude may add an advisory suggestion to criteria
that need a person. It can only cite evidence it actually retrieved through
the Patient Agent's tools, and the status stays NEEDS_HUMAN_REVIEW.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any, Callable, Dict, List, Optional, Set

from trialmatch import config
from trialmatch.activity_log import ActivityLog
from trialmatch.agents.claude_loop import run_tool_loop
from trialmatch.agents.eligibility_logic import (
    D,
    Finding,
    as_number,
    biopsy_is_proliferative,
    classify_result,
    combine_all,
    combine_any,
    compare,
    describe_comparison,
    exposure_date,
    fmt_num,
    in_window,
    merge,
    parse_date,
    says_none,
    window_start,
)
from trialmatch.agents.patient_agent import PATIENT_TOOLS, PatientAgent
from trialmatch.data_dictionary import field_label
from trialmatch.eligibility_models import (
    AISuggestion,
    CriterionEvaluation,
    EvidenceItem,
    ExclusionStatus,
    InclusionStatus,
    Method,
    ScreeningResult,
)
from trialmatch.protocol_models import Comparator, Criterion, CriterionType, FlagCode, Logic, ProtocolExtraction
from trialmatch.study_parser import StudyMetadata

AGENT_NAME = "Eligibility Agent"

ANCHOR_FIELDS = {
    "visit -1": "visits.visit_minus_1_date",
    "study enrollment": "visits.enrollment_date",
    "screening": "visits.screening_date",
}

# Named B-cell therapies recognised for EXC-25 ("... or other biologic B cell therapy").
B_CELL_THERAPIES = ["rituximab", "belimumab", "atacicept", "ocrelizumab", "obinutuzumab", "ofatumumab",
                    "epratuzumab", "tabalumab", "blisibimod"]

_STATUS = {
    CriterionType.INCLUSION: {D.TRUE: InclusionStatus.MET, D.FALSE: InclusionStatus.NOT_MET,
                              D.UNKNOWN: InclusionStatus.UNKNOWN, D.REVIEW: InclusionStatus.NEEDS_HUMAN_REVIEW},
    CriterionType.EXCLUSION: {D.TRUE: ExclusionStatus.PRESENT, D.FALSE: ExclusionStatus.ABSENT,
                              D.UNKNOWN: ExclusionStatus.UNKNOWN, D.REVIEW: ExclusionStatus.NEEDS_HUMAN_REVIEW},
}


class EligibilityAgent:
    def __init__(self, log: ActivityLog, patient_agent: PatientAgent, client: Any = None,
                 model: str = config.ANTHROPIC_MODEL) -> None:
        self.log = log
        self.patients = patient_agent
        self.client = client
        self.model = model

    # ================================================================== #
    # Entry point
    # ================================================================== #

    def screen(self, patient_id: str, extraction: ProtocolExtraction, study: StudyMetadata) -> ScreeningResult:
        self.pid = patient_id
        self.extraction = extraction
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._findings: Dict[str, Finding] = {}
        self._children = {c.criterion_id: [k for k in extraction.criteria if k.parent_id == c.criterion_id]
                          for c in extraction.criteria}

        self.log.add(AGENT_NAME, "info", f"Pre-screening {patient_id} (SYNTHETIC) against "
                     f"{len(extraction.criteria)} criteria of {extraction.nct_id}.")
        evaluations = [self._age_check(study)]
        for c in extraction.criteria:
            evaluations.append(self._to_evaluation(c, self._evaluate(c)))

        mode = "offline (rule-based)"
        if self.client is not None:
            if self._claude_suggestions(evaluations):
                mode = f"rule-based + Claude suggestions ({self.model})"
        result = ScreeningResult(patient_id=patient_id, nct_id=extraction.nct_id, mode=mode,
                                 screened_at_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 evaluations=evaluations)
        summary = "; ".join(f"{t.lower()}: " + ", ".join(f"{n} {s}" for s, n in sorted(c.items()))
                            for t, c in result.counts().items())
        self.log.add(AGENT_NAME, "info", f"Finished {patient_id}. {summary}. No overall eligibility decision is made.")
        return result

    # ================================================================== #
    # Dispatch
    # ================================================================== #

    def _evaluate(self, c: Criterion) -> Finding:
        if c.criterion_id in self._findings:
            return self._findings[c.criterion_id]
        rule = RULES.get(c.criterion_id)
        children = self._children.get(c.criterion_id, [])
        if rule:
            finding = rule(self, c)
        elif children and c.logic == Logic.ANY_OF:
            finding = self._any_of_children(c)
        elif c.thresholds:
            finding = self._numeric(c)
        else:
            finding = self._needs_person(c, "The wording needs a person to interpret.", [])
        self._findings[c.criterion_id] = finding
        decision_word = {D.TRUE: "condition holds", D.FALSE: "condition does not hold",
                         D.UNKNOWN: "information missing", D.REVIEW: "needs a person"}[finding.decision]
        self.log.add(AGENT_NAME, "info", f"{c.criterion_id}: {decision_word}. {finding.reason}")
        return finding

    def _to_evaluation(self, c: Criterion, f: Finding) -> CriterionEvaluation:
        return CriterionEvaluation(
            criterion_id=c.criterion_id, type=c.type.value, criterion_text=c.display_text, parent_id=c.parent_id,
            status=_STATUS[c.type][f.decision].value, method=Method(f.method),
            reason=f.reason, comparisons=f.comparisons, evidence=[EvidenceItem(**e) for e in f.evidence],
            assumptions=f.assumptions, missing_information=f.missing, review_reasons=f.review_reasons,
        )

    # ================================================================== #
    # Patient data (always through the Patient Agent's tools)
    # ================================================================== #

    def _get(self, field: str) -> Dict[str, Any]:
        key = f"field:{field}"
        if key not in self._cache:
            self._cache[key] = self.patients.call("get_field", {"patient_id": self.pid, "field": field})
        return self._cache[key]

    def _meds(self, drug: str) -> Dict[str, Any]:
        key = f"med:{drug}"
        if key not in self._cache:
            self._cache[key] = self.patients.call("get_medication_history", {"patient_id": self.pid, "drug": drug})
        return self._cache[key]

    def _date_of(self, field: str) -> Optional[dt.date]:
        res = self._get(field)
        if res["status"] != "FOUND" or res.get("consistency") == "CONFLICTING":
            return None
        return parse_date(res["observations"][0]["value"])

    @staticmethod
    def _evidence(res: Dict[str, Any], observations: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        out = []
        for o in observations if observations is not None else res.get("observations", []):
            out.append({"evidence_id": o["evidence_id"], "field": o["evidence_id"].split("/")[1],
                        "value": o["value"], "unit": o.get("unit"), "date": o.get("date"),
                        "end_date": o.get("end_date"), "source": o.get("source"), "note": o.get("note")})
        return out

    @staticmethod
    def _latest(observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        return max(observations, key=lambda o: o.get("date") or "")

    def _not_recorded(self, field: str, what: Optional[str] = None) -> Finding:
        label = what or field_label(field)
        return Finding(D.UNKNOWN, f"{label} is not in the record, so this cannot be checked.", missing=[field],
                       method="record lookup")

    def _conflict(self, res: Dict[str, Any], what: Optional[str] = None) -> Finding:
        label = what or field_label(res["field"])
        return Finding(D.REVIEW, f"The record has disagreeing entries for {label.lower() if what else label}. "
                       "A person must check the original documents.", evidence=self._evidence(res),
                       review_reasons=["Sources disagree"], method="needs a person")

    # ================================================================== #
    # Building blocks
    # ================================================================== #

    def _numeric(self, c: Criterion) -> Finding:
        """Compare every numeric threshold of a criterion with the record. Several thresholds = OR."""
        if len(c.thresholds) > 1 and " or " not in c.display_text.lower():
            return self._needs_person(c, "This criterion has several numbers and it is unclear how they combine.", [])
        parts: List[Finding] = []
        for t in c.thresholds:
            label = field_label(t.parameter) if t.parameter else "the measurement"
            if t.parameter is None:
                parts.append(Finding(D.REVIEW, "The app could not tell what this number measures.",
                                     review_reasons=["Unclear what is measured"], method="needs a person"))
                continue
            res = self._get(t.parameter)
            if t.value is None:
                parts.append(Finding(
                    D.REVIEW, "The protocol leaves out the number for this limit, so the app will not compare.",
                    evidence=self._evidence(res), review_reasons=["Number missing from the protocol"],
                    comparisons=[f"Protocol: {label} below [number missing] {t.unit or ''}".strip() + "."],
                    method="needs a person"))
                continue
            if res["status"] != "FOUND":
                parts.append(self._not_recorded(t.parameter))
                continue
            if res.get("consistency") == "CONFLICTING":
                parts.append(self._conflict(res))
                continue
            obs = self._latest(res["observations"])
            value = as_number(obs["value"])
            if value is None:
                parts.append(Finding(D.REVIEW, f"The recorded {label} is not a number.", evidence=self._evidence(res, [obs]),
                                     review_reasons=["Result is not a number"], method="needs a person"))
                continue
            assumptions = []
            if len(res["observations"]) > 1:
                assumptions.append(f"The record has {label} on several dates; the most recent ({obs['date']}) was used.")

            if t.relative_to:
                uln = as_number((obs.get("attributes") or {}).get("uln"))
                if uln is None:
                    parts.append(Finding(D.UNKNOWN, f"The lab's upper limit of normal for {label} is not recorded.",
                                         evidence=self._evidence(res, [obs]), missing=[t.parameter]))
                    continue
                limit = t.value * uln
                text = describe_comparison(label, value, obs.get("unit") or "", t.comparator, limit,
                                           f"{fmt_num(t.value)} × the upper limit of normal "
                                           f"({fmt_num(t.value)} × {fmt_num(uln)} = {fmt_num(limit)} {obs.get('unit') or ''})".rstrip())
                decision = D.TRUE if compare(value, t.comparator, limit) else D.FALSE
                parts.append(Finding(decision, text, comparisons=[text], evidence=self._evidence(res, [obs]),
                                     assumptions=assumptions))
                continue

            if t.unit is None:
                hypothetical = describe_comparison(label, value, obs.get("unit") or "", t.comparator, t.value,
                                                   fmt_num(t.value))
                parts.append(Finding(
                    D.REVIEW, f"The protocol gives no unit for {label}, so the app will not assume the record's "
                    f"unit ({obs.get('unit') or 'none'}) matches.",
                    comparisons=[f"For information only, if the units match: {hypothetical}"],
                    evidence=self._evidence(res, [obs]), review_reasons=["No unit given"], method="needs a person"))
                continue
            if (obs.get("unit") or "") != t.unit:
                parts.append(Finding(
                    D.REVIEW, f"The record gives {label} in {obs.get('unit')} but the protocol uses {t.unit}. "
                    "The app does not convert units.", evidence=self._evidence(res, [obs]),
                    review_reasons=["Units differ"], method="needs a person"))
                continue
            text = describe_comparison(label, value, t.unit, t.comparator, t.value)
            decision = D.TRUE if compare(value, t.comparator, t.value) else D.FALSE
            parts.append(Finding(decision, text, comparisons=[text], evidence=self._evidence(res, [obs]),
                                 assumptions=assumptions))

        decision = combine_any(p.decision for p in parts)
        reason = " ".join(p.reason for p in parts) if len(parts) > 1 else parts[0].reason
        method = "deterministic" if decision in (D.TRUE, D.FALSE) else (
            "needs a person" if decision == D.REVIEW else "deterministic")
        return merge(decision, reason, parts, method=method)

    def _lookup(self, field: str, holds_when_true: bool = True, what: Optional[str] = None) -> Finding:
        """A documented yes/no. `holds_when_true=False` inverts it (e.g. 'able to comply' vs 'inability')."""
        res = self._get(field)
        if res["status"] != "FOUND":
            return self._not_recorded(field, what)
        if res.get("consistency") == "CONFLICTING":
            return self._conflict(res, what)
        obs = self._latest(res["observations"])
        value = obs["value"]
        label = what or field_label(field)
        if isinstance(value, bool):
            answer: Optional[bool] = value
        else:
            none = says_none(value)
            answer = False if none else None
        shown = {True: "Yes", False: "No"}.get(value, str(value)) if isinstance(value, bool) else str(value)
        line = f"{label}: {shown} ({obs.get('source')}{', ' + obs['date'] if obs.get('date') else ''})."
        if answer is None:
            return Finding(D.REVIEW, f"The record describes {label.lower()} in words that need a person to read.",
                           comparisons=[line], evidence=self._evidence(res, [obs]),
                           review_reasons=["Record entry needs interpretation"], method="needs a person")
        holds = answer if holds_when_true else not answer
        return Finding(D.TRUE if holds else D.FALSE, line, comparisons=[line], evidence=self._evidence(res, [obs]),
                       method="record lookup")

    def _test_result(self, field: str, what: Optional[str] = None) -> Finding:
        """A documented positive/negative test. Positive -> condition holds."""
        res = self._get(field)
        label = what or field_label(field)
        if res["status"] != "FOUND":
            return self._not_recorded(field, what)
        if res.get("consistency") == "CONFLICTING":
            return self._conflict(res, what)
        obs = self._latest(res["observations"])
        values = obs["value"].values() if isinstance(obs["value"], dict) else [obs["value"]]
        classes = [classify_result(v, obs.get("note")) for v in values]
        shown = (", ".join(f"{k.replace('_', '-')} {v}" for k, v in obs["value"].items())
                 if isinstance(obs["value"], dict) else str(obs["value"]))
        line = f"{label}: {shown} ({obs.get('source')}{', ' + obs['date'] if obs.get('date') else ''})."
        ev = self._evidence(res, [obs])
        if "positive" in classes:
            return Finding(D.TRUE, line, comparisons=[line], evidence=ev, method="record lookup")
        if classes and all(c == "negative" for c in classes):
            return Finding(D.FALSE, line, comparisons=[line], evidence=ev, method="record lookup")
        return Finding(D.REVIEW, f"The {label} result is not clearly positive or negative.", comparisons=[line],
                       evidence=ev, review_reasons=["Result unclear"], method="needs a person")

    def _window(self, event: dt.date, amount: int, unit: str, anchor: Optional[str], what: str) -> Finding:
        """Is `event` within the time limit? Handles a stated or an unstated reference date."""
        limit = f"{amount} {unit}"
        if anchor:
            ref = self._date_of(ANCHOR_FIELDS[anchor])
            if ref is None:
                return self._not_recorded(ANCHOR_FIELDS[anchor])
            d = in_window(event, ref, amount, unit)
            start = window_start(ref, amount, unit)
            line = (f"{what} on {event} is {_window_words(d)} the {limit} before {anchor} ({ref}); "
                    f"the window starts {start}.")
            f = Finding(d, line, comparisons=[line], method="deterministic")
            if d == D.REVIEW:
                f.reason = f"{what} falls exactly on the first day of the window; the protocol does not say whether it counts."
                f.review_reasons.append("Date exactly on the boundary")
                f.method = "needs a person"
            return f

        refs = {name: self._date_of(ANCHOR_FIELDS[name]) for name in ("screening", "study enrollment")}
        known = {n: r for n, r in refs.items() if r is not None}
        if not known:
            return Finding(D.UNKNOWN, "Neither the screening date nor the planned enrollment date is recorded.",
                           missing=[ANCHOR_FIELDS["screening"], ANCHOR_FIELDS["study enrollment"]])
        results = {n: in_window(event, r, amount, unit) for n, r in known.items()}
        lines = [f"Counted from {n} ({r}): {what} on {event} is {_window_words(results[n])} the {limit} "
                 f"(window starts {window_start(r, amount, unit)})." for n, r in known.items()]
        if len(set(results.values())) == 1:
            d = next(iter(results.values()))
            if d == D.REVIEW:
                return Finding(D.REVIEW, f"{what} falls exactly on the first day of the window.", comparisons=lines,
                               review_reasons=["Date exactly on the boundary"], method="needs a person")
            note = ("The protocol does not say which date this time limit counts from. "
                    + ("Counting from screening and from planned enrollment gives the same answer."
                       if len(known) == 2 else f"Only the {next(iter(known))} date is recorded, so it was used."))
            return Finding(d, lines[0] if len(lines) == 1 else f"{what} is {_window_words(d)} the {limit} either way.",
                           comparisons=lines, assumptions=[note], method="deterministic")
        return Finding(D.REVIEW, "The answer depends on which date the time limit counts from: "
                       + "; ".join(f"{_window_words(d)} if counted from {n}" for n, d in results.items())
                       + ". The protocol does not say, so a person must decide.",
                       comparisons=lines, review_reasons=["Result depends on the start date of the time limit"],
                       method="needs a person")

    def _event_within(self, field: str, amount: int, unit: str, anchor: Optional[str]) -> Finding:
        """Did a documented event (vaccine, procedure, hospital stay ...) happen within the time limit?"""
        res = self._get(field)
        label = field_label(field)
        if res["status"] != "FOUND":
            return self._not_recorded(field)
        if res.get("consistency") == "CONFLICTING":
            return self._conflict(res)
        parts = []
        for obs in res["observations"]:
            if says_none(obs["value"]):
                line = f"{label}: none documented ({obs.get('source')})."
                parts.append(Finding(D.FALSE, line, comparisons=[line], evidence=self._evidence(res, [obs]),
                                     method="record lookup"))
                continue
            when = parse_date(obs.get("date"))
            if when is None:
                parts.append(Finding(D.UNKNOWN, f"{label}: '{obs['value']}' has no date.", missing=[field],
                                     evidence=self._evidence(res, [obs])))
                continue
            f = self._window(when, amount, unit, anchor, f"{obs['value']}")
            f.evidence = self._evidence(res, [obs])
            parts.append(f)
        d = combine_any(p.decision for p in parts)
        return merge(d, " ".join(p.reason for p in parts), parts)

    def _medicine_within(self, drugs: List[str], amount: int, unit: str, anchor: Optional[str]) -> Finding:
        """Was any of `drugs` taken within the time limit (using the last dose or ongoing use)?"""
        parts: List[Finding] = []
        base = self._get("medications.history")
        if base["status"] != "FOUND":
            return self._not_recorded("medications.history")
        reference = self._date_of(ANCHOR_FIELDS[anchor or "screening"]) or dt.date.today()
        for drug in drugs:
            res = self._meds(drug)
            for obs in res.get("observations", []):
                last = exposure_date(obs, reference)
                if last is None:
                    parts.append(Finding(D.UNKNOWN, f"The record does not say when {drug} was stopped.",
                                         evidence=self._evidence(res, [obs]), missing=["medications.history"]))
                    continue
                f = self._window(last, amount, unit, anchor, f"Last {drug} dose" if obs.get("end_date") else
                                 f"{drug.capitalize()} (ongoing)")
                f.evidence = self._evidence(res, [obs])
                parts.append(f)
        if not parts:
            names = " or ".join(drugs)
            return Finding(D.FALSE, f"No {names} in the medication history.",
                           comparisons=[f"Medication history searched for: {names}. None found."],
                           evidence=self._evidence(base), method="record lookup",
                           assumptions=["Assumes the medication history in the record is complete."])
        d = combine_any(p.decision for p in parts)
        return merge(d, " ".join(p.reason for p in parts), parts)

    def _ever_taken(self, drugs: List[str], biologics: bool = False) -> Finding:
        """Was any of `drugs` ever given?

        If `biologics` is True, a statement such as "no prior biologic therapy"
        contradicts a documented administration and sends the criterion to a person.
        """
        base = self._get("medications.history")
        if base["status"] != "FOUND":
            return self._not_recorded("medications.history")
        found, statements = [], []
        for drug in drugs:
            res = self._meds(drug)
            found += [(drug, o, res) for o in res.get("observations", [])]
            statements += [(o, res) for o in res.get("negative_statements", [])
                           if biologics and "biologic" in str(o["value"]).lower() and (o, res) not in statements]
        if found:
            ev = [e for _, o, res in found for e in self._evidence(res, [o])]
            lines = [f"{drug.capitalize()} given {o.get('date')}" + (f" to {o['end_date']}" if o.get("end_date") else "")
                     + f" ({o.get('source')})." for drug, o, _ in found]
            if statements:
                stmt, res = statements[0]
                return Finding(D.REVIEW, f"The record documents {found[0][0]}, but also says '{stmt['value']}'. "
                               "A person must resolve the contradiction.", comparisons=lines,
                               evidence=ev + self._evidence(res, [stmt]),
                               review_reasons=["Sources disagree"], method="needs a person")
            return Finding(D.TRUE, " ".join(lines), comparisons=lines, evidence=ev, method="record lookup")
        return Finding(D.FALSE, f"None of these is in the medication history: {', '.join(drugs)}.",
                       comparisons=[f"Medication history searched for: {', '.join(drugs)}. None found."],
                       evidence=self._evidence(base), method="record lookup",
                       assumptions=["Assumes the medication history in the record is complete."])

    def _needs_person(self, c: Criterion, why: str, fields: List[str]) -> Finding:
        evidence: List[Dict[str, Any]] = []
        missing = []
        for f in fields:
            res = self._get(f)
            if res["status"] == "FOUND":
                evidence += self._evidence(res)
            else:
                missing.append(f)
        return Finding(D.REVIEW, why, evidence=evidence, missing=missing,
                       review_reasons=[why.rstrip(".")], method="needs a person")

    def _any_of_children(self, c: Criterion) -> Finding:
        kids = self._children[c.criterion_id]
        parts = [self._evaluate(k) for k in kids]
        d = combine_any(p.decision for p in parts)
        summary = ", ".join(f"{k.criterion_id}: {_short(p.decision, c.type)}" for k, p in zip(kids, parts))
        f = Finding(d, f"Needs any one of {', '.join(k.criterion_id for k in kids)} ({summary}).", method="combined")
        return f

    # ================================================================== #
    # Registry age limits (structured fields, not in the eligibility text)
    # ================================================================== #

    def _age_check(self, study: StudyMetadata) -> CriterionEvaluation:
        lo = _years(study.eligibility.minimum_age)
        hi = _years(study.eligibility.maximum_age)
        text = (f"Age at least {lo} years" if lo is not None else "No minimum age") + (
            f" and at most {hi} years" if hi is not None else ", no maximum age") + \
            " (registry age limits, protocolSection.eligibilityModule)."
        res = self._get("demographics.age")
        if res["status"] != "FOUND":
            f = self._not_recorded("demographics.age")
        elif res.get("consistency") == "CONFLICTING":
            f = self._conflict(res)
        else:
            obs = self._latest(res["observations"])
            age = as_number(obs["value"])
            lines, ds = [], []
            if lo is not None:
                lines.append(describe_comparison("Age", age, "years", Comparator.GE, lo))
                ds.append(D.TRUE if age >= lo else D.FALSE)
            if hi is not None:
                lines.append(describe_comparison("Age", age, "years", Comparator.LE, hi))
                ds.append(D.TRUE if age <= hi else D.FALSE)
            f = Finding(combine_all(ds) if ds else D.TRUE, " ".join(lines) or "No age limits.", comparisons=lines,
                        evidence=self._evidence(res, [obs]),
                        assumptions=["Registry minimum age is treated as inclusive (18 years old counts)."]
                        if lo is not None else [])
        self.log.add(AGENT_NAME, "info", f"REG-AGE: {f.reason}")
        return CriterionEvaluation(criterion_id="REG-AGE", type="INCLUSION", criterion_text=text,
                                   status=_STATUS[CriterionType.INCLUSION][f.decision].value,
                                   method=Method(f.method), reason=f.reason, comparisons=f.comparisons,
                                   evidence=[EvidenceItem(**e) for e in f.evidence], assumptions=f.assumptions,
                                   missing_information=f.missing, review_reasons=f.review_reasons)

    # ================================================================== #
    # Optional Claude suggestions (advisory only)
    # ================================================================== #

    def _claude_suggestions(self, evaluations: List[CriterionEvaluation]) -> bool:
        candidates = {e.criterion_id: e for e in evaluations
                      if e.status == "NEEDS_HUMAN_REVIEW" and e.method == Method.NEEDS_PERSON}
        if not candidates:
            return False
        seen: Set[str] = set()

        def patient_tool(name: str) -> Callable[[Dict[str, Any]], Any]:
            def run(inp: Dict[str, Any]) -> Any:
                if inp.get("patient_id") not in (None, self.pid):
                    raise ValueError(f"Only {self.pid} may be queried in this screening.")
                out = self.patients.call(name, inp)
                for key in ("observations", "negative_statements"):
                    seen.update(o["evidence_id"] for o in out.get(key, []))
                return out
            return run

        def submit(inp: Dict[str, Any]) -> Dict[str, Any]:
            e = candidates.get(inp["criterion_id"])
            if e is None:
                return {"status": "rejected", "reason": "Only criteria that need a person can get a suggestion."}
            allowed = InclusionStatus if e.type == "INCLUSION" else ExclusionStatus
            if inp["suggested_status"] not in {s.value for s in allowed} - {"NEEDS_HUMAN_REVIEW"}:
                return {"status": "rejected", "reason": f"suggested_status must be one of "
                        f"{sorted({s.value for s in allowed} - {'NEEDS_HUMAN_REVIEW'})}."}
            unknown = [i for i in inp["evidence_ids"] if i not in seen]
            if unknown or not inp["evidence_ids"]:
                return {"status": "rejected", "reason": "Cite at least one evidence_id you retrieved with the "
                        f"patient tools. Not retrieved: {unknown}"}
            e.ai_suggestion = AISuggestion(suggested_status=inp["suggested_status"], rationale=inp["rationale"],
                                           evidence_ids=inp["evidence_ids"])
            return {"status": "accepted", "note": "Recorded as an advisory suggestion; status stays NEEDS_HUMAN_REVIEW."}

        handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {t["name"]: patient_tool(t["name"]) for t in PATIENT_TOOLS}
        handlers["submit_suggestion"] = submit
        listing = "\n".join(f"- {e.criterion_id} ({e.type}): {e.criterion_text} | why a person is needed: {e.reason}"
                            for e in candidates.values())
        return run_tool_loop(
            client=self.client, model=self.model, system=ELIGIBILITY_SYSTEM_PROMPT,
            tools=PATIENT_TOOLS + [SUBMIT_SUGGESTION_TOOL],
            first_message=(f"Patient {self.pid} (SYNTHETIC). These criteria need a person:\n{listing}\n\n"
                           "For each one you can reasonably assess, look up the evidence with the patient tools "
                           "and call submit_suggestion. Skip any you cannot support with evidence."),
            handlers=handlers, log=self.log, agent_name=AGENT_NAME,
        )


# ---------------------------------------------------------------------- #
# Criterion rules for NCT02260934 (CALIBRATE)
# ---------------------------------------------------------------------- #

def _inc01(a: EligibilityAgent, c: Criterion) -> Finding:
    return a._lookup("diagnosis.sle_acr_criteria")


def _inc02(a: EligibilityAgent, c: Criterion) -> Finding:
    """Positive ANA or anti-dsDNA at visit -1 or within 14 days before it."""
    ref = a._date_of(ANCHOR_FIELDS["visit -1"])
    if ref is None:
        return a._not_recorded(ANCHOR_FIELDS["visit -1"])
    tests = []
    for field in ("labs.ana", "labs.anti_dsdna"):
        res = a._get(field)
        label = field_label(field)
        if res["status"] != "FOUND":
            tests.append(a._not_recorded(field))
            continue
        if res.get("consistency") == "CONFLICTING":
            tests.append(a._conflict(res))
            continue
        per_obs = []
        for obs in res["observations"]:
            when = parse_date(obs.get("date"))
            result = classify_result(obs["value"], obs.get("note"))
            ev = a._evidence(res, [obs])
            if when is None or result is None:
                per_obs.append(Finding(D.REVIEW, f"{label}: date or result unclear.", evidence=ev,
                                       review_reasons=["Result unclear"], method="needs a person"))
                continue
            if when > ref:
                w = Finding(D.FALSE, f"{label} on {when} is after visit -1 ({ref}).",
                            comparisons=[f"{label} on {when} is after visit -1 ({ref}), so it does not count."])
            else:
                w = a._window(when, 14, "days", "visit -1", f"{label} ({result})")
            positive = Finding(D.TRUE if result == "positive" else D.FALSE, f"{label} is {result}.",
                               comparisons=[f"{label}: {obs['value']} ({result})."], evidence=ev)
            per_obs.append(merge(combine_all([positive.decision, w.decision]),
                                 f"{label} {result} on {when}: {_window_words(w.decision)} the 14-day window.",
                                 [positive, w]))
        tests.append(merge(combine_any(p.decision for p in per_obs), " ".join(p.reason for p in per_obs), per_obs))
    return merge(combine_any(t.decision for t in tests), " ".join(t.reason for t in tests), tests)


def _biopsy(a: EligibilityAgent, months: int) -> Finding:
    res = a._get("renal.biopsy_class")
    if res["status"] != "FOUND":
        return a._not_recorded("renal.biopsy_class")
    if res.get("consistency") == "CONFLICTING":
        return a._conflict(res)
    obs = a._latest(res["observations"])
    ev = a._evidence(res, [obs])
    proliferative = biopsy_is_proliferative(str(obs["value"]))
    if proliferative is None:
        cls = Finding(D.REVIEW, "The biopsy class could not be read.", evidence=ev,
                      review_reasons=["Biopsy class unclear"], method="needs a person")
    else:
        line = (f"Biopsy {obs['value']} {'is' if proliferative else 'is not'} Class III, IV, "
                "or V in combination with III or IV.")
        cls = Finding(D.TRUE if proliferative else D.FALSE, line, comparisons=[line], evidence=ev)
    when = parse_date(obs.get("date"))
    if when is None:
        return merge(D.UNKNOWN if cls.decision != D.FALSE else D.FALSE, "Biopsy date not recorded.",
                     [cls, a._not_recorded("renal.biopsy_date")])
    w = a._window(when, months, "months", None, "Biopsy")
    d = combine_all([cls.decision, w.decision])
    return merge(d, f"{cls.reason} {w.reason}", [cls, w])


def _inc03a(a: EligibilityAgent, c: Criterion) -> Finding:
    return _biopsy(a, 3)


def _inc03b(a: EligibilityAgent, c: Criterion) -> Finding:
    biopsy = _biopsy(a, 12)
    kids = a._children[c.criterion_id]
    sediment_parts = [a._evaluate(k) for k in kids]
    sediment = combine_any(p.decision for p in sediment_parts)
    summary = ", ".join(f"{k.criterion_id}: {_short(p.decision, c.type)}" for k, p in zip(kids, sediment_parts))
    f = merge(combine_all([biopsy.decision, sediment]),
              f"Needs a qualifying biopsy within 12 months AND active urinary sediment (any one of {summary}). "
              f"{biopsy.reason}", [biopsy])
    if any(k.has_flag(FlagCode.STRUCTURE_INFERRED) for k in kids):
        f.assumptions.append("Grouping of the urinary sediment items under INC-03.b was worked out by the app "
                             "from the protocol layout.")
    f.method = "combined"
    return f


def _no_menses_or_infection(a: EligibilityAgent, menses: bool) -> List[Finding]:
    parts = [a._lookup("infections.current", holds_when_true=False, what="Infection at collection")]
    if menses:
        parts.insert(0, a._lookup("reproductive.menses_at_collection", holds_when_true=False,
                                  what="Menstruating at collection"))
    return parts


def _inc03b1(a: EligibilityAgent, c: Criterion) -> Finding:
    parts = [a._numeric(c)] + _no_menses_or_infection(a, menses=True)
    return merge(combine_all(p.decision for p in parts), " ".join(p.reason for p in parts), parts)


def _inc03b2(a: EligibilityAgent, c: Criterion) -> Finding:
    parts = [a._numeric(c)] + _no_menses_or_infection(a, menses=False)
    return merge(combine_all(p.decision for p in parts), " ".join(p.reason for p in parts), parts)


def _inc03b3(a: EligibilityAgent, c: Criterion) -> Finding:
    res = a._get("urinalysis.cellular_casts")
    if res["status"] != "FOUND":
        return a._not_recorded("urinalysis.cellular_casts")
    if res.get("consistency") == "CONFLICTING":
        return a._conflict(res)
    obs = a._latest(res["observations"])
    text = str(obs["value"]).lower()
    line = f"Urine casts: {obs['value']} ({obs.get('source')}, {obs.get('date')})."
    ev = a._evidence(res, [obs])
    if re.search(r"\b(rbc|wbc) casts? present", text):
        return Finding(D.TRUE, line, comparisons=[line], evidence=ev, method="record lookup")
    if says_none(text) or "none seen" in text:
        return Finding(D.FALSE, line, comparisons=[line], evidence=ev, method="record lookup")
    return Finding(D.REVIEW, "The casts description needs a person to read.", comparisons=[line], evidence=ev,
                   review_reasons=["Record entry needs interpretation"], method="needs a person")


def _inc04(a: EligibilityAgent, c: Criterion) -> Finding:
    """UPCR > 1 based on a 24-hour collection."""
    method = a._get("labs.upcr_collection_method")
    if method["status"] != "FOUND":
        return a._not_recorded("labs.upcr_collection_method")
    obs = a._latest(method["observations"])
    if "24-hour" not in str(obs["value"]).lower():
        upcr = a._get("labs.upcr")
        line = f"UPCR collection: {obs['value']}. The protocol requires a 24-hour collection."
        return Finding(D.UNKNOWN, "The record only has a UPCR from a " + str(obs["value"]) +
                       ", not the 24-hour collection the protocol requires.", comparisons=[line],
                       evidence=a._evidence(method, [obs]) + a._evidence(upcr),
                       missing=["labs.upcr"])
    num = a._numeric(c)
    line = f"UPCR collection: {obs['value']} (matches the protocol)."
    num.comparisons.insert(0, line)
    num.evidence += a._evidence(method, [obs])
    return num


def _inc05(a: EligibilityAgent, c: Criterion) -> Finding:
    return a._lookup("consent.able_to_consent")


def _exc01(a: EligibilityAgent, c: Criterion) -> Finding:
    """New onset LN = never treated with mycophenolate mofetil or cyclophosphamide (exclusion PRESENT)."""
    ever = a._ever_taken(["mycophenolate", "cyclophosphamide"])
    flipped = {D.TRUE: D.FALSE, D.FALSE: D.TRUE}.get(ever.decision, ever.decision)
    reason = ("Previously treated, so this is not new-onset lupus nephritis. " + ever.reason
              if flipped == D.FALSE else ever.reason)
    return merge(flipped, reason, [ever])


def _exc07(a: EligibilityAgent, c: Criterion) -> Finding:
    res = a._get("imaging.chest_radiograph")
    if res["status"] != "FOUND":
        return a._not_recorded("imaging.chest_radiograph")
    obs = a._latest(res["observations"])
    line = f"Chest X-ray: {obs['value']} ({obs.get('source')}, {obs.get('date')})."
    if "no fibrotic changes" in str(obs["value"]).lower():
        return Finding(D.FALSE, line, comparisons=[line], evidence=a._evidence(res, [obs]), method="record lookup")
    return Finding(D.REVIEW, "The chest X-ray report needs a person to judge.", comparisons=[line],
                   evidence=a._evidence(res, [obs]), review_reasons=["Record entry needs interpretation"],
                   method="needs a person")


def _exc09(a: EligibilityAgent, c: Criterion) -> Finding:
    parts = [a._test_result("infections.hiv"), a._test_result("infections.hepatitis_b"),
             a._test_result("infections.hepatitis_c")]
    return merge(combine_any(p.decision for p in parts), " ".join(p.reason for p in parts), parts)


def _exc10(a: EligibilityAgent, c: Criterion) -> Finding:
    parts = [a._event_within("infections.hospitalizations", 60, "days", None),
             a._event_within("infections.parenteral_antimicrobials", 60, "days", None)]
    return merge(combine_any(p.decision for p in parts), " ".join(p.reason for p in parts), parts)


def _exc15(a: EligibilityAgent, c: Criterion) -> Finding:
    f = a._lookup("history.malignancy")
    if f.decision == D.TRUE:
        f.decision, f.method = D.REVIEW, "needs a person"
        f.reason = "A malignancy is recorded; a person must check whether it is one of the allowed exceptions."
        f.review_reasons.append("Exceptions need a person")
    return f


def _exc22(a: EligibilityAgent, c: Criterion) -> Finding:
    return a._needs_person(c, "The time limit depends on each drug's half-life, which the protocol does not give.",
                           ["medications.history"])


def _exc24(a: EligibilityAgent, c: Criterion) -> Finding:
    base = a._get("medications.history")
    if base["status"] != "FOUND":
        return a._not_recorded("medications.history")
    investigational = [o for o in base["observations"]
                       if "investigational" in f"{o['value']} {o.get('note') or ''} {o.get('attributes')}".lower()]
    if not investigational:
        return Finding(D.FALSE, "No investigational biologic agent in the medication history.",
                       comparisons=["Medication history checked for entries marked investigational. None found."],
                       evidence=a._evidence(base), method="record lookup",
                       assumptions=["Assumes the medication history in the record is complete."])
    return a._medicine_within([str(o["value"]) for o in investigational], 12, "months", None)


def _exc25(a: EligibilityAgent, c: Criterion) -> Finding:
    f = a._ever_taken(B_CELL_THERAPIES, biologics=True)
    if f.decision == D.FALSE:
        f.assumptions.append("'Other biologic B cell therapy' was checked against this list: "
                             + ", ".join(B_CELL_THERAPIES) + ".")
    return f


def _exc28(a: EligibilityAgent, c: Criterion) -> Finding:
    comorbid = a._lookup("history.comorbidities")
    courses = a._get("medications.corticosteroid_courses")
    if courses["status"] != "FOUND":
        return merge(D.UNKNOWN, "Steroid courses in the past year are not recorded.",
                     [comorbid, a._not_recorded("medications.corticosteroid_courses")])
    obs = a._latest(courses["observations"])
    n = as_number(obs["value"])
    line = f"Steroid courses in the past 12 months: {obs['value']} ({obs.get('note') or obs.get('source')})."
    ev = a._evidence(courses, [obs])
    if comorbid.decision == D.FALSE and n is not None and n < 3:
        return merge(D.FALSE, "No significant comorbidities recorded and fewer than three steroid courses.",
                     [comorbid, Finding(D.FALSE, line, comparisons=[line], evidence=ev)], method="record lookup")
    return merge(D.REVIEW, "A person must decide whether the steroid courses were for a comorbidity "
                 "(the protocol's concern) or for lupus itself.",
                 [comorbid, Finding(D.REVIEW, line, comparisons=[line], evidence=ev)], method="needs a person")


def _exc33(a: EligibilityAgent, c: Criterion) -> Finding:
    f = a._lookup("psychiatric.history")
    if f.decision == D.TRUE:
        f.decision, f.method = D.REVIEW, "needs a person"
        f.reason = "A psychiatric history is recorded; a person must judge whether it is severe."
        f.review_reasons.append("Severity needs a person")
    return f


def _investigator(fields: List[str]) -> Callable[[EligibilityAgent, Criterion], Finding]:
    def rule(a: EligibilityAgent, c: Criterion) -> Finding:
        return a._needs_person(c, "Part of this criterion depends on the study doctor's opinion.", fields)
    return rule


def _lookup_rule(field: str, holds_when_true: bool = True) -> Callable[[EligibilityAgent, Criterion], Finding]:
    return lambda a, c: a._lookup(field, holds_when_true)


def _test_rule(field: str) -> Callable[[EligibilityAgent, Criterion], Finding]:
    return lambda a, c: a._test_result(field)


RULES: Dict[str, Callable[[EligibilityAgent, Criterion], Finding]] = {
    "INC-01": _inc01,
    "INC-02": _inc02,
    "INC-03.a": _inc03a,
    "INC-03.b": _inc03b,
    "INC-03.b.1": _inc03b1,
    "INC-03.b.2": _inc03b2,
    "INC-03.b.3": _inc03b3,
    "INC-04": _inc04,
    "INC-05": _inc05,
    "EXC-01": _exc01,
    # EXC-02, -03, -04, -05, -14, -26: generic numeric comparison
    "EXC-06": _test_rule("infections.quantiferon_tb"),
    "EXC-07": _exc07,
    "EXC-08": _lookup_rule("infections.current"),
    "EXC-09": _exc09,
    "EXC-10": _exc10,
    "EXC-11": _lookup_rule("infections.chronic_suppressive_therapy"),
    "EXC-12": _investigator(["infections.history"]),
    "EXC-13": lambda a, c: a._event_within("vaccinations.live_attenuated", 3, "months", "study enrollment"),
    "EXC-15": _exc15,
    "EXC-16": _lookup_rule("history.transplantation"),
    "EXC-17": _lookup_rule("history.primary_immunodeficiency"),
    "EXC-18": _test_rule("reproductive.pregnancy_status"),
    "EXC-19": _lookup_rule("reproductive.breastfeeding"),
    "EXC-20": _lookup_rule("reproductive.contraception_willingness", holds_when_true=False),
    "EXC-21": lambda a, c: a._medicine_within(["cyclophosphamide"], 6, "months", None),
    "EXC-22": _exc22,
    "EXC-23": lambda a, c: a._event_within("procedures.ivig_plasmapheresis_leukopheresis", 90, "days", None),
    "EXC-24": _exc24,
    "EXC-25": _exc25,
    "EXC-27": _investigator(["history.comorbidities"]),
    "EXC-28": _exc28,
    "EXC-29": _lookup_rule("social.substance_abuse"),
    "EXC-30": _lookup_rule("allergies.monoclonal_antibodies"),
    "EXC-31": _lookup_rule("allergies.contrast_agents"),
    "EXC-32": _lookup_rule("access.peripheral_venous", holds_when_true=False),
    "EXC-33": _exc33,
    "EXC-34": _investigator(["psychiatric.suicidality"]),
    "EXC-35": _lookup_rule("compliance.able_to_comply", holds_when_true=False),
}


# ---------------------------------------------------------------------- #
# Claude tool for advisory suggestions
# ---------------------------------------------------------------------- #

ELIGIBILITY_SYSTEM_PROMPT = """You help a clinical research coordinator in an EDUCATIONAL SIMULATION. The \
patient record is SYNTHETIC. Some criteria could not be decided by code and need a person. For those, you may \
offer an advisory suggestion.

Rules:
- Look up evidence only with the patient tools, and cite the evidence_ids they return.
- Never infer missing data. If the record lacks what is needed, do not submit a suggestion.
- Never introduce numeric thresholds that are not in the criterion text.
- Your suggestion is advisory; a person makes the decision. Never describe the patient as eligible or ineligible."""

SUBMIT_SUGGESTION_TOOL = {
    "name": "submit_suggestion",
    "description": ("Record an advisory suggestion for a criterion that needs a person. Must cite evidence_ids "
                    "retrieved with the patient tools. The criterion's status remains NEEDS_HUMAN_REVIEW."),
    "input_schema": {
        "type": "object",
        "properties": {
            "criterion_id": {"type": "string"},
            "suggested_status": {"type": "string",
                                 "enum": ["MET", "NOT_MET", "UNKNOWN", "PRESENT", "ABSENT"]},
            "rationale": {"type": "string"},
            "evidence_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["criterion_id", "suggested_status", "rationale", "evidence_ids"],
        "additionalProperties": False,
    },
    "strict": True,
}


# ---------------------------------------------------------------------- #
# Small helpers
# ---------------------------------------------------------------------- #

def _window_words(d: D) -> str:
    return {D.TRUE: "inside", D.FALSE: "outside", D.REVIEW: "exactly on the boundary of",
            D.UNKNOWN: "of unknown position relative to"}[d]


def _short(d: D, ctype: CriterionType) -> str:
    return _STATUS[ctype][d].value.replace("_", " ").lower()


def _years(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.match(r"\s*(\d+)\s*Years?", text, re.IGNORECASE)
    return int(m.group(1)) if m else None
