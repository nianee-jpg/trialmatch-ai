"""Verification Agent: independently checks every pre-screening conclusion.

It does not trust the Eligibility Agent. For each result it re-reads the
SYNTHETIC record through the Patient Agent's tools and the original protocol
text, then checks:

  criterion_text        the criterion shown is exactly the protocol's text
  evidence_matches      every cited evidence ID exists with the same value, unit and date
  evidence_present      decided results cite evidence (no unsupported claims)
  numbers_supported     every number in the explanation comes from the protocol or the record
  dates_supported       every date is in the record or is a window start computed from it
  missing_threshold     no result was decided where the protocol omits a number or unit
  contradiction         no result was decided while the record has conflicting entries
  recomputed            numeric comparisons give the same answer when redone independently
  combination           a criterion made of parts agrees with its parts
  missing_confirmed     "missing" information really is missing
  decision_language     no wording that sounds like a final eligibility decision
  assumption            assumptions are listed for the reviewer (note)
  ai_suggestion         any Claude suggestion cites real evidence

Every outcome is written to the activity log.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any, Dict, List, Optional, Set

from trialmatch.activity_log import ActivityLog
from trialmatch.agents.eligibility_agent import ANCHOR_FIELDS, _years
from trialmatch.agents.eligibility_logic import as_number, compare, parse_date, window_start
from trialmatch.agents.patient_agent import PatientAgent
from trialmatch.agents.protocol_tools import source_numbers
from trialmatch.data_dictionary import field_label
from trialmatch.eligibility_models import CriterionEvaluation, ScreeningResult
from trialmatch.protocol_models import Criterion, Logic, ProtocolExtraction
from trialmatch.study_parser import StudyMetadata
from trialmatch.text_utils import numbers_in, unescape_markdown
from trialmatch.verification_models import (
    CriterionVerification,
    Outcome,
    Severity,
    VerificationCheck,
    VerificationReport,
)

AGENT_NAME = "Verification Agent"

DECIDED = {"MET", "NOT_MET", "PRESENT", "ABSENT"}
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_IDS = re.compile(r"\b(?:INC|EXC|REG)-[0-9A-Z]+(?:\.[a-z](?:\.\d+)?)?|\bSYN-\d{3}\b")
_DECISION_WORDS = re.compile(r"\b(in)?eligible\b|\bqualifies for (the )?(study|trial)\b|\bcan be enrolled\b",
                             re.IGNORECASE)


class VerificationAgent:
    def __init__(self, log: ActivityLog, patient_agent: PatientAgent) -> None:
        self.log = log
        self.patients = patient_agent

    def verify(self, screening: ScreeningResult, extraction: ProtocolExtraction,
               study: StudyMetadata) -> VerificationReport:
        self.pid = screening.patient_id
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.study = study
        criteria = {c.criterion_id: c for c in extraction.criteria}
        evaluations = {e.criterion_id: e for e in screening.evaluations}
        self.log.add(AGENT_NAME, "info", f"Verifying {len(screening.evaluations)} results for {self.pid} (SYNTHETIC).")

        general: List[VerificationCheck] = []
        expected = ["REG-AGE"] + [c.criterion_id for c in extraction.criteria]
        actual = [e.criterion_id for e in screening.evaluations]
        if actual != expected:
            general.append(_chk(None, "coverage", Severity.ERROR,
                                f"Results do not cover each criterion exactly once (missing: "
                                f"{sorted(set(expected) - set(actual))}, extra: {sorted(set(actual) - set(expected))})."))
        if not extraction.passed_verification:
            general.append(_chk(None, "protocol_extraction", Severity.ERROR,
                                "The criteria extraction did not pass its accuracy check."))
        if screening.record_label != "SYNTHETIC":
            general.append(_chk(None, "synthetic", Severity.ERROR, "The record is not labelled SYNTHETIC."))

        results = []
        for e in screening.evaluations:
            checks = self._check(e, criteria.get(e.criterion_id), criteria, evaluations)
            outcome = (Outcome.FAILED if any(c.severity == Severity.ERROR for c in checks) else
                       Outcome.VERIFIED_WITH_WARNINGS if any(c.severity == Severity.WARNING for c in checks) else
                       Outcome.VERIFIED)
            results.append(CriterionVerification(criterion_id=e.criterion_id, outcome=outcome, checks=checks))
            problems = [c for c in checks if c.severity != Severity.NOTE]
            self.log.add(AGENT_NAME, "verification",
                         f"{e.criterion_id}: {outcome.value}" + (
                             " — " + "; ".join(f"{c.check}: {c.message}" for c in problems) if problems else ""))

        report = VerificationReport(patient_id=self.pid, results=results, general=general,
                                    verified_at_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        counts = report.counts()
        self.log.add(AGENT_NAME, "verification",
                     f"Summary for {self.pid}: {counts['VERIFIED']} verified, {counts['VERIFIED_WITH_WARNINGS']} "
                     f"with warnings, {counts['FAILED']} failed" + (
                         f"; {len(general)} general problem(s)" if general else "") + ".")
        return report

    # ------------------------------------------------------------------ #

    def _get(self, field: str) -> Dict[str, Any]:
        if field not in self._cache:
            self._cache[field] = self.patients.call("get_field", {"patient_id": self.pid, "field": field})
        return self._cache[field]

    def _record_obs(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        field = evidence_id.split("/")[1] if evidence_id.count("/") == 2 else None
        if field is None:
            return None
        for o in self._get(field).get("observations", []):
            if o["evidence_id"] == evidence_id:
                return o
        return None

    def _check(self, e: CriterionEvaluation, c: Optional[Criterion], criteria: Dict[str, Criterion],
               evaluations: Dict[str, CriterionEvaluation]) -> List[VerificationCheck]:
        cid = e.criterion_id
        out: List[VerificationCheck] = []
        decided = e.status in DECIDED

        # 1. Criterion text ---------------------------------------------------
        if c is not None:
            source = (self.study.eligibility.full_text or "")[c.source.char_start:c.source.char_end]
            if source != c.original_text or e.criterion_text != unescape_markdown(c.original_text):
                out.append(_chk(cid, "criterion_text", Severity.ERROR,
                                "The criterion text does not match the protocol at its recorded location."))
        elif cid == "REG-AGE":
            lo = _years(self.study.eligibility.minimum_age)
            if lo is not None and str(lo) not in e.criterion_text:
                out.append(_chk(cid, "criterion_text", Severity.ERROR,
                                "The age limit shown does not match the registry minimum age."))
        else:
            out.append(_chk(cid, "criterion_text", Severity.ERROR, "This result has no matching protocol criterion."))

        # 2. Evidence exists and matches the record ---------------------------
        record_obs: List[Dict[str, Any]] = []
        for ev in e.evidence:
            obs = self._record_obs(ev.evidence_id)
            if obs is None:
                out.append(_chk(cid, "evidence_matches", Severity.ERROR,
                                f"Evidence {ev.evidence_id} is not in the record."))
                continue
            record_obs.append(obs)
            for key in ("value", "unit", "date", "end_date"):
                if getattr(ev, key) != obs.get(key):
                    out.append(_chk(cid, "evidence_matches", Severity.ERROR,
                                    f"Evidence {ev.evidence_id}: {key} '{getattr(ev, key)}' differs from the "
                                    f"record ('{obs.get(key)}')."))

        # 3. Decided results must cite evidence --------------------------------
        if decided and not e.evidence and e.method.value != "combined":
            out.append(_chk(cid, "evidence_present", Severity.ERROR,
                            f"The result is {e.status} but cites no evidence from the record."))

        # 4. Protocol gaps must not be decided ----------------------------------
        if c is not None:
            for t in c.thresholds:
                if t.value is None:
                    if decided:
                        out.append(_chk(cid, "missing_threshold", Severity.ERROR,
                                        "The protocol omits this number, but the result was decided anyway."))
                    else:
                        out.append(_chk(cid, "missing_threshold", Severity.NOTE,
                                        "The protocol omits the number; correctly left undecided."))
                elif t.unit is None and not t.relative_to:
                    if decided and e.method.value == "deterministic":
                        out.append(_chk(cid, "missing_threshold", Severity.ERROR,
                                        "The protocol states no unit, but the result was decided by comparison."))
                    elif not decided:
                        out.append(_chk(cid, "missing_threshold", Severity.NOTE,
                                        "The protocol states no unit; correctly not compared."))

        # 5. Contradictions must not be decided ---------------------------------
        fields = {ev.field for ev in e.evidence}
        if c is not None:
            fields |= {r.field for r in c.required_data}
        for f in sorted(fields):
            res = self._get(f)
            if res.get("consistency") == "CONFLICTING":
                if decided:
                    out.append(_chk(cid, "contradiction", Severity.ERROR,
                                    f"The record has conflicting entries for {field_label(f)}, but the result was "
                                    f"decided as {e.status}."))
                else:
                    out.append(_chk(cid, "contradiction", Severity.NOTE,
                                    f"Conflicting entries for {field_label(f)}; correctly left undecided."))

        # 6. Recompute numeric comparisons independently ------------------------
        if c is not None and c.thresholds and decided and e.method.value == "deterministic":
            expected = self._recompute(c)
            if expected is None:
                out.append(_chk(cid, "recomputed", Severity.WARNING,
                                "The comparison could not be recomputed independently."))
            elif expected != e.status:
                out.append(_chk(cid, "recomputed", Severity.ERROR,
                                f"Recomputing the comparison gives {expected}, not {e.status}."))

        # 7. Parts and whole agree ---------------------------------------------
        if c is not None and c.logic == Logic.ANY_OF:
            kids = [evaluations[k] for k in evaluations if evaluations[k].parent_id == cid]
            positive = "MET" if c.type.value == "INCLUSION" else "PRESENT"
            negative = "NOT_MET" if c.type.value == "INCLUSION" else "ABSENT"
            if e.status == positive and not any(k.status == positive for k in kids):
                out.append(_chk(cid, "combination", Severity.ERROR,
                                f"Marked {positive} but none of its parts is {positive}."))
            if kids and all(k.status == negative for k in kids) and e.status != negative:
                out.append(_chk(cid, "combination", Severity.ERROR,
                                f"Every part is {negative}, so this cannot be {e.status}."))

        # 8. Numbers and dates in the explanation are supported -----------------
        text = " ".join([e.reason] + e.comparisons)
        allowed_dates = self._allowed_dates(c, record_obs)
        for d in sorted(set(_ISO_DATE.findall(text))):
            if d not in allowed_dates:
                out.append(_chk(cid, "dates_supported", Severity.ERROR,
                                f"The date {d} is not in the record and is not a window start computed from it."))
        allowed = self._allowed_numbers(c, e, record_obs)
        for n in numbers_in(_IDS.sub(" ", _ISO_DATE.sub(" ", text))):
            if n not in allowed:
                out.append(_chk(cid, "numbers_supported", Severity.ERROR,
                                f"The number {n:g} is not in the protocol text or the record."))

        # 9. Missing information really is missing ------------------------------
        if e.status == "UNKNOWN" and not e.missing_information and e.method.value != "combined":
            out.append(_chk(cid, "missing_confirmed", Severity.ERROR,
                            "The result is UNKNOWN but does not say what information is missing."))
        for f in e.missing_information:
            res = self._get(f)
            if res["status"] == "FOUND":
                if f in fields:
                    out.append(_chk(cid, "missing_confirmed", Severity.NOTE,
                                    f"{field_label(f)} is recorded but not usable here (see the explanation)."))
                else:
                    out.append(_chk(cid, "missing_confirmed", Severity.ERROR,
                                    f"{field_label(f)} is listed as missing, but the record has it."))

        # 10. No final-decision language ----------------------------------------
        if _DECISION_WORDS.search(text):
            out.append(_chk(cid, "decision_language", Severity.ERROR,
                            "The explanation uses wording that sounds like a final eligibility decision."))

        # 11. Assumptions are surfaced ------------------------------------------
        for a in e.assumptions:
            out.append(_chk(cid, "assumption", Severity.NOTE, f"Assumption: {a}"))

        # 12. Claude suggestions cite real evidence -----------------------------
        if e.ai_suggestion:
            for i in e.ai_suggestion.evidence_ids:
                if self._record_obs(i) is None:
                    out.append(_chk(cid, "ai_suggestion", Severity.ERROR,
                                    f"Claude's suggestion cites {i}, which is not in the record."))
            if e.status != "NEEDS_HUMAN_REVIEW":
                out.append(_chk(cid, "ai_suggestion", Severity.ERROR,
                                "A Claude suggestion changed the status; suggestions must stay advisory."))
        return out

    # ------------------------------------------------------------------ #

    def _recompute(self, c: Criterion) -> Optional[str]:
        """Redo the numeric comparison from scratch. Returns the expected status, or None."""
        any_true, all_known = False, True
        for t in c.thresholds:
            if t.parameter is None or t.value is None:
                return None
            res = self._get(t.parameter)
            if res["status"] != "FOUND" or res.get("consistency") == "CONFLICTING":
                all_known = False
                continue
            obs = max(res["observations"], key=lambda o: o.get("date") or "")
            value = as_number(obs["value"])
            if value is None:
                return None
            if t.relative_to:
                uln = as_number((obs.get("attributes") or {}).get("uln"))
                if uln is None:
                    all_known = False
                    continue
                limit = t.value * uln
            else:
                if t.unit != obs.get("unit"):
                    return None
                limit = t.value
            any_true = any_true or compare(value, t.comparator, limit)
        holds = True if any_true else (False if all_known else None)
        if holds is None:
            return None
        if c.type.value == "INCLUSION":
            return "MET" if holds else "NOT_MET"
        return "PRESENT" if holds else "ABSENT"

    def _allowed_dates(self, c: Optional[Criterion], record_obs: List[Dict[str, Any]]) -> Set[str]:
        dates: Set[str] = set()
        for o in record_obs:
            for key in ("date", "end_date"):
                if o.get(key):
                    dates.add(o[key])
            if parse_date(o["value"]):
                dates.add(str(o["value"]))
        anchors = {}
        for name, field in ANCHOR_FIELDS.items():
            res = self._get(field)
            if res["status"] == "FOUND":
                anchors[name] = parse_date(res["observations"][0]["value"])
                dates.add(anchors[name].isoformat())
        for w in (c.time_windows if c else []):
            for a in anchors.values():
                dates.add(window_start(a, w.amount, w.unit).isoformat())
        return dates

    def _allowed_numbers(self, c: Optional[Criterion], e: CriterionEvaluation,
                         record_obs: List[Dict[str, Any]]) -> Set[float]:
        allowed: Set[float] = set()
        if c is not None:
            allowed |= set(source_numbers(c.original_text))
        else:
            for text in (self.study.eligibility.minimum_age, self.study.eligibility.maximum_age):
                allowed |= set(numbers_in(text or ""))
        for o in record_obs:
            v = as_number(o["value"])
            if v is not None:
                allowed.add(v)
            for text in (o["value"], o.get("unit"), o.get("note"), o.get("source")):
                if isinstance(text, (str, dict)):
                    allowed |= set(numbers_in(str(text)))
            uln = as_number((o.get("attributes") or {}).get("uln"))
            if uln is not None:
                allowed.add(uln)
                for t in (c.thresholds if c else []):
                    if t.relative_to and t.value is not None:
                        allowed.add(t.value * uln)
        return allowed


def _chk(cid: Optional[str], check: str, severity: Severity, message: str) -> VerificationCheck:
    return VerificationCheck(criterion_id=cid, check=check, severity=severity, message=message)
