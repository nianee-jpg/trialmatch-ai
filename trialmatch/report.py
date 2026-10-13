"""Downloadable pre-screening report (self-contained HTML, printable to PDF).

The report repeats the SYNTHETIC label and the "not an eligibility decision"
disclaimer at the top and bottom, and records exactly which protocol version
and record entries every result came from.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from html import escape
from typing import Iterable, List, Optional

from trialmatch.activity_log import ActivityLog
from trialmatch.data_dictionary import field_label
from trialmatch.eligibility_models import ScreeningResult
from trialmatch.patient_models import PatientRecord
from trialmatch.protocol_models import ProtocolExtraction
from trialmatch.study_parser import StudyMetadata
from trialmatch.ui_text import CHECK_LABELS, OUTCOME_LABELS, RESULT_LABELS, RESULT_MEANINGS
from trialmatch.verification_models import Severity, VerificationReport

DISCLAIMER = ("EDUCATIONAL SIMULATION. This report uses a SYNTHETIC (fictional) patient record and a historical "
              "protocol. It is a pre-screening aid, not an eligibility decision, and not a medical device. "
              "Every result must be confirmed by the study team.")

_CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#1f2933;margin:32px auto;max-width:1100px;
     padding:0 16px;line-height:1.45;background:#fff}
h1{margin-bottom:4px} h2{border-bottom:2px solid #e4e7eb;padding-bottom:4px;margin-top:32px}
.banner{background:#fde8e8;border:2px solid #c81e1e;color:#7f1d1d;padding:12px 16px;border-radius:8px;font-weight:600}
.note{background:#f0f4f8;padding:10px 14px;border-radius:8px}
table{border-collapse:collapse;width:100%;margin:8px 0 16px;font-size:13px}
th,td{border:1px solid #d9e2ec;padding:6px 8px;text-align:left;vertical-align:top}
th{background:#f0f4f8} .muted{color:#627d98;font-size:12px} .grid{display:flex;gap:24px;flex-wrap:wrap}
.grid>div{flex:1;min-width:280px} code{font-size:12px} .fail{color:#c81e1e;font-weight:600}
.sub td:first-child{padding-left:22px}
@media print{body{margin:0} h2{page-break-after:avoid} tr{page-break-inside:avoid}}
"""


def _e(x: object) -> str:
    return escape("" if x is None else str(x))


def _table(headers: List[str], rows: Iterable[List[object]], raw_cols: Iterable[int] = (),
           row_class: Optional[List[str]] = None) -> str:
    raw = set(raw_cols)
    head = "".join(f"<th>{_e(h)}</th>" for h in headers) if any(headers) else ""
    body = []
    for i, row in enumerate(rows):
        cls = f' class="{row_class[i]}"' if row_class and row_class[i] else ""
        cells = "".join(f"<td>{c if j in raw else _e(c)}</td>" for j, c in enumerate(row))
        body.append(f"<tr{cls}>{cells}</tr>")
    thead = f"<thead><tr>{head}</tr></thead>" if head else ""
    return f"<table>{thead}<tbody>{''.join(body)}</tbody></table>"


def build_html_report(
    *,
    study: StudyMetadata,
    source_url: str,
    retrieved_at: str,
    raw_file: str,
    raw_sha256: str,
    extraction: ProtocolExtraction,
    record: PatientRecord,
    screening: ScreeningResult,
    verification: VerificationReport,
    log: Optional[ActivityLog] = None,
) -> str:
    generated = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    top = [e for e in screening.evaluations if e.parent_id is None]
    counts = {t: Counter(e.status for e in top if e.type == t) for t in ("INCLUSION", "EXCLUSION")}
    vcounts = verification.counts()
    parts: List[str] = []

    # Header ------------------------------------------------------------------
    parts.append(f"<h1>Pre-screening report: {_e(record.display_name)} ({_e(record.patient_id)})</h1>")
    parts.append(f'<p class="banner">SYNTHETIC RECORD · {_e(DISCLAIMER)}</p>')
    parts.append(_table(["", ""], [
        ["Study", f"{study.nct_id}: {study.official_title or study.brief_title}"],
        ["Protocol source", f"{source_url} (retrieved {retrieved_at})"],
        ["Raw response", f"{raw_file} · SHA-256 {raw_sha256}"],
        ["Criteria extraction", f"{len(extraction.criteria)} criteria · {extraction.mode} · accuracy check "
                                f"{'passed' if extraction.passed_verification else 'FAILED'}"],
        ["Patient", f"{record.patient_id} · {record.display_name} · SYNTHETIC"],
        ["What this case tests", record.scenario],
        ["Screened", f"{screening.screened_at_utc} · {screening.mode}"],
        ["Verified", f"{verification.verified_at_utc} · "
                     + ("all results passed" if verification.passed else "SOME RESULTS FAILED")],
        ["Report generated", generated],
    ]))

    # Summary ---------------------------------------------------------------
    parts.append("<h2>Summary</h2><div class='grid'>")
    parts.append("<div><b>Inclusion criteria</b>" + _table(["Result", "Count"], [
        [RESULT_LABELS[s], counts["INCLUSION"].get(s, 0)] for s in ("MET", "NOT_MET", "UNKNOWN", "NEEDS_HUMAN_REVIEW")
    ]) + "</div>")
    parts.append("<div><b>Exclusion criteria</b>" + _table(["Result", "Count"], [
        [RESULT_LABELS[s], counts["EXCLUSION"].get(s, 0)] for s in ("ABSENT", "PRESENT", "UNKNOWN", "NEEDS_HUMAN_REVIEW")
    ]) + "</div>")
    parts.append("<div><b>Verification</b>" + _table(["Outcome", "Count"], [
        [OUTCOME_LABELS[k], v] for k, v in vcounts.items()]) + "</div></div>")
    parts.append("<p class='muted'>Counts are for main criteria (sub-criteria such as INC-03.a are listed in full "
                 "below). There is deliberately no overall eligibility result.</p>")
    parts.append("<p class='muted'>" + " · ".join(f"<b>{_e(RESULT_LABELS[s])}</b>: {_e(m)}"
                                                  for s, m in RESULT_MEANINGS.items()) + "</p>")

    # Follow-up ---------------------------------------------------------------
    parts.append("<h2>What to follow up</h2><div class='grid'>")
    rule_out = [e for e in top if e.status in ("NOT_MET", "PRESENT")]
    unknown = [e for e in top if e.status == "UNKNOWN"]
    review = [e for e in top if e.status == "NEEDS_HUMAN_REVIEW"]
    parts.append("<div><b>⛔ May rule the patient out</b><ul>" + ("".join(
        f"<li><b>{_e(e.criterion_id)}</b>: {_e(e.reason)}</li>" for e in rule_out)
        or "<li>Nothing in the record clearly rules this patient out. That is not the same as eligible.</li>")
        + "</ul></div>")
    missing = sorted({field_label(m) for e in unknown for m in e.missing_information})
    parts.append("<div><b>❔ Information to request</b><ul>" + (
        "".join(f"<li>{_e(m)}</li>" for m in missing) or "<li>Nothing missing.</li>") + "</ul></div>")
    parts.append("<div><b>👤 Decisions for the study team</b><ul>" + ("".join(
        f"<li><b>{_e(e.criterion_id)}</b>: {_e(e.review_reasons[0] if e.review_reasons else e.reason)}</li>"
        for e in review) or "<li>None.</li>") + "</ul></div></div>")

    # Every criterion ---------------------------------------------------------
    parts.append("<h2>Every criterion</h2>")
    rows, classes = [], []
    for e in screening.evaluations:
        v = verification.by_id(e.criterion_id)
        problems = "; ".join(f"{CHECK_LABELS.get(c.check, c.check)}: {c.message}" for c in v.problems)
        rows.append([
            e.criterion_id, e.type.title(), e.criterion_text, RESULT_LABELS[e.status], e.reason,
            ", ".join(x.evidence_id for x in e.evidence) or "—",
            OUTCOME_LABELS[v.outcome.value] + (f" ({problems})" if problems else ""),
        ])
        classes.append("sub" if e.parent_id else "")
    parts.append(_table(["ID", "Type", "Criterion (protocol wording)", "Result", "Why", "Evidence", "Verification"],
                        rows, row_class=classes))

    # Evidence --------------------------------------------------------------
    parts.append("<h2>Evidence cited</h2><p class='muted'>Every record entry used above, from the SYNTHETIC "
                 "record. References have the form patient/field/entry.</p>")
    seen = {}
    for e in screening.evaluations:
        for x in e.evidence:
            seen.setdefault(x.evidence_id, (x, []))[1].append(e.criterion_id)
    parts.append(_table(["Reference", "Information", "Result", "Unit", "Date", "Until", "Source", "Used by"], [
        [i, field_label(x.field), x.value, x.unit or "", x.date or "", x.end_date or "", x.source or "",
         ", ".join(dict.fromkeys(ids))] for i, (x, ids) in sorted(seen.items())]))

    # Assumptions and verification notes ----------------------------------------
    parts.append("<h2>Assumptions and verification notes</h2>")
    notes = [[c.criterion_id or "—", CHECK_LABELS.get(c.check, c.check), c.severity.value, c.message]
             for r in verification.results for c in r.checks]
    notes += [[c.criterion_id or "general", CHECK_LABELS.get(c.check, c.check), c.severity.value, c.message]
              for c in verification.general]
    parts.append(_table(["Criterion", "Check", "Level", "Detail"], notes) if notes else "<p>None.</p>")
    errors = [n for n in notes if n[2] == Severity.ERROR.value]
    if errors:
        parts.append(f"<p class='fail'>{len(errors)} verification error(s): do not rely on the affected results.</p>")

    # Activity --------------------------------------------------------------
    if log is not None and log.entries:
        by_agent = Counter(e.agent for e in log.entries)
        parts.append("<h2>Agent activity</h2>")
        parts.append(_table(["Agent", "Logged steps"], sorted(by_agent.items())))
        verif_lines = [e for e in log.entries if e.agent == "Verification Agent" and e.event == "verification"
                       and record.patient_id in e.detail and "Summary" in e.detail]
        if verif_lines:
            parts.append(f"<p class='note'>{_e(verif_lines[-1].detail)}</p>")
        parts.append("<p class='muted'>The full step-by-step log is in the app's Activity log tab.</p>")

    parts.append(f'<p class="banner">SYNTHETIC RECORD · {_e(DISCLAIMER)}</p>')
    body = "\n".join(parts)
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Pre-screening report {_e(record.patient_id)} (SYNTHETIC)</title><style>{_CSS}</style></head>"
            f"<body>{body}</body></html>")
