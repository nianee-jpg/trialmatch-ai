"""TrialMatch AI — Streamlit dashboard (Milestone 1).

Run with:   streamlit run app.py

Milestone 1 scope:
  * Retrieve study NCT02260934 (CALIBRATE) from ClinicalTrials.gov API v2
  * Save the raw response for traceability
  * Display study metadata and the complete, original eligibility text

EDUCATIONAL SIMULATION ONLY — not a medical device, not a recruitment tool,
and never a final clinical eligibility determination.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Optional

import pandas as pd
import streamlit as st

from trialmatch import config
from trialmatch.clinicaltrials_client import (
    ClinicalTrialsAPIError,
    StudyRetrieval,
    fetch_study,
    get_study,
)
from trialmatch.activity_log import ActivityLog
from trialmatch.agents.eligibility_agent import EligibilityAgent
from trialmatch.agents.verification_agent import VerificationAgent
from trialmatch.agents.patient_agent import PatientAgent, PatientDataError, load_patients
from trialmatch.agents.protocol_agent import ProtocolAgent, make_claude_client
from trialmatch.data_dictionary import DATA_FIELDS, field_group, field_label
from trialmatch.text_utils import unescape_markdown
from trialmatch.eligibility_models import CriterionEvaluation, ScreeningResult
from trialmatch.report import build_html_report
from trialmatch.screening_summary import follow_up, headline, option_label, part_context, status_counts
from trialmatch.verification_models import CriterionVerification, Severity, VerificationReport
from trialmatch.ui_text import (
    CHECK_LABELS,
    EVENT_LABELS,
    METHOD_LABELS,
    OUTCOME_LABELS,
    RESULT_LABELS,
    RESULT_MEANINGS,
    TOOL_LABELS,
    flag_title,
    plain_log_detail,
)
from trialmatch.protocol_models import Criterion, ProtocolExtraction
from trialmatch.study_parser import StudyMetadata, parse_study

st.set_page_config(page_title="TrialMatch AI", page_icon="🧪", layout="wide")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_study(force_refresh: bool = False) -> None:
    """Fetch (or load from cache) the study and store it in session state."""
    try:
        if force_refresh:
            retrieval = fetch_study(config.STUDY_NCT_ID)
        else:
            # First page load: reuse a saved response if we have one, so the
            # app opens instantly and works offline.
            retrieval = get_study(config.STUDY_NCT_ID, prefer_cache=True)
        st.session_state["retrieval"] = retrieval
        st.session_state["study"] = parse_study(retrieval.data)
        st.session_state.pop("load_error", None)
    except ClinicalTrialsAPIError as exc:
        st.session_state["load_error"] = str(exc)


def show_value(label: str, value: object) -> None:
    """Render a label/value pair, making missing values explicit."""
    if value in (None, "", []):
        value = "— not reported in record —"
    elif isinstance(value, list):
        value = ", ".join(str(v) for v in value)
    st.markdown(f"**{label}:** {value}")


# --------------------------------------------------------------------------- #
# Page sections
# --------------------------------------------------------------------------- #

def render_disclaimer() -> None:
    st.warning(
        "**Educational simulation only.** TrialMatch AI is a portfolio project that "
        "replays a historical study's eligibility criteria against **SYNTHETIC** patient "
        "records. It is not a medical device, is not used for real recruitment, and never "
        "produces a final clinical eligibility determination. Do not enter real patient "
        "information.",
        icon="⚠️",
    )


def render_sidebar(retrieval: Optional[StudyRetrieval]) -> None:
    with st.sidebar:
        st.header("Data source")
        st.caption(f"Study: `{config.STUDY_NCT_ID}`")
        if st.button("🔄 Fetch latest from ClinicalTrials.gov", width="stretch"):
            with st.spinner("Contacting ClinicalTrials.gov…"):
                load_study(force_refresh=True)
            if "load_error" not in st.session_state:
                st.success("Fetched a fresh copy.")

        if retrieval is not None:
            st.subheader("Traceability")
            st.markdown(f"**Source:** {'saved copy (cache)' if retrieval.from_cache else 'live API call'}")
            st.markdown(f"**Retrieved (UTC):** {retrieval.retrieved_at}")
            st.markdown(f"**URL:** {retrieval.source_url}")
            st.markdown("**SHA-256:**")
            st.code(retrieval.sha256, language=None)
            st.markdown("**Raw file:**")
            st.code(str(retrieval.raw_path.relative_to(config.PROJECT_ROOT)), language=None)

        st.divider()
        st.subheader("Protocol Agent mode")
        if config.ANTHROPIC_API_KEY:
            st.markdown(f"🤖 **Claude** (`{config.ANTHROPIC_MODEL}`) refines interpretations after the rule-based pass.")
        else:
            st.markdown("⚙️ **Offline (rule-based)**. No API key needed. Add `ANTHROPIC_API_KEY` to `.env` "
                        "to enable optional Claude interpretations.")



def render_overview(study: StudyMetadata) -> None:
    st.subheader(study.official_title or study.brief_title or study.nct_id)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("NCT ID", study.nct_id)
    col2.metric("Status", study.overall_status or "—")
    col3.metric("Phase", ", ".join(study.phases) or "—")
    col4.metric(
        "Enrollment",
        f"{study.enrollment_count} ({(study.enrollment_type or '').lower()})"
        if study.enrollment_count is not None
        else "—",
    )

    left, right = st.columns(2)
    with left:
        show_value("Brief title", study.brief_title)
        show_value("Acronym", study.acronym)
        show_value("Lead sponsor", study.lead_sponsor)
        show_value("Collaborators", study.collaborators)
        show_value("Conditions", study.conditions)
        show_value("Keywords", study.keywords)
    with right:
        show_value("Study type", study.study_type)
        show_value("Allocation", study.allocation)
        show_value("Intervention model", study.intervention_model)
        show_value("Primary purpose", study.primary_purpose)
        show_value("Masking", study.masking)
        show_value("Start date", study.start_date)
        show_value("Primary completion", study.primary_completion_date)
        show_value("Study completion", study.completion_date)
        show_value("Last update posted", study.last_update_posted)
        show_value("Results posted", study.has_results)

    st.markdown("#### Brief summary")
    st.write(study.brief_summary or "— not reported in record —")
    if study.detailed_description:
        with st.expander("Detailed description"):
            st.write(study.detailed_description)


def render_eligibility(study: StudyMetadata) -> None:
    elig = study.eligibility

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sex", elig.sex or "—")
    col2.metric("Minimum age", elig.minimum_age or "—")
    col3.metric("Maximum age", elig.maximum_age or "—")
    col4.metric(
        "Healthy volunteers",
        "—" if elig.healthy_volunteers is None else ("Yes" if elig.healthy_volunteers else "No"),
    )

    if not elig.full_text:
        st.error("The study record contains no eligibility criteria text.")
        return

    st.caption(
        f"Source: `{elig.source_path}.eligibilityCriteria` — displayed exactly as "
        "returned by ClinicalTrials.gov, without paraphrasing."
    )

    view = st.radio(
        "View",
        ["Inclusion / Exclusion sections", "Complete original text"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # st.text (not st.markdown) is used throughout so symbols such as *, _, <, >
    # are shown exactly as written instead of being interpreted as formatting.
    if view == "Complete original text" or not elig.sections.split_succeeded:
        if not elig.sections.split_succeeded:
            st.info("Section headings could not be detected reliably, so the complete text is shown.")
        st.text(elig.full_text)
    else:
        left, right = st.columns(2)
        with left:
            st.markdown("#### ✅ Inclusion criteria")
            st.text(elig.sections.inclusion_text or "")
        with right:
            st.markdown("#### ⛔ Exclusion criteria")
            st.text(elig.sections.exclusion_text or "")

    st.download_button(
        "⬇️ Download original eligibility text (.txt)",
        data=elig.full_text,
        file_name=f"{study.nct_id}_eligibility_original.txt",
        mime="text/plain",
    )


def render_arms_outcomes(study: StudyMetadata) -> None:
    st.markdown("#### Arms")
    if study.arm_groups:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Arm": a.label,
                        "Type": a.type,
                        "Interventions": ", ".join(a.intervention_names),
                        "Description": a.description,
                    }
                    for a in study.arm_groups
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("— not reported in record —")

    st.markdown("#### Interventions")
    if study.interventions:
        st.dataframe(
            pd.DataFrame([i.model_dump() for i in study.interventions]),
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("— not reported in record —")

    for heading, outcomes in (("Primary outcomes", study.primary_outcomes), ("Secondary outcomes", study.secondary_outcomes)):
        st.markdown(f"#### {heading}")
        if outcomes:
            st.dataframe(
                pd.DataFrame([o.model_dump() for o in outcomes]),
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("— not reported in record —")


def render_raw(retrieval: StudyRetrieval) -> None:
    st.caption(
        "The exact bytes received from the API are saved on disk (see sidebar). "
        "Below is the same response, pretty-printed for reading."
    )
    st.download_button(
        "⬇️ Download raw API response (.json)",
        data=retrieval.raw_path.read_bytes(),
        file_name=retrieval.raw_path.name,
        mime="application/json",
    )
    st.json(retrieval.data, expanded=False)


def get_log() -> ActivityLog:
    """One activity log per browser session, shared by all agents."""
    if "activity_log" not in st.session_state:
        st.session_state["activity_log"] = ActivityLog()
    return st.session_state["activity_log"]


def run_protocol_agent(retrieval: StudyRetrieval, study: StudyMetadata) -> None:
    """Run the Protocol Agent and keep its result in session state."""
    log = get_log()
    agent = ProtocolAgent(log, client=make_claude_client())
    extraction = agent.run(
        nct_id=study.nct_id,
        eligibility_text=study.eligibility.full_text or "",
        raw_file=str(retrieval.raw_path.relative_to(config.PROJECT_ROOT)),
        raw_sha256=retrieval.sha256,
    )
    st.session_state["extraction"] = extraction
    st.session_state["extraction_sha"] = retrieval.sha256


def _rule_text(t) -> str:
    words = {"<": "below", "<=": "at or below", ">": "above", ">=": "at or above"}
    return words[t.comparator.value]


def _numbers_summary(c: Criterion) -> str:
    out = []
    for t in c.thresholds:
        value = "[missing]" if t.value is None else f"{t.value:,g}"
        unit = "× upper limit of normal" if t.relative_to else (t.unit or "(no unit)")
        name = field_label(t.parameter) if t.parameter else "?"
        out.append(f"{name} {_rule_text(t)} {value} {unit}")
    return "; ".join(out)


def _criteria_table(criteria: list) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID": c.criterion_id,
                "Type": c.type.value.title(),
                "Criterion": c.display_text,
                "Numbers to check": _numbers_summary(c),
                "Needs review because": "; ".join(dict.fromkeys(flag_title(f.code.value) for f in c.flags)),
                "Information needed": ", ".join(dict.fromkeys(field_label(r.field) for r in c.required_data)),
            }
            for c in criteria
        ]
    )


def render_criterion_detail(c: Criterion) -> None:
    kind = "Inclusion criterion" if c.type.value == "INCLUSION" else "Exclusion criterion"
    st.markdown(f"#### {c.criterion_id} · {kind}" + (f" (part of {c.parent_id})" if c.parent_id else ""))

    st.markdown("**What the protocol says**")
    st.text(c.display_text)
    st.markdown("**What to check**")
    st.info(c.normalized_interpretation)
    if c.interpretation_source == "claude":
        st.caption("Wording suggested by Claude and checked by the app for added numbers.")

    left, right = st.columns(2)
    with left:
        if c.thresholds:
            st.markdown("**Numbers to check**")
            st.dataframe(pd.DataFrame([{
                "Information": field_label(t.parameter) if t.parameter else "Unclear",
                "Rule": _rule_text(t),
                "Value": "missing from protocol" if t.value is None else f"{t.value:,g}",
                "Unit": "× upper limit of normal" if t.relative_to else (t.unit or "not given"),
                "Exactly equal counts?": "No" if t.comparator.is_strict else "Yes",
            } for t in c.thresholds]), hide_index=True, width="stretch")
        if c.time_windows:
            st.markdown("**Time limits**")
            st.dataframe(pd.DataFrame([{
                "Limit (as written)": unescape_markdown(w.source_text),
                "Counted from": w.anchor or "not stated in the protocol",
            } for w in c.time_windows]), hide_index=True, width="stretch")
        st.markdown("**Information needed from the patient record**")
        if c.required_data:
            for r in c.required_data:
                st.markdown(f"- {field_label(r.field)}" + (f" ({r.detail})" if r.detail else ""))
        else:
            st.caption("See the parts of this criterion listed above.")
    with right:
        st.markdown("**Why a person needs to review this**")
        if c.flags:
            for f in c.flags:
                st.warning(f"**{flag_title(f.code.value)}.** {f.message}")
        else:
            st.success("Nothing flagged. The wording is clear enough to check from the record.")

    with st.expander("Exact source text and location"):
        st.caption("Exactly as stored by ClinicalTrials.gov. The backslashes are formatting codes in the original.")
        st.code(c.original_text, language=None, wrap_lines=True)
        st.caption(
            f"Line {c.source.line_number} of the eligibility text (characters "
            f"{c.source.char_start}–{c.source.char_end})"
            + (f", file `{c.source.raw_file}`" if c.source.raw_file else "")
            + (f", SHA-256 `{c.source.raw_sha256[:16]}…`" if c.source.raw_sha256 else "") + "."
        )
        if c.flags:
            st.caption("Internal flag codes: " + ", ".join(f.code.value for f in c.flags))


def render_extracted_criteria(retrieval: StudyRetrieval, study: StudyMetadata) -> None:
    st.markdown(
        "The app has read the eligibility text and broken it into individual criteria. For each one it shows "
        "**what to check**, **which patient information is needed**, and **anything a person must decide**. "
        "No patient is evaluated on this page."
    )
    extraction = ensure_extraction(retrieval, study)
    stale = extraction is not None and st.session_state.get("extraction_sha") != retrieval.sha256

    label = "🔄 Run again" if extraction is not None else "▶️ Extract the criteria"
    if st.button(label, help="Re-reads the eligibility text from scratch."):
        with st.spinner("Reading the criteria…"):
            run_protocol_agent(retrieval, study)
        extraction, stale = st.session_state["extraction"], False

    if extraction is None:
        st.info("Click **Extract the criteria** to start.")
        return
    if stale:
        st.warning("The study record has changed since these criteria were extracted. Click **Run again**.")

    criteria = extraction.criteria
    flagged = [c for c in criteria if c.flags]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Criteria", len(criteria))
    m2.metric("Inclusion", sum(c.type.value == "INCLUSION" for c in criteria))
    m3.metric("Exclusion", sum(c.type.value == "EXCLUSION" for c in criteria))
    m4.metric("⚠️ Need a person", len(flagged))
    m5.metric("Accuracy check", "✅ Passed" if extraction.passed_verification else "❌ Problems")

    if extraction.passed_verification:
        st.success("**Accuracy check passed.** Every criterion matches the published text word for word, no text "
                   "was skipped, and no numbers were added that are not in the protocol.")
    else:
        st.error("**Accuracy check found problems.** Do not rely on these criteria until they are fixed.")
        st.dataframe(pd.DataFrame([{"Criterion": i.criterion_id or "", "Problem": i.message}
                                   for i in extraction.verification_issues]), hide_index=True)
    st.caption("Made by: " + ("the built-in rules (no AI)" if extraction.mode.startswith("offline")
                              else f"the built-in rules, with wording suggested by Claude (`{extraction.model}`)"))

    f1, f2 = st.columns([2, 3])
    which = f1.radio("Show", ["All", "Inclusion", "Exclusion"], horizontal=True)
    review_only = f2.toggle("Only criteria that need a person to review")
    reasons: list = []
    if review_only:
        all_reasons = sorted({flag_title(f.code.value) for c in criteria for f in c.flags})
        reasons = st.multiselect("Reason (leave empty for any)", all_reasons)

    shown = [
        c for c in criteria
        if (which == "All" or c.type.value.title() == which)
        and (not review_only or (c.flags and (not reasons or any(flag_title(f.code.value) in reasons for f in c.flags))))
    ]
    st.caption(f"Showing {len(shown)} of {len(criteria)} criteria.")
    st.dataframe(_criteria_table(shown), hide_index=True, width="stretch")

    if shown:
        chosen = st.selectbox(
            "Open a criterion for details",
            [c.criterion_id for c in shown],
            format_func=lambda i: f"{i} · {extraction.by_id(i).display_text[:80]}",
        )
        render_criterion_detail(extraction.by_id(chosen))

    st.download_button(
        "⬇️ Download all criteria (.json)",
        data=extraction.model_dump_json(indent=2),
        file_name=f"{extraction.nct_id}_criteria.json",
        mime="application/json",
        help="Machine-readable version with exact source locations, for traceability.",
    )


STATUS_TEXT = {
    "FOUND": "✅ In the record",
    "NOT_RECORDED": "⚠️ Missing",
    "CONFLICTING": "❗ Sources disagree",
}


def _fmt_value(value: object) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, dict):
        return ", ".join(f"{k.replace('_', '-')} {v}" for k, v in value.items())
    return str(value)


def _fmt_date(value: object) -> str:
    """'2015-03-02' -> '2 Mar 2015'."""
    try:
        d = value if isinstance(value, dt.date) else dt.date.fromisoformat(str(value))
        return f"{d.day} {d:%b %Y}"
    except ValueError:
        return str(value)


def _observations_table(observations: list) -> pd.DataFrame:
    """Turn Patient Agent evidence into a readable table."""
    rows = []
    for o in observations:
        attrs = o.get("attributes") or {}
        row = {
            "Result": _fmt_value(o["value"]),
            "Unit": o.get("unit") or "",
            "Date": _fmt_date(o["date"]) if o.get("date") else "",
            "Until": _fmt_date(o["end_date"]) if o.get("end_date") else "",
            "Source": o.get("source", ""),
            "Note": o.get("note") or "",
        }
        if "uln" in attrs:
            row["Upper limit of normal"] = attrs["uln"]
        if "route" in attrs:
            row["Route"] = attrs["route"]
        row["Reference"] = o["evidence_id"]
        rows.append(row)
    return pd.DataFrame(rows)


def render_lookup_result(result: dict, label: str) -> None:
    status = result.get("status")
    count = len(result.get("observations", []))
    if result.get("consistency") == "CONFLICTING":
        st.warning(
            f"**{label}: sources disagree.** The entries below do not match. The app will not "
            "choose between them; a person needs to check the original documents."
        )
    elif status == "FOUND":
        st.success(f"**{label}: found {count} {'entry' if count == 1 else 'entries'}** in the record.")
    elif status == "NOT_RECORDED" and result.get("drug_query"):
        st.warning(
            f"**No record of {result['drug_query']}.** That does not prove it was never given; "
            "it just is not in this record."
        )
    elif status == "NOT_RECORDED":
        st.warning(
            f"**{label}: not in this record.** The app will not guess or fill in a value. "
            "In a real screening, the coordinator would request it."
        )
    else:
        st.error(result.get("message", "The lookup could not be completed."))

    if count:
        st.dataframe(_observations_table(result["observations"]), hide_index=True, width="stretch")
    if result.get("negative_statements"):
        st.info("The record also contains this statement, which contradicts the entry above:")
        st.dataframe(_observations_table(result["negative_statements"]), hide_index=True, width="stretch")
    with st.expander("Technical details (exactly what the Patient Agent returned)"):
        st.json(result, expanded=True)


def ensure_extraction(retrieval: StudyRetrieval, study: StudyMetadata) -> Optional[ProtocolExtraction]:
    """The completeness check needs the criteria. In offline mode they take under a second, so run them."""
    if st.session_state.get("extraction") is None and not config.ANTHROPIC_API_KEY:
        run_protocol_agent(retrieval, study)
    return st.session_state.get("extraction")


def render_patients(retrieval: StudyRetrieval, study: StudyMetadata) -> None:
    try:
        patients = load_patients()
    except PatientDataError as exc:
        st.error(f"A patient file was rejected and not loaded: {exc}")
        return
    if not patients:
        st.info("No practice patients found in the `data/patients/` folder.")
        return

    st.warning(
        "**Practice patients only.** These five patients are made up (SYNTHETIC). Each was written to test "
        "specific study rules. No real patient information is used.",
        icon="🧪",
    )
    agent = PatientAgent(get_log(), patients)

    # 1. Choose a patient -----------------------------------------------------
    st.markdown("### 1. Choose a patient")
    pid = st.selectbox(
        "Patient",
        list(patients),
        format_func=lambda i: f"{patients[i].display_name.replace('Synthetic ', '')} · {i}",
        label_visibility="collapsed",
    )
    record = patients[pid]
    st.markdown(f"**What this case tests:** {record.scenario}")

    facts = agent.get_patient_overview(pid)["key_facts"]

    def fact(field: str, as_date: bool = False) -> str:
        res = facts[field]
        if res["status"] != "FOUND":
            return "Missing"
        v = res["observations"][0]["value"]
        return _fmt_date(v) if as_date else _fmt_value(v)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Age", fact("demographics.age"))
    c2.metric("Sex", fact("demographics.sex"))
    c3.metric("Screening date", fact("visits.screening_date", as_date=True))
    c4.metric("Planned enrollment", fact("visits.enrollment_date", as_date=True))

    # 2. Completeness -----------------------------------------------------------
    st.markdown("### 2. Is the record complete?")
    extraction = ensure_extraction(retrieval, study)
    if extraction is None:
        st.info("Run the Protocol Agent in the **Extracted criteria** tab first, so the app knows what the "
                "study asks about.")
    else:
        needed: dict = {}
        for c in extraction.criteria:
            for r in c.required_data:
                needed.setdefault(r.field, set()).add(c.criterion_id)
        rows = []
        for field, ids in needed.items():
            res = agent.get_field(pid, field)
            status = "CONFLICTING" if res.get("consistency") == "CONFLICTING" else res["status"]
            rows.append({"Information": field_label(field), "Category": field_group(field),
                         "Status": STATUS_TEXT.get(status, status), "Used by criteria": ", ".join(sorted(ids)),
                         "_status": status})
        df = pd.DataFrame(rows)
        df["_order"] = df["_status"].map({"CONFLICTING": 0, "NOT_RECORDED": 1, "FOUND": 2})
        df = df.sort_values(["_order", "Category", "Information"]).drop(columns="_order")
        n_found = int((df["_status"] == "FOUND").sum())
        n_missing = int((df["_status"] == "NOT_RECORDED").sum())
        n_conflict = int((df["_status"] == "CONFLICTING").sum())

        st.caption(f"The study's criteria ask about {len(df)} pieces of information. Here is what this record has.")
        a, b, c = st.columns(3)
        a.metric("✅ In the record", n_found)
        b.metric("⚠️ Missing", n_missing)
        c.metric("❗ Sources disagree", n_conflict)

        problems_only = st.toggle("Show only missing or disagreeing items", value=True)
        shown = df[df["_status"] != "FOUND"] if problems_only else df
        if shown.empty:
            st.success("Everything the criteria ask about is in this record, and no sources disagree.")
        else:
            st.dataframe(shown.drop(columns="_status"), hide_index=True, width="stretch")
        if "assessment.investigator_judgement" in needed:
            st.caption("*Investigator's clinical judgement* is always missing: it has to come from the study "
                       "doctor, not from the record.")
        st.caption("This only checks whether information is present. It does not decide eligibility.")

    # 3. Look something up --------------------------------------------------------
    st.markdown("### 3. Look something up")
    st.caption("The Patient Agent finds it in the record and shows exactly where each value came from. "
               "Every lookup is also recorded in the **Activity log** tab.")
    fields = sorted(DATA_FIELDS, key=lambda f: (field_group(f), field_label(f)))
    q1, q2 = st.columns([3, 2])
    field = q1.selectbox("What do you want to look up?", fields,
                         index=fields.index("labs.anc"),
                         format_func=lambda f: f"{field_group(f)} › {field_label(f)}")
    drug = ""
    if field == "medications.history":
        drug = q2.text_input("Search for a medicine (optional)", placeholder="e.g. rituximab")

    if st.button("🔎 Look up", type="primary"):
        if field == "medications.history":
            result = agent.call("get_medication_history", {"patient_id": pid, "drug": drug})
        else:
            result = agent.call("get_field", {"patient_id": pid, "field": field})
        st.session_state["patient_answer"] = ((pid, field, drug), result)
    answer = st.session_state.get("patient_answer")
    if answer and answer[0] == (pid, field, drug):
        render_lookup_result(answer[1], field_label(field))

    # 4. Full record ----------------------------------------------------------------
    st.markdown("### 4. Full record")
    with st.expander(f"See everything in {record.display_name.replace('Synthetic ', '')}'s record"):
        rows = []
        for f, observations in sorted(record.observations.items(), key=lambda kv: (field_group(kv[0]), field_label(kv[0]))):
            for i, o in enumerate(observations):
                rows.append({
                    "Category": field_group(f), "Information": field_label(f), "Result": _fmt_value(o.value),
                    "Unit": o.unit or "", "Date": _fmt_date(o.date) if o.date else "",
                    "Until": _fmt_date(o.end_date) if o.end_date else "", "Source": o.source,
                    "Note": o.note or "", "Reference": f"{pid}/{f}/{i}",
                })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.download_button(
            "⬇️ Download this practice record (.json)",
            data=record.model_dump_json(indent=2),
            file_name=f"{pid}_SYNTHETIC.json",
            mime="application/json",
        )


def run_screening(pid: str, patients: dict, extraction: ProtocolExtraction, study: StudyMetadata) -> ScreeningResult:
    """Eligibility Agent, then an independent Verification Agent pass."""
    log = get_log()
    patient_agent = PatientAgent(log, patients)
    result = EligibilityAgent(log, patient_agent, client=make_claude_client()).screen(pid, extraction, study)
    verification = VerificationAgent(log, patient_agent).verify(result, extraction, study)
    st.session_state.setdefault("screenings", {})[pid] = result
    st.session_state.setdefault("verifications", {})[pid] = verification
    return result


def render_verification_detail(v: CriterionVerification) -> None:
    st.markdown(f"**Accuracy check of this result: {OUTCOME_LABELS[v.outcome.value]}**")
    passed = sorted({CHECK_LABELS.get(c, c) for c in (
        "criterion_text", "evidence_matches", "numbers_supported", "dates_supported", "decision_language")}
        - {CHECK_LABELS.get(c.check, c.check) for c in v.problems})
    for c in v.problems:
        (st.error if c.severity == Severity.ERROR else st.warning)(
            f"**{CHECK_LABELS.get(c.check, c.check)}**: {c.message}")
    notes = [c for c in v.checks if c.severity == Severity.NOTE]
    for c in notes:
        st.caption(f"ℹ️ {CHECK_LABELS.get(c.check, c.check)}: {c.message}")
    st.caption("Checks passed: " + ", ".join(passed) + ".")


def _evidence_table(evaluation: CriterionEvaluation) -> pd.DataFrame:
    return pd.DataFrame([{
        "Information": field_label(e.field),
        "Result": _fmt_value(e.value),
        "Unit": e.unit or "",
        "Date": _fmt_date(e.date) if e.date else "",
        "Until": _fmt_date(e.end_date) if e.end_date else "",
        "Source": e.source or "",
        "Reference": e.evidence_id,
    } for e in evaluation.evidence])


def render_evaluation_detail(e: CriterionEvaluation, v: Optional[CriterionVerification] = None,
                             context: Optional[str] = None) -> None:
    kind = "Inclusion" if e.type == "INCLUSION" else "Exclusion"
    st.markdown(f"#### {e.criterion_id} · {kind}" + (f" (an option of {e.parent_id})" if e.parent_id else ""))
    st.text(e.criterion_text)
    if context:
        st.info(f"**What this means for {e.parent_id}:** {context}")
    box = {"MET": st.success, "ABSENT": st.success, "NOT_MET": st.error, "PRESENT": st.error}.get(e.status, st.warning)
    box(f"**{RESULT_LABELS[e.status]}**: {RESULT_MEANINGS[e.status]}\n\n{e.reason}")
    st.caption(f"How it was decided: {METHOD_LABELS.get(e.method.value, e.method.value)}")

    if e.comparisons:
        st.markdown("**Checks performed**")
        for line in e.comparisons:
            st.markdown(f"- {line}")
    st.markdown("**Evidence from the record**")
    if e.evidence:
        st.dataframe(_evidence_table(e), hide_index=True, width="stretch")
    else:
        st.caption("No record entries were used.")
    if e.missing_information:
        st.markdown("**Missing information**")
        for f in e.missing_information:
            st.markdown(f"- {field_label(f)}")
    if e.assumptions:
        st.markdown("**Assumptions the app made**")
        for a in e.assumptions:
            st.markdown(f"- {a}")
    if e.ai_suggestion:
        st.info(f"**Claude's advisory suggestion:** {e.ai_suggestion.suggested_status}. {e.ai_suggestion.rationale} "
                f"(cites {', '.join(e.ai_suggestion.evidence_ids)}). The status stays NEEDS REVIEW.")
    if v is not None:
        render_verification_detail(v)


def render_prescreening(retrieval: StudyRetrieval, study: StudyMetadata) -> None:
    st.warning(
        "**Pre-screening aid, not an eligibility decision.** The app checks each criterion against a "
        "SYNTHETIC record and lists what needs follow-up. Only the study team can decide eligibility.",
        icon="🩺",
    )
    try:
        patients = load_patients()
    except PatientDataError as exc:
        st.error(f"A patient file was rejected and not loaded: {exc}")
        return
    extraction = ensure_extraction(retrieval, study)
    if extraction is None:
        st.info("Extract the criteria first in the **Extracted criteria** tab.")
        return

    c1, c2 = st.columns([3, 1])
    pid = c1.selectbox("Patient to pre-screen", list(patients),
                       format_func=lambda i: f"{patients[i].display_name.replace('Synthetic ', '')} · {i}",
                       key="screen_patient")
    c2.markdown("&nbsp;")
    if c2.button("▶️ Run pre-screening", type="primary", width="stretch"):
        with st.spinner("Checking every criterion…"):
            run_screening(pid, patients, extraction, study)
    st.caption(f"**What this case tests:** {patients[pid].scenario}")

    result: Optional[ScreeningResult] = st.session_state.get("screenings", {}).get(pid)
    verification: Optional[VerificationReport] = st.session_state.get("verifications", {}).get(pid)
    if result is None or verification is None:
        st.info("Click **Run pre-screening** to check this patient against every criterion. "
                "The results are then checked independently by the Verification Agent.")
        return

    # Status headline ------------------------------------------------------------
    head = headline(result)
    box = {"rule_out": st.error, "review": st.warning, "unknown": st.warning, "none": st.info}[head.level]
    box(f"### {patients[pid].display_name.replace('Synthetic ', '')}: {head.title}\n\n"
        + "\n".join(f"- {d}" for d in head.details)
        + "\n\n*This is a pre-screening status, not an eligibility decision.*")

    vc = verification.counts()
    if verification.passed and not vc["VERIFIED_WITH_WARNINGS"]:
        st.caption(f"✅ **Accuracy check of the app's work passed:** all {len(verification.results)} results were "
                   "re-checked against the protocol and the record. This checks the app, not the patient.")
    elif verification.passed:
        st.warning(f"**Accuracy check of the app's work: {vc['VERIFIED_WITH_WARNINGS']} warning(s).** "
                   "See the Accuracy check column.")
    else:
        st.error(f"**Accuracy check of the app's work failed for {vc['FAILED']} result(s).** Do not rely on those "
                 "results; see the Accuracy check column and each result's details.")

    # Summary -----------------------------------------------------------------
    counts = status_counts(result)
    st.markdown("### Summary")
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Inclusion criteria** ({sum(counts['INCLUSION'].values())})")
        cols = st.columns(4)
        for col, s in zip(cols, ["MET", "NOT_MET", "UNKNOWN", "NEEDS_HUMAN_REVIEW"]):
            col.metric(RESULT_LABELS[s], counts["INCLUSION"].get(s, 0))
    with right:
        st.markdown(f"**Exclusion criteria** ({sum(counts['EXCLUSION'].values())})")
        cols = st.columns(4)
        for col, s in zip(cols, ["ABSENT", "PRESENT", "UNKNOWN", "NEEDS_HUMAN_REVIEW"]):
            col.metric(RESULT_LABELS[s], counts["EXCLUSION"].get(s, 0))
    st.caption("Counts are for the main criteria. Some criteria offer several options (for example INC-03.b.1, "
               "INC-03.b.2 and INC-03.b.3); only one option has to be met, so an option that is not met does not "
               "count against the patient when another option is met.")
    with st.expander("What do these results mean?"):
        for s, meaning in RESULT_MEANINGS.items():
            st.markdown(f"- **{RESULT_LABELS[s]}**: {meaning}")

    # Follow-up -----------------------------------------------------------------
    st.markdown("### What to follow up")
    todo = follow_up(result)
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(f"**⛔ May rule the patient out ({len(todo.rule_out)})**")
        if todo.rule_out:
            for e in todo.rule_out:
                st.markdown(f"- **{e.criterion_id}**: {e.reason}")
        else:
            st.caption("Nothing in the record clearly rules this patient out. That is not the same as eligible.")
    with f2:
        st.markdown(f"**❔ Information to request ({len(todo.unknown)})**")
        missing = sorted({field_label(m) for e in todo.unknown for m in e.missing_information})
        if missing:
            for m in missing:
                st.markdown(f"- {m}")
        else:
            st.caption("Nothing missing.")
    with f3:
        st.markdown(f"**👤 Decisions for the study team ({len(todo.review)})**")
        if todo.review:
            for e in todo.review:
                why = e.review_reasons[0] if e.review_reasons else "Needs judgement"
                st.markdown(f"- **{e.criterion_id}**: {why}")
        else:
            st.caption("None.")

    # Table ---------------------------------------------------------------------
    st.markdown("### Every criterion")
    t1, t2, t3 = st.columns([2, 2, 3])
    which = t1.radio("Show", ["All", "Inclusion", "Exclusion"], horizontal=True, key="screen_which")
    hide_parts = t2.toggle("Hide options (sub-criteria)", value=False)
    statuses = t3.multiselect("Result", list(RESULT_LABELS), format_func=lambda s: RESULT_LABELS[s])
    shown = [e for e in result.evaluations
             if (which == "All" or e.type.title() == which)
             and (not hide_parts or e.parent_id is None)
             and (not statuses or e.status in statuses)]

    def result_cell(e: CriterionEvaluation) -> str:
        tag = option_label(result, e)
        return RESULT_LABELS[e.status] + (f" ({tag})" if tag else "")

    def why_cell(e: CriterionEvaluation) -> str:
        context = part_context(result, e)
        return f"{context} {e.reason}" if context else e.reason

    table = pd.DataFrame([{
        "ID": ("↳ " if e.parent_id else "") + e.criterion_id,
        "Result": result_cell(e),
        "Criterion": e.criterion_text,
        "Why": why_cell(e),
        "How decided": METHOD_LABELS.get(e.method.value, e.method.value),
        "Accuracy check": OUTCOME_LABELS[verification.by_id(e.criterion_id).outcome.value],
    } for e in shown])
    st.caption(f"Showing {len(shown)} of {len(result.evaluations)}. Rows marked ↳ are options of the "
               "criterion above them.")
    if not table.empty:
        st.dataframe(table, hide_index=True, width="stretch")
        chosen = st.selectbox("Open a result to see its evidence", [e.criterion_id for e in shown],
                              format_func=lambda i: f"{i} · {result_cell(result.by_id(i))} · "
                                                    f"{result.by_id(i).criterion_text[:70]}")
        render_evaluation_detail(result.by_id(chosen), verification.by_id(chosen), part_context(result, result.by_id(chosen)))

    # Downloads -----------------------------------------------------------------
    st.markdown("### Download")
    csv = pd.DataFrame([{
        "patient_id": result.patient_id, "record_label": "SYNTHETIC", "criterion_id": e.criterion_id,
        "type": e.type, "status": e.status, "method": e.method.value, "reason": e.reason,
        "evidence_ids": "; ".join(x.evidence_id for x in e.evidence),
        "missing_information": "; ".join(e.missing_information), "assumptions": " | ".join(e.assumptions),
        "verification": verification.by_id(e.criterion_id).outcome.value,
    } for e in result.evaluations]).to_csv(index=False)
    report = build_html_report(
        study=study, source_url=retrieval.source_url, retrieved_at=retrieval.retrieved_at,
        raw_file=str(retrieval.raw_path.relative_to(config.PROJECT_ROOT)), raw_sha256=retrieval.sha256,
        extraction=extraction, record=patients[pid], screening=result, verification=verification, log=get_log(),
    )
    bundle = json.dumps({"screening": result.model_dump(mode="json"),
                         "verification": verification.model_dump(mode="json")}, indent=2)
    d1, d2, d3 = st.columns(3)
    d1.download_button("📄 Screening report (.html)", data=report, type="primary",
                       file_name=f"{result.patient_id}_SYNTHETIC_prescreening_report.html", mime="text/html",
                       help="Open in a browser; use File → Print to save as PDF.")
    d2.download_button("⬇️ Results table (.csv)", data=csv, file_name=f"{result.patient_id}_SYNTHETIC_prescreen.csv",
                       mime="text/csv")
    d3.download_button("⬇️ Results + verification (.json)", data=bundle,
                       file_name=f"{result.patient_id}_SYNTHETIC_prescreen.json", mime="application/json")
    st.caption("The report opens in any web browser. To make a PDF, open it and choose File → Print → Save as PDF.")
    st.caption(f"Screened {result.screened_at_utc} · {result.mode} · {result.disclaimer}")


def render_activity_log() -> None:
    st.markdown("Every step the agents took, in order. Use it to see exactly how a result was reached.")
    log = get_log()
    if not log.entries:
        st.info("Nothing yet. Open **Extracted criteria** or **Practice patients** and the steps will appear here.")
        return

    df = pd.DataFrame(log.as_rows())
    readable = pd.DataFrame({
        "Time (UTC)": df["timestamp"],
        "Agent": df["agent"],
        "What happened": df["event"].map(lambda e: EVENT_LABELS.get(e, e)),
        "Step": df["tool"].map(lambda t: TOOL_LABELS.get(t, t) if isinstance(t, str) else ""),
        "Details": df["detail"].map(plain_log_detail),
    })

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Steps recorded", len(df))
    m2.metric("Protocol Agent", int((df["agent"] == "Protocol Agent").sum()))
    m3.metric("Patient Agent", int((df["agent"] == "Patient Agent").sum()))
    m4.metric("Eligibility Agent", int((df["agent"] == "Eligibility Agent").sum()))
    m5.metric("Verification Agent", int((df["agent"] == "Verification Agent").sum()))

    f1, f2 = st.columns(2)
    agents = sorted(readable["Agent"].unique())
    chosen_agents = f1.multiselect("Agent", agents, default=agents)
    kinds = list(dict.fromkeys(readable["What happened"]))
    chosen_kinds = f2.multiselect("What happened", kinds, default=kinds)
    view = readable[readable["Agent"].isin(chosen_agents) & readable["What happened"].isin(chosen_kinds)]
    st.caption(f"Showing {len(view)} of {len(readable)} entries, oldest first.")
    st.dataframe(view, hide_index=True, width="stretch")

    with st.expander("Raw log (internal names)"):
        st.dataframe(df, hide_index=True, width="stretch")
    if st.button("🧹 Clear the log"):
        log.entries.clear()
        st.rerun()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    st.title("🧪 TrialMatch AI")
    st.caption("Clinical Trial Pre-Screening Assistant — educational simulation")
    render_disclaimer()

    if "retrieval" not in st.session_state and "load_error" not in st.session_state:
        with st.spinner("Loading study from ClinicalTrials.gov…"):
            load_study()

    retrieval: Optional[StudyRetrieval] = st.session_state.get("retrieval")
    study: Optional[StudyMetadata] = st.session_state.get("study")

    render_sidebar(retrieval)

    if "load_error" in st.session_state:
        st.error(f"Could not load the study: {st.session_state['load_error']}")
    if study is None or retrieval is None:
        st.info("Check your internet connection and click **Fetch latest from ClinicalTrials.gov**.")
        return

    tab_overview, tab_elig, tab_criteria, tab_patients, tab_screen, tab_log, tab_arms, tab_raw = st.tabs(
        ["📋 Study overview", "📝 Eligibility criteria", "🧩 Extracted criteria", "🧑‍⚕️ Practice patients",
         "🩺 Pre-screening", "🛰️ Activity log", "💊 Arms & outcomes", "🗂️ Raw API response"]
    )
    with tab_overview:
        render_overview(study)
    with tab_elig:
        render_eligibility(study)
    with tab_criteria:
        render_extracted_criteria(retrieval, study)
    with tab_patients:
        render_patients(retrieval, study)
    with tab_screen:
        render_prescreening(retrieval, study)
    with tab_log:
        render_activity_log()
    with tab_arms:
        render_arms_outcomes(study)
    with tab_raw:
        render_raw(retrieval)


main()
