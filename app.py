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
from trialmatch.agents.patient_agent import PatientAgent, PatientDataError, load_patients
from trialmatch.agents.protocol_agent import ProtocolAgent, make_claude_client
from trialmatch.data_dictionary import DATA_FIELDS, field_group, field_label
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

        st.subheader("Coming next")
        st.caption("Eligibility Agent · Verification Agent · Screening report")


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


def _criteria_table(criteria: list) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID": c.criterion_id,
                "Type": c.type.value.title(),
                "Criterion (readable)": c.display_text,
                "Thresholds": "; ".join(
                    f"{t.parameter or '?'} {t.comparator.value} "
                    f"{'MISSING' if t.value is None else f'{t.value:g}'}"
                    f"{' x ULN' if t.relative_to else (' ' + t.unit if t.unit else '')}"
                    for t in c.thresholds
                ),
                "Time windows": "; ".join(f"{w.amount} {w.unit} / {w.anchor or 'no anchor'}" for w in c.time_windows),
                "Required data": ", ".join(sorted({r.field for r in c.required_data})),
                "Flags": ", ".join(sorted({f.code.value for f in c.flags})),
            }
            for c in criteria
        ]
    )


def render_criterion_detail(c: Criterion) -> None:
    st.markdown(f"#### {c.criterion_id} · {c.type.value.title()}"
                + (f" · sub-criterion of {c.parent_id}" if c.parent_id else ""))
    left, right = st.columns(2)
    with left:
        st.markdown("**Exact original text** (as stored by ClinicalTrials.gov, Markdown escapes included)")
        st.code(c.original_text, language=None, wrap_lines=True)
        st.markdown("**Readable text**")
        st.text(c.display_text)
        st.markdown(f"**Normalized interpretation** _(source: {c.interpretation_source})_")
        st.info(c.normalized_interpretation)
    with right:
        st.markdown("**Source reference**")
        st.caption(
            f"`{c.source.json_path}` · characters {c.source.char_start}–{c.source.char_end} · "
            f"line {c.source.line_number}"
            + (f" · file `{c.source.raw_file}`" if c.source.raw_file else "")
        )
        if c.thresholds:
            st.markdown("**Thresholds**")
            st.dataframe(pd.DataFrame([t.model_dump(mode="json") for t in c.thresholds]), hide_index=True)
        if c.time_windows:
            st.markdown("**Time windows**")
            st.dataframe(pd.DataFrame([w.model_dump(mode="json") for w in c.time_windows]), hide_index=True)
        st.markdown("**Required patient data**")
        if c.required_data:
            for r in c.required_data:
                st.markdown(f"- `{r.field}`: {DATA_FIELDS.get(r.field, '')}" + (f" ({r.detail})" if r.detail else ""))
        else:
            st.caption("None mapped (see sub-criteria or flags).")
        if c.flags:
            st.markdown("**Flags for human review**")
            for f in c.flags:
                st.warning(f"**{f.code.value}**: {f.message}")


def render_extracted_criteria(retrieval: StudyRetrieval, study: StudyMetadata) -> None:
    st.caption(
        "The Protocol Agent splits the eligibility text into individual criteria, extracts thresholds and "
        "time windows, and flags anything it cannot read reliably. It does not evaluate any patient."
    )
    extraction: Optional[ProtocolExtraction] = st.session_state.get("extraction")
    stale = extraction is not None and st.session_state.get("extraction_sha") != retrieval.sha256

    if st.button("▶️ Run Protocol Agent", type="primary"):
        with st.spinner("Extracting criteria…"):
            run_protocol_agent(retrieval, study)
        extraction, stale = st.session_state["extraction"], False

    if extraction is None:
        st.info("Click **Run Protocol Agent** to extract the criteria.")
        return
    if stale:
        st.warning("The study record changed since this extraction. Run the agent again.")

    criteria = extraction.criteria
    flagged = [c for c in criteria if c.flags]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Criteria", len(criteria))
    col2.metric("Inclusion", sum(c.type.value == "INCLUSION" for c in criteria))
    col3.metric("Exclusion", sum(c.type.value == "EXCLUSION" for c in criteria))
    col4.metric("Flagged for review", len(flagged))
    col5.metric("Verification", "PASS" if extraction.passed_verification else "FAIL")
    st.caption(f"Mode: **{extraction.mode}**" + (f" · model `{extraction.model}`" if extraction.model else ""))

    if extraction.verification_issues:
        st.error("Verification found problems. Do not rely on this extraction until they are resolved.")
        st.dataframe(pd.DataFrame([i.model_dump() for i in extraction.verification_issues]), hide_index=True)
    else:
        st.success(
            "Verification passed: every criterion's text matches the source exactly, all source text is "
            "covered, no numbers were introduced, and missing thresholds are flagged."
        )

    f1, f2 = st.columns(2)
    type_filter = f1.multiselect("Type", ["Inclusion", "Exclusion"], default=["Inclusion", "Exclusion"])
    all_flags = sorted({f.code.value for c in criteria for f in c.flags})
    flag_filter = f2.multiselect("Only criteria with flag", all_flags)

    shown = [
        c for c in criteria
        if c.type.value.title() in type_filter
        and (not flag_filter or any(f.code.value in flag_filter for f in c.flags))
    ]
    st.dataframe(_criteria_table(shown), hide_index=True, width="stretch")

    if shown:
        chosen = st.selectbox("Criterion detail", [c.criterion_id for c in shown])
        render_criterion_detail(extraction.by_id(chosen))

    st.download_button(
        "⬇️ Download structured criteria (.json)",
        data=extraction.model_dump_json(indent=2),
        file_name=f"{extraction.nct_id}_criteria.json",
        mime="application/json",
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
               "Every lookup is also recorded in the **Agent activity** tab.")
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


def render_activity_log() -> None:
    log = get_log()
    if not log.entries:
        st.info("No agent activity yet. Run the Protocol Agent, or ask the Patient Agent a question.")
        return
    if st.button("🧹 Clear log"):
        log.entries.clear()
        st.rerun()
    df = pd.DataFrame(log.as_rows())
    agents = sorted(df["agent"].unique())
    chosen_agents = st.multiselect("Agents", agents, default=agents)
    df = df[df["agent"].isin(chosen_agents)]
    events = sorted(df["event"].unique())
    chosen = st.multiselect("Event types", events, default=events)
    tools_used = sorted(t for t in df["tool"].dropna().unique())
    tool_filter = st.multiselect("Only these tools", tools_used)
    view = df[df["event"].isin(chosen)]
    if tool_filter:
        view = view[view["tool"].isin(tool_filter)]
    st.caption(f"{len(view)} of {len(df)} entries")
    st.dataframe(view, hide_index=True, width="stretch")



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

    tab_overview, tab_elig, tab_criteria, tab_patients, tab_log, tab_arms, tab_raw = st.tabs(
        ["📋 Study overview", "📝 Eligibility criteria", "🧩 Extracted criteria", "🧑‍⚕️ Practice patients",
         "🛰️ Agent activity", "💊 Arms & outcomes", "🗂️ Raw API response"]
    )
    with tab_overview:
        render_overview(study)
    with tab_elig:
        render_eligibility(study)
    with tab_criteria:
        render_extracted_criteria(retrieval, study)
    with tab_patients:
        render_patients(retrieval, study)
    with tab_log:
        render_activity_log()
    with tab_arms:
        render_arms_outcomes(study)
    with tab_raw:
        render_raw(retrieval)


main()
