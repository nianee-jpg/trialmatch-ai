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
        st.subheader("Coming next")
        st.caption(
            "Protocol Agent · SYNTHETIC patients · Patient Agent · "
            "Eligibility Agent · Verification Agent · Screening report"
        )


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

    tab_overview, tab_elig, tab_arms, tab_raw = st.tabs(
        ["📋 Study overview", "📝 Eligibility criteria", "💊 Arms & outcomes", "🗂️ Raw API response"]
    )
    with tab_overview:
        render_overview(study)
    with tab_elig:
        render_eligibility(study)
    with tab_arms:
        render_arms_outcomes(study)
    with tab_raw:
        render_raw(retrieval)


main()
