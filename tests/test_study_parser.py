"""Tests for trialmatch.study_parser (Milestone 1)."""

from trialmatch.study_parser import parse_study, split_eligibility_sections


def test_parse_basic_metadata(mock_study):
    study = parse_study(mock_study)
    assert study.nct_id == "NCT02260934"
    assert study.overall_status == "COMPLETED"
    assert study.phases == ["PHASE2"]
    assert study.enrollment_count == 10
    assert study.collaborators == ["MOCK Collaborator A", "MOCK Collaborator B"]
    assert study.arm_groups[0].intervention_names == ["Drug: MOCK"]
    assert study.eligibility.minimum_age == "18 Years"


def test_full_eligibility_text_is_unmodified(mock_study):
    original = mock_study["protocolSection"]["eligibilityModule"]["eligibilityCriteria"]
    study = parse_study(mock_study)
    assert study.eligibility.full_text == original  # byte-for-byte identical


def test_sections_are_exact_substrings(mock_study):
    study = parse_study(mock_study)
    sections = study.eligibility.sections
    assert sections.split_succeeded
    assert sections.inclusion_text in study.eligibility.full_text
    assert sections.exclusion_text in study.eligibility.full_text
    # Symbols that matter for thresholds must survive untouched.
    assert "> 1.0" in sections.inclusion_text
    assert "< 50" in sections.exclusion_text
    assert "Exclusion" not in sections.inclusion_text


def test_split_refuses_to_guess_without_headings():
    result = split_eligibility_sections("Some criteria without any headings")
    assert not result.split_succeeded
    assert result.inclusion_text is None and result.exclusion_text is None


def test_split_refuses_when_headings_out_of_order():
    text = "Exclusion Criteria:\n* a\nInclusion Criteria:\n* b"
    assert not split_eligibility_sections(text).split_succeeded


def test_missing_fields_stay_missing():
    """Absent data must be None/empty — never filled in with a default guess."""
    study = parse_study({"protocolSection": {"identificationModule": {"nctId": "NCT00000000"}}})
    assert study.enrollment_count is None
    assert study.eligibility.full_text is None
    assert study.eligibility.minimum_age is None
    assert study.phases == []
