"""Unit tests for the Protocol Agent's deterministic tools.

Strings use the same Markdown escaping as ClinicalTrials.gov ("\\<" etc.).
"""

import pytest

from trialmatch.agents.protocol_tools import (
    extract_thresholds,
    extract_time_windows,
    map_required_data,
    segment_criteria,
)
from trialmatch.protocol_models import Comparator, CriterionType, FlagCode
from trialmatch.text_utils import numbers_in, unescape_markdown


def codes(flags):
    return {f.code for f in flags}


# --- Markdown escapes ----------------------------------------------------- #

def test_unescape_markdown():
    assert unescape_markdown(r"count \<1500/mm\^3") == "count <1500/mm^3"
    assert unescape_markdown(r"\[AST\] \>=2") == "[AST] >=2"


def test_numbers_in_handles_thousands_separator():
    assert numbers_in("platelets <50,000 and eGFR 1.73") == [50000.0, 1.73]


# --- Strict vs non-strict inequalities ------------------------------------ #

@pytest.mark.parametrize(
    "text, comparator, strict",
    [
        (r"absolute neutrophil count \<1500/mm\^3", Comparator.LT, True),
        (r"(UPCR) \>1 at study entry", Comparator.GT, True),
        (r"results that are \>=2 times the upper limit of normal", Comparator.GE, False),
        ("platelets <= 100 mg/dL", Comparator.LE, False),
    ],
)
def test_comparator_and_strictness(text, comparator, strict):
    thresholds, _ = extract_thresholds(text)
    assert thresholds, text
    assert thresholds[0].comparator == comparator
    assert thresholds[0].comparator.is_strict is strict


def test_threshold_value_unit_and_parameter():
    (t,), flags = extract_thresholds(r"Thrombocytopenia (platelets \<50,000/mm\^3).")
    assert (t.parameter, t.value, t.unit) == ("labs.platelets", 50000.0, "/mm^3")
    assert t.source_text == r"\<50,000/mm\^3"
    assert not flags


def test_two_thresholds_in_one_criterion_are_not_mixed_up():
    thresholds, _ = extract_thresholds(
        r"Moderately severe hypogammaglobulinemia (IgG \<450 mg/dL) or Immunoglobulin A (IgA) \<10mg/dL.")
    assert [(t.parameter, t.value) for t in thresholds] == [("labs.igg", 450.0), ("labs.iga", 10.0)]


def test_parameter_after_number():
    (t,), _ = extract_thresholds(r"\>5 RBC/hpf in the absence of menses and infection;")
    assert (t.parameter, t.value, t.unit) == ("urinalysis.rbc_per_hpf", 5.0, "/hpf")


# --- Missing data: never invent a threshold ------------------------------- #

def test_missing_threshold_is_flagged_not_guessed():
    (t,), flags = extract_thresholds(r"Moderately severe anemia (Hgb \< mg/dL).")
    assert t.value is None
    assert t.parameter == "labs.hemoglobin"
    assert not t.complete
    assert FlagCode.MISSING_THRESHOLD in codes(flags)


def test_missing_unit_is_flagged_not_assumed():
    (t,), flags = extract_thresholds(r"Urine protein-to-creatinine ratio (UPCR) \>1 at study entry")
    assert t.unit is None
    assert FlagCode.UNIT_NOT_STATED in codes(flags)


def test_relative_threshold_creates_one_entry_per_lab():
    thresholds, flags = extract_thresholds(
        r"Liver function test \[aspartate aminotransferase (AST), alanine aminotransferase (ALT), or "
        r"alkaline phosphatase\] results that are \>=2 times the upper limit of normal.")
    assert {t.parameter for t in thresholds} == {"labs.ast", "labs.alt", "labs.alkaline_phosphatase"}
    assert all(t.relative_to == "upper limit of normal" and t.value == 2.0 for t in thresholds)
    assert FlagCode.RELATIVE_THRESHOLD in codes(flags)


def test_unidentified_measurement_is_flagged():
    (t,), flags = extract_thresholds(r"Some score \>7 mg/dL")
    assert t.parameter is None
    assert FlagCode.THRESHOLD_PARAMETER_UNKNOWN in codes(flags)


def test_text_without_comparators_has_no_thresholds():
    thresholds, flags = extract_thresholds("History of transplantation.")
    assert thresholds == [] and flags == []


def test_word_number_threshold():
    (t,), _ = extract_thresholds("three or more courses of systemic corticosteroids within the previous 12 months")
    assert (t.comparator, t.value, t.unit) == (Comparator.GE, 3.0, "courses")


# --- Date windows --------------------------------------------------------- #

@pytest.mark.parametrize(
    "text, amount, unit, anchor",
    [
        ("any time within 14 days before visit -1.", 14, "days", "visit -1"),
        ("live-attenuated vaccine within 3 months of study enrollment.", 3, "months", "study enrollment"),
        ("Use of cyclophosphamide within the past 6 months.", 6, "months", None),
        ("history of substance abuse within the past year.", 1, "years", None),
        ("biopsy documentation within the last 12 months of ISN/RPS", 12, "months", None),
    ],
)
def test_time_windows(text, amount, unit, anchor):
    (w,), flags = extract_time_windows(text)
    assert (w.amount, w.unit, w.anchor) == (amount, unit, anchor)
    assert (FlagCode.TIME_ANCHOR_NOT_STATED in codes(flags)) is (anchor is None)


def test_multiple_windows_in_one_criterion():
    windows, _ = extract_time_windows(
        "History of suicidal thoughts within the past 2 months or suicidal behavior within the past 6 months")
    assert [(w.amount, w.unit) for w in windows] == [(2, "months"), (6, "months")]


# --- Prior-treatment rules ------------------------------------------------ #

def test_prior_treatment_drugs_are_each_required():
    required, _ = map_required_data("Prior treatment with rituximab, belimumab, atacicept, or other biologic B cell therapy.")
    details = {r.detail for r in required if r.field == "medications.history"}
    assert {"rituximab", "belimumab", "atacicept", "other biologic B-cell therapy"} <= details


def test_new_onset_rule_requires_both_prior_drugs():
    required, _ = map_required_data(
        "New onset lupus nephritis, defined as lupus nephritis for which the participant has not yet been "
        "treated with either mycophenolate mofetil or cyclophosphamide.")
    details = {r.detail for r in required if r.field == "medications.history"}
    assert details == {"mycophenolate mofetil", "cyclophosphamide"}


def test_unmapped_criterion_is_flagged():
    required, flags = map_required_data("Something the dictionary does not know about.")
    assert required == []
    assert FlagCode.REQUIRED_DATA_UNMAPPED in codes(flags)


# --- Segmentation --------------------------------------------------------- #

SMALL = (
    "Inclusion Criteria:\n\n"
    "1. First.\n"
    "2. Defined by either of the following:\n\n"
    "   * Option A.\n"
    "   * Option B, defined as any one of the following:\n"
    "   * B1;\n"
    "   * B2.\n\n"
    "Exclusion Criteria:\n\n"
    "1. Bad thing.\n"
    "   continued on a second line.\n"
)


def test_segment_ids_offsets_and_nesting():
    segs = {s.criterion_id: s for s in segment_criteria(SMALL)}
    assert list(segs) == ["INC-01", "INC-02", "INC-02.a", "INC-02.b", "INC-02.b.1", "INC-02.b.2", "EXC-01"]
    for s in segs.values():
        assert SMALL[s.char_start:s.char_end].strip() == SMALL[s.char_start:s.char_end]
    assert SMALL[segs["INC-02.b.1"].char_start:segs["INC-02.b.1"].char_end] == "B1;"
    assert segs["INC-02.b.1"].parent_id == "INC-02.b" and segs["INC-02.b.1"].structure_note
    assert segs["INC-02.a"].structure_note is None
    assert segs["EXC-01"].type == CriterionType.EXCLUSION
    # Continuation line is included in the criterion's span.
    assert SMALL[segs["EXC-01"].char_start:segs["EXC-01"].char_end].endswith("second line.")
