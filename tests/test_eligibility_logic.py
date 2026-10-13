"""Unit tests for the Eligibility Agent's pure logic."""

import datetime as dt

import pytest

from trialmatch.agents.eligibility_logic import (
    D,
    biopsy_is_proliferative,
    classify_result,
    combine_all,
    combine_any,
    compare,
    describe_comparison,
    exposure_date,
    in_window,
    says_none,
    subtract_months,
    window_start,
)
from trialmatch.protocol_models import Comparator

day = dt.date


# --- Strict inequalities ---------------------------------------------------- #

@pytest.mark.parametrize("value, comparator, limit, expected", [
    (1499, Comparator.LT, 1500, True),
    (1500, Comparator.LT, 1500, False),   # strict: equal does not count
    (1500, Comparator.LE, 1500, True),
    (1.0, Comparator.GT, 1, False),       # UPCR exactly 1 is not > 1
    (1.01, Comparator.GT, 1, True),
    (80, Comparator.GE, 80, True),        # >= 2 x ULN: equal counts
])
def test_compare(value, comparator, limit, expected):
    assert compare(value, comparator, limit) is expected


def test_describe_comparison_explains_equality():
    text = describe_comparison("ANC", 1500, "/mm^3", Comparator.LT, 1500)
    assert text == "ANC 1,500 /mm^3 is not below 1,500 /mm^3 (exactly at the limit, which does not count)."
    assert "which counts" in describe_comparison("AST", 80, "U/L", Comparator.GE, 80)


# --- Date windows ----------------------------------------------------------- #

def test_calendar_months():
    assert subtract_months(day(2015, 3, 31), 1) == day(2015, 2, 28)
    assert subtract_months(day(2016, 3, 31), 1) == day(2016, 2, 29)  # leap year
    assert subtract_months(day(2015, 3, 2), 6) == day(2014, 9, 2)
    assert subtract_months(day(2015, 3, 2), 12) == day(2014, 3, 2)


def test_window_start_units():
    anchor = day(2015, 3, 2)
    assert window_start(anchor, 14, "days") == day(2015, 2, 16)
    assert window_start(anchor, 2, "weeks") == day(2015, 2, 16)
    assert window_start(anchor, 3, "months") == day(2014, 12, 2)
    assert window_start(anchor, 1, "years") == day(2014, 3, 2)


@pytest.mark.parametrize("event, expected", [
    (day(2015, 2, 17), D.TRUE),    # 13 days before: inside
    (day(2015, 2, 16), D.REVIEW),  # exactly 14 days: boundary day -> a person decides
    (day(2015, 2, 10), D.FALSE),   # 20 days before: outside
    (day(2015, 3, 10), D.TRUE),    # after the anchor (e.g. ongoing): inside
])
def test_in_window(event, expected):
    assert in_window(event, day(2015, 3, 2), 14, "days") == expected


def test_exposure_date():
    anchor = day(2015, 3, 2)
    assert exposure_date({"end_date": "2014-10-10", "note": None}, anchor) == day(2014, 10, 10)
    assert exposure_date({"end_date": None, "note": "3 g/day, ongoing"}, anchor) == anchor
    assert exposure_date({"end_date": None, "note": "dose unknown"}, anchor) is None  # never assumed


# --- Combining -------------------------------------------------------------- #

def test_combine_any():
    assert combine_any([D.FALSE, D.TRUE, D.UNKNOWN]) == D.TRUE
    assert combine_any([D.FALSE, D.FALSE]) == D.FALSE
    assert combine_any([D.FALSE, D.UNKNOWN]) == D.UNKNOWN
    assert combine_any([D.UNKNOWN, D.REVIEW]) == D.REVIEW


def test_combine_all():
    assert combine_all([D.TRUE, D.FALSE, D.REVIEW]) == D.FALSE
    assert combine_all([D.TRUE, D.TRUE]) == D.TRUE
    assert combine_all([D.TRUE, D.UNKNOWN]) == D.UNKNOWN
    assert combine_all([D.TRUE, D.REVIEW, D.UNKNOWN]) == D.REVIEW


# --- Reading documented results --------------------------------------------- #

@pytest.mark.parametrize("value, note, expected", [
    ("positive (1:640)", None, "positive"),
    ("negative serum hCG", None, "negative"),
    ("patient reports a positive home pregnancy test", None, "positive"),
    (210, "Positive (reference < 30 IU/mL)", "positive"),
    ("indeterminate", None, "indeterminate"),
    ("see report", None, None),
])
def test_classify_result(value, note, expected):
    assert classify_result(value, note) == expected


def test_says_none():
    assert says_none(False) is True and says_none(True) is False
    assert says_none("None significant") is True
    assert says_none("Hypertension") is None


@pytest.mark.parametrize("text, expected", [
    ("ISN/RPS Class IV-G (A)", True),
    ("ISN/RPS Class III (A)", True),
    ("ISN/RPS Class V + IV-G (A)", True),
    ("ISN/RPS Class V", False),          # pure membranous does not qualify
    ("ISN/RPS Class II", False),
    ("biopsy pending", None),
])
def test_biopsy_class(text, expected):
    assert biopsy_is_proliferative(text) is expected
