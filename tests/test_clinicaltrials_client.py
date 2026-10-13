"""Tests for trialmatch.clinicaltrials_client (no real network calls)."""

import hashlib
import json

import pytest
import requests

from trialmatch.clinicaltrials_client import (
    ClinicalTrialsAPIError,
    build_study_url,
    fetch_study,
    get_study,
    load_latest_cached_study,
    validate_nct_id,
)


class FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200, url: str = "https://example.test/x"):
        self.content = content
        self.status_code = status_code
        self.url = url


class FakeSession:
    """Stands in for requests.Session so tests never touch the internet."""

    def __init__(self, response=None, exc=None):
        self.response, self.exc, self.calls = response, exc, []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.exc:
            raise self.exc
        return self.response


def test_validate_nct_id():
    assert validate_nct_id(" nct02260934 ") == "NCT02260934"
    for bad in ["NCT123", "02260934", "NCT0226093X", ""]:
        with pytest.raises(ValueError):
            validate_nct_id(bad)


def test_build_study_url():
    assert build_study_url("NCT02260934", "https://clinicaltrials.gov/api/v2/") == (
        "https://clinicaltrials.gov/api/v2/studies/NCT02260934"
    )


def test_fetch_saves_exact_raw_bytes_and_sidecar(tmp_path, mock_study_bytes):
    session = FakeSession(FakeResponse(mock_study_bytes))
    result = fetch_study("NCT02260934", raw_dir=tmp_path, session=session)

    assert result.raw_path.read_bytes() == mock_study_bytes
    assert result.sha256 == hashlib.sha256(mock_study_bytes).hexdigest()
    assert not result.from_cache

    sidecar = json.loads(result.raw_path.with_suffix(".meta.json").read_text())
    assert sidecar["sha256"] == result.sha256
    assert sidecar["http_status"] == 200
    assert sidecar["nct_id"] == "NCT02260934"


@pytest.mark.parametrize("status", [404, 500, 503])
def test_fetch_raises_on_http_error(tmp_path, status):
    session = FakeSession(FakeResponse(b"{}", status_code=status))
    with pytest.raises(ClinicalTrialsAPIError):
        fetch_study("NCT02260934", raw_dir=tmp_path, session=session)
    assert list(tmp_path.iterdir()) == []  # nothing saved on failure


def test_fetch_raises_on_invalid_json(tmp_path):
    with pytest.raises(ClinicalTrialsAPIError):
        fetch_study("NCT02260934", raw_dir=tmp_path, session=FakeSession(FakeResponse(b"<html>")))


def test_fetch_rejects_response_for_different_study(tmp_path, mock_study):
    mock_study["protocolSection"]["identificationModule"]["nctId"] = "NCT99999999"
    session = FakeSession(FakeResponse(json.dumps(mock_study).encode()))
    with pytest.raises(ClinicalTrialsAPIError, match="NCT99999999"):
        fetch_study("NCT02260934", raw_dir=tmp_path, session=session)


def test_fetch_wraps_network_errors(tmp_path):
    session = FakeSession(exc=requests.ConnectionError("offline"))
    with pytest.raises(ClinicalTrialsAPIError, match="Network error"):
        fetch_study("NCT02260934", raw_dir=tmp_path, session=session)


def test_cache_round_trip(tmp_path, mock_study_bytes):
    fetched = fetch_study("NCT02260934", raw_dir=tmp_path, session=FakeSession(FakeResponse(mock_study_bytes)))
    cached = load_latest_cached_study("NCT02260934", raw_dir=tmp_path)
    assert cached is not None and cached.from_cache
    assert cached.data == fetched.data
    assert cached.sha256 == fetched.sha256
    assert cached.retrieved_at == fetched.retrieved_at


def test_cache_empty_returns_none(tmp_path):
    assert load_latest_cached_study("NCT02260934", raw_dir=tmp_path) is None


def test_get_study_falls_back_to_cache_when_offline(tmp_path, mock_study_bytes, monkeypatch):
    fetch_study("NCT02260934", raw_dir=tmp_path, session=FakeSession(FakeResponse(mock_study_bytes)))

    def offline(*args, **kwargs):
        raise ClinicalTrialsAPIError("offline")

    monkeypatch.setattr("trialmatch.clinicaltrials_client.fetch_study", offline)
    result = get_study("NCT02260934", raw_dir=tmp_path)
    assert result.from_cache
