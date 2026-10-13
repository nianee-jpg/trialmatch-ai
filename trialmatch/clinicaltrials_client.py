"""ClinicalTrials.gov API v2 client.

Responsibilities:
  1. Download a single study record (e.g. NCT02260934) from the public API.
  2. Save the *exact bytes* the server returned, plus a small metadata
     "sidecar" file (URL, timestamp, HTTP status, SHA-256 checksum), so every
     downstream conclusion can be traced back to a specific API response.
  3. Re-load the most recently saved response when the network is unavailable.

API documentation: https://clinicaltrials.gov/data-api/api
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from trialmatch import config

# NCT IDs are "NCT" followed by exactly 8 digits.
_NCT_ID_PATTERN = re.compile(r"^NCT\d{8}$")

USER_AGENT = "TrialMatchAI/0.1 (educational portfolio project)"


class ClinicalTrialsAPIError(RuntimeError):
    """Raised when a study cannot be retrieved or the response is unusable."""


@dataclass(frozen=True)
class StudyRetrieval:
    """The result of fetching (or loading) one study record.

    Attributes:
        nct_id: The study identifier that was requested.
        data: The parsed JSON response (a dict).
        source_url: The exact URL the data came from.
        retrieved_at: When the API was called (UTC, ISO 8601).
        sha256: Checksum of the raw response bytes, for traceability.
        raw_path: Where the raw response is stored on disk.
        from_cache: True if loaded from disk instead of a live API call.
    """

    nct_id: str
    data: dict[str, Any]
    source_url: str
    retrieved_at: str
    sha256: str
    raw_path: Path
    from_cache: bool


def validate_nct_id(nct_id: str) -> str:
    """Return a normalised NCT ID, or raise ValueError if it is malformed."""
    cleaned = nct_id.strip().upper()
    if not _NCT_ID_PATTERN.match(cleaned):
        raise ValueError(f"Invalid NCT ID {nct_id!r}: expected 'NCT' followed by 8 digits.")
    return cleaned


def build_study_url(nct_id: str, base_url: str = config.CTGOV_BASE_URL) -> str:
    """Build the API v2 URL for a single study."""
    return f"{base_url.rstrip('/')}/studies/{validate_nct_id(nct_id)}"


def fetch_study(
    nct_id: str = config.STUDY_NCT_ID,
    *,
    raw_dir: Path = config.RAW_DATA_DIR,
    base_url: str = config.CTGOV_BASE_URL,
    timeout: float = config.CTGOV_TIMEOUT_SECONDS,
    session: Optional[requests.Session] = None,
) -> StudyRetrieval:
    """Download a study from ClinicalTrials.gov and save the raw response.

    Raises:
        ClinicalTrialsAPIError: on network errors, non-200 responses, invalid
            JSON, or a response that does not look like the requested study.
    """
    nct_id = validate_nct_id(nct_id)
    url = build_study_url(nct_id, base_url)
    http = session or requests.Session()

    try:
        response = http.get(
            url,
            params={"format": "json"},
            headers={"Accept": "application/json", "User-Agent": USER_AGENT},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise ClinicalTrialsAPIError(f"Network error contacting ClinicalTrials.gov: {exc}") from exc

    if response.status_code == 404:
        raise ClinicalTrialsAPIError(f"Study {nct_id} was not found on ClinicalTrials.gov (HTTP 404).")
    if response.status_code != 200:
        raise ClinicalTrialsAPIError(
            f"ClinicalTrials.gov returned HTTP {response.status_code} for {url}."
        )

    raw_bytes = response.content
    try:
        data = json.loads(raw_bytes)
    except ValueError as exc:
        raise ClinicalTrialsAPIError("ClinicalTrials.gov returned a response that is not valid JSON.") from exc

    _check_response_matches(data, nct_id)

    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    raw_path = _save_raw_response(
        raw_dir=raw_dir,
        nct_id=nct_id,
        raw_bytes=raw_bytes,
        sidecar={
            "nct_id": nct_id,
            "source_url": response.url or url,
            "retrieved_at_utc": retrieved_at,
            "http_status": response.status_code,
            "sha256": sha256,
            "byte_length": len(raw_bytes),
        },
        retrieved_at=retrieved_at,
    )

    return StudyRetrieval(
        nct_id=nct_id,
        data=data,
        source_url=response.url or url,
        retrieved_at=retrieved_at,
        sha256=sha256,
        raw_path=raw_path,
        from_cache=False,
    )


def load_latest_cached_study(
    nct_id: str = config.STUDY_NCT_ID,
    *,
    raw_dir: Path = config.RAW_DATA_DIR,
) -> Optional[StudyRetrieval]:
    """Load the most recently saved raw response for a study, if one exists."""
    nct_id = validate_nct_id(nct_id)
    # File names embed a sortable UTC timestamp, so the last one is the newest.
    candidates = sorted(p for p in raw_dir.glob(f"{nct_id}_*.json") if not p.name.endswith(".meta.json"))
    if not candidates:
        return None

    raw_path = candidates[-1]
    raw_bytes = raw_path.read_bytes()
    data = json.loads(raw_bytes)
    _check_response_matches(data, nct_id)

    sidecar_path = raw_path.with_suffix(".meta.json")
    sidecar: dict[str, Any] = {}
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    return StudyRetrieval(
        nct_id=nct_id,
        data=data,
        source_url=sidecar.get("source_url", build_study_url(nct_id)),
        retrieved_at=sidecar.get("retrieved_at_utc", "unknown"),
        # Recompute the checksum so tampering/corruption is detectable.
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        raw_path=raw_path,
        from_cache=True,
    )


def get_study(
    nct_id: str = config.STUDY_NCT_ID,
    *,
    raw_dir: Path = config.RAW_DATA_DIR,
    prefer_cache: bool = False,
) -> StudyRetrieval:
    """Convenience wrapper used by the app.

    - prefer_cache=True: use a saved response if one exists, else fetch live.
    - prefer_cache=False: fetch live; if that fails, fall back to the cache.
    """
    if prefer_cache:
        cached = load_latest_cached_study(nct_id, raw_dir=raw_dir)
        if cached is not None:
            return cached
        return fetch_study(nct_id, raw_dir=raw_dir)

    try:
        return fetch_study(nct_id, raw_dir=raw_dir)
    except ClinicalTrialsAPIError:
        cached = load_latest_cached_study(nct_id, raw_dir=raw_dir)
        if cached is None:
            raise
        return cached


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _check_response_matches(data: Any, nct_id: str) -> None:
    """Make sure the JSON is a study record for the NCT ID we asked for."""
    if not isinstance(data, dict) or "protocolSection" not in data:
        raise ClinicalTrialsAPIError("Response does not contain a 'protocolSection'; unexpected format.")
    returned_id = data["protocolSection"].get("identificationModule", {}).get("nctId")
    if returned_id != nct_id:
        raise ClinicalTrialsAPIError(f"Requested {nct_id} but the response describes {returned_id!r}.")


def _save_raw_response(
    *,
    raw_dir: Path,
    nct_id: str,
    raw_bytes: bytes,
    sidecar: dict[str, Any],
    retrieved_at: str,
) -> Path:
    """Write the untouched response bytes plus a `.meta.json` sidecar file."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = retrieved_at.replace("-", "").replace(":", "")  # e.g. 20261013T162300Z
    raw_path = raw_dir / f"{nct_id}_{stamp}.json"
    raw_path.write_bytes(raw_bytes)  # exact bytes — no re-formatting
    raw_path.with_suffix(".meta.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return raw_path
