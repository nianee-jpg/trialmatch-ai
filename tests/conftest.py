"""Shared pytest fixtures."""

import json
import sys
from pathlib import Path

import pytest

# Make the project root importable when running `pytest` from any folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_study_bytes() -> bytes:
    return (FIXTURES / "mock_study_response.json").read_bytes()


@pytest.fixture
def mock_study(mock_study_bytes) -> dict:
    return json.loads(mock_study_bytes)


RAW_DIR = PROJECT_ROOT / "data" / "raw"


@pytest.fixture(scope="session")
def calibrate_eligibility_text() -> str:
    """Eligibility text from the committed ClinicalTrials.gov response for NCT02260934."""
    files = sorted(p for p in RAW_DIR.glob("NCT02260934_*.json") if not p.name.endswith(".meta.json"))
    if not files:
        pytest.skip("No saved NCT02260934 response in data/raw (run the app once to fetch it).")
    data = json.loads(files[-1].read_bytes())
    return data["protocolSection"]["eligibilityModule"]["eligibilityCriteria"]
