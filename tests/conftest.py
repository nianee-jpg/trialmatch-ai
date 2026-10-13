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
