"""Central configuration.

Values are read from environment variables (optionally loaded from a `.env`
file at the project root) with safe defaults, so Milestone 1 runs without any
configuration at all.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder that contains this package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load `.env` if present. Existing environment variables take precedence.
load_dotenv(PROJECT_ROOT / ".env")

CTGOV_BASE_URL: str = os.getenv("CTGOV_BASE_URL", "https://clinicaltrials.gov/api/v2").rstrip("/")
STUDY_NCT_ID: str = os.getenv("STUDY_NCT_ID", "NCT02260934")
CTGOV_TIMEOUT_SECONDS: float = float(os.getenv("CTGOV_TIMEOUT_SECONDS", "30"))

# Where raw API responses are saved for traceability.
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"

# Where SYNTHETIC patient records will live (Milestone 3).
PATIENT_DATA_DIR: Path = PROJECT_ROOT / "data" / "patients"

# Used from Milestone 2 onward. Not required for Milestone 1.
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY") or None
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
