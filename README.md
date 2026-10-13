# TrialMatch AI — Clinical Trial Pre-Screening Assistant

An AI-powered clinical trial pre-screening **simulation** that uses multi-agent workflows to review
**SYNTHETIC** patient records against a real, historical study protocol:
**NCT02260934 — CALIBRATE** (Rituximab and Belimumab for Lupus Nephritis).

> ⚠️ **Educational simulation only.** This is not a medical device or a recruitment tool. It never
> makes a final clinical eligibility determination, and it must only be used with synthetic data.
> Do not enter real patient information or PHI.

## Project status

| Milestone | Scope | Status |
|---|---|---|
| 1 | ClinicalTrials.gov retrieval, raw-response traceability, metadata and eligibility display, basic Streamlit UI | ✅ Done |
| 2 | Protocol Agent: structured Pydantic criteria | ⏳ Planned |
| 3 | 5 SYNTHETIC patient profiles + Patient Agent tools | ⏳ Planned |
| 4 | Eligibility Agent (deterministic thresholds + Claude for ambiguous text) | ⏳ Planned |
| 5 | Verification Agent, activity logs, screening report | ⏳ Planned |

## Folder structure

```
trialmatch-ai/
├── app.py                          # Streamlit dashboard (entry point)
├── requirements.txt
├── .env.example                    # copy to .env (never commit .env)
├── trialmatch/
│   ├── config.py                   # settings loaded from .env, with defaults
│   ├── clinicaltrials_client.py    # API v2 retrieval + raw-response saving/caching
│   ├── study_parser.py             # Pydantic models for study metadata + eligibility
│   └── agents/                     # Protocol / Patient / Eligibility / Verification (later)
├── data/
│   ├── raw/                        # exact API responses + .meta.json sidecars (URL, time, SHA-256)
│   └── patients/                   # SYNTHETIC patient records (Milestone 3)
└── tests/
    ├── fixtures/mock_study_response.json   # MOCK data shaped like the API (not the real protocol)
    ├── test_clinicaltrials_client.py
    └── test_study_parser.py
```

## Install and run on macOS

You need **Python 3.10 or newer**. Check with `python3 --version`. If it is missing or older,
install it from <https://www.python.org/downloads/macos/> or with Homebrew (`brew install python`).

Open **Terminal** and run:

```bash
# 1. Get the code and go into the project folder
git clone https://github.com/nianee-jpg/trialmatch-ai.git
cd trialmatch-ai
git checkout claude/trialmatch-ai-prescreening-i28zkd   # until this branch is merged

# 2. Create and activate a virtual environment (an isolated Python install for this project)
python3 -m venv .venv
source .venv/bin/activate          # your prompt now starts with (.venv)

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Create your settings file (no key is needed for Milestone 1)
cp .env.example .env

# 5. Run the tests
pytest

# 6. Start the app
streamlit run app.py
```

Streamlit opens <http://localhost:8501> in your browser. To stop it, press **Control + C** in Terminal.
Next time, you only need `cd trialmatch-ai`, `source .venv/bin/activate`, and `streamlit run app.py`.

### What you should see

- A warning banner saying this is an educational simulation.
- **Study overview** tab: title, status, phase, enrollment, sponsor, design, dates, summary.
- **Eligibility criteria** tab: sex/age limits and the eligibility text **exactly as published**,
  shown either as Inclusion/Exclusion sections or as the complete original text, with a download button.
- **Arms & outcomes** tab: arms, interventions, primary and secondary outcomes.
- **Raw API response** tab: the full JSON plus a download of the exact saved bytes.
- Sidebar: **Fetch latest from ClinicalTrials.gov** and traceability details (URL, UTC timestamp,
  SHA-256 checksum, raw file path).

On first launch the app calls the API and saves the response to `data/raw/`. Later launches reuse
the newest saved copy, so the app also works offline. Click the sidebar button to fetch a fresh copy.

### Troubleshooting

- **`command not found: streamlit`**: the virtual environment is not active. Run `source .venv/bin/activate`.
- **`Could not load the study: Network error…`**: check your internet connection or VPN and try the
  sidebar button again.
- **SSL certificate errors** with python.org Python: run
  `/Applications/Python\ 3.*/Install\ Certificates.command` once.
- **Port 8501 already in use**: `streamlit run app.py --server.port 8502`.

## Data and safety principles

- Protocol text is never paraphrased in storage. Every displayed criterion traces back to a saved
  raw response with a checksum.
- Missing values are shown as "not reported in record". They are never guessed.
- Patient records (from Milestone 3) are fictional and labelled **SYNTHETIC**. No PHI is used or transmitted.
