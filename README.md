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
| 2 | Protocol Agent: structured Pydantic criteria, verification, activity log | ✅ Done |
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
│   ├── protocol_models.py          # Pydantic models for structured criteria
│   ├── data_dictionary.py          # patient data fields + rules mapping protocol wording to them
│   ├── text_utils.py               # Markdown un-escaping, number parsing
│   ├── activity_log.py             # visible log of agent tool calls and checks
│   └── agents/
│       ├── protocol_tools.py       # deterministic tools: segment, thresholds, windows, flags, verify
│       └── protocol_agent.py       # Protocol Agent (offline by default, optional Claude tool calling)
├── data/
│   ├── raw/                        # exact API responses + .meta.json sidecars (URL, time, SHA-256)
│   └── patients/                   # SYNTHETIC patient records (Milestone 3)
└── tests/
    ├── fixtures/mock_study_response.json   # MOCK data shaped like the API (not the real protocol)
    ├── test_clinicaltrials_client.py
    ├── test_study_parser.py
    ├── test_protocol_tools.py      # strict inequalities, missing data, date windows, prior treatment
    └── test_protocol_agent.py      # end-to-end on the real CALIBRATE text, incl. a fake Claude client
```

## How the Protocol Agent works (Milestone 2)

The agent turns the eligibility text into 45 structured criteria (5 inclusion and 35 exclusion
criteria, plus their sub-criteria). Each one has an ID such as `EXC-14` or `INC-03.b.1`, its type,
its **exact original text**, a readable version, a normalized interpretation, thresholds, time
windows, the patient data it needs, review flags, and a source reference (JSON path, character
offsets, raw file and SHA-256).

**It runs fully offline. No API key is needed.** The steps are deterministic tools, each one logged
in the **Agent activity** tab:

1. `segment_criteria`: split the text at numbered items and bullets, keeping exact character offsets.
2. `extract_thresholds`: find comparisons such as `\<1500/mm\^3` (strict `<`) or `\>=2 times the upper limit of normal`.
3. `extract_time_windows`: find "within 14 days before visit -1", "within the past 6 months", and similar.
4. `detect_flags`: mark investigator judgement, subjective terms, non-exhaustive lists and half-life rules.
5. `map_required_data`: list the patient data fields from `data_dictionary.py` that a criterion needs.
6. `verify_extraction`: check that every criterion's text matches the source exactly, that no source
   text was skipped, and that no number appears that is not in the source.

What it found in the published CALIBRATE record:

- **EXC-04** "Moderately severe anemia (Hgb \< mg/dL)" has **no number**. It is flagged
  `MISSING_THRESHOLD`, and no value is ever filled in.
- **INC-04** "UPCR \>1" states no unit (`UNIT_NOT_STATED`).
- **INC-03.b.1–3** (urinary sediment) are printed at the same indent as the biopsy options. A
  documented rule nests them under INC-03.b, flagged `STRUCTURE_INFERRED` so a person can confirm it.
- Many windows ("within the past 6 months") do not say which date they count from (`TIME_ANCHOR_NOT_STATED`).

**Optional Claude mode.** If you add `ANTHROPIC_API_KEY` to `.env`, the same rule-based pass runs
first. Claude then reads the criteria through tool calls (`list_criteria`, `get_criterion_details`)
and proposes plain-language interpretations with `submit_interpretation`. That tool **rejects** any
text containing a number that is not in the criterion's source text. Claude cannot change IDs,
original text or thresholds. If the API fails or declines, the rule-based results are kept.

## Install and run on macOS

Works with **Python 3.9 or newer**, including the Python 3.9 that ships with macOS.
Check with `python3 --version`.

Run these commands in **Terminal**, one block at a time. The blocks have no `#` comments
because macOS's default shell (zsh) does not treat `#` as a comment when you paste.

**1. Get the code.** First run `cd ~` so you don't clone it inside another copy.

```bash
cd ~
git clone https://github.com/nianee-jpg/trialmatch-ai.git
cd trialmatch-ai
git checkout claude/trialmatch-ai-prescreening-i28zkd
```

**2. Create and activate a virtual environment.** This is an isolated Python install for the
project. Your prompt will start with `(.venv)`.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install the dependencies and create your settings file.** No API key is needed yet.

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

**4. Run the tests.** You should see `56 passed`.

```bash
python -m pytest
```

**5. Start the app.**

```bash
streamlit run app.py
```

The first time, Streamlit asks for an email address. Leave it blank and press **Return**.

Streamlit opens <http://localhost:8501> in your browser. To stop it, press **Control + C** in Terminal.
Next time, run `cd ~/trialmatch-ai`, then `source .venv/bin/activate`, then `streamlit run app.py`.

### What you should see

- A warning banner saying this is an educational simulation.
- **Study overview** tab: title, status, phase, enrollment, sponsor, design, dates, summary.
- **Eligibility criteria** tab: sex/age limits and the eligibility text **exactly as published**,
  shown either as Inclusion/Exclusion sections or as the complete original text, with a download button.
- **Arms & outcomes** tab: arms, interventions, primary and secondary outcomes.
- **Extracted criteria** tab: click **Run Protocol Agent** to see every criterion, its thresholds,
  windows, required data and review flags, with a JSON download.
- **Agent activity** tab: every tool call the agent made, and the verification result.
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
- **`file or directory not found: #`** or **`cp: yet: Not a directory`**: a `#` comment was pasted
  into zsh. Re-run the command without the comment.
- **Port 8501 already in use**: `streamlit run app.py --server.port 8502`.

## Data and safety principles

- Protocol text is never paraphrased in storage. Every displayed criterion traces back to a saved
  raw response with a checksum.
- Missing values are shown as "not reported in record". They are never guessed.
- Patient records (from Milestone 3) are fictional and labelled **SYNTHETIC**. No PHI is used or transmitted.
