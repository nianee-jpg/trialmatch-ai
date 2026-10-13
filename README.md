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
| 3 | 5 SYNTHETIC patient profiles + Patient Agent tools | ✅ Done |
| 4 | Eligibility Agent (deterministic thresholds; ambiguous text to a person, optional Claude suggestions) | ✅ Done |
| 5 | Verification Agent, activity logs, downloadable screening report | ✅ Done |

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
│   ├── ui_text.py                  # plain-language wording for codes shown in the dashboard
│   ├── patient_models.py           # Pydantic models for SYNTHETIC records (+ identifier checks)
│   ├── eligibility_models.py       # per-criterion results (MET/NOT_MET/..., PRESENT/ABSENT/...)
│   ├── verification_models.py      # verification checks and outcomes
│   ├── report.py                   # self-contained HTML screening report
│   ├── screening_summary.py        # pre-screening status headline and option explanations
│   └── agents/
│       ├── protocol_tools.py       # deterministic tools: segment, thresholds, windows, flags, verify
│       ├── protocol_agent.py       # Protocol Agent (offline by default, optional Claude tool calling)
│       ├── patient_agent.py        # Patient Agent: field-lookup tools over one synthetic record
│       ├── eligibility_logic.py    # comparisons, calendar windows, combining results (pure functions)
│       ├── eligibility_agent.py    # Eligibility Agent + the CALIBRATE criterion rules
│       ├── verification_agent.py   # Verification Agent: independent re-check of every result
│       └── claude_loop.py          # shared optional Claude tool-calling loop
├── data/
│   ├── raw/                        # exact API responses + .meta.json sidecars (URL, time, SHA-256)
│   └── patients/                   # SYN-001 … SYN-005: fictional records, labelled SYNTHETIC
└── tests/
    ├── fixtures/mock_study_response.json   # MOCK data shaped like the API (not the real protocol)
    ├── test_clinicaltrials_client.py
    ├── test_study_parser.py
    ├── test_protocol_tools.py      # strict inequalities, missing data, date windows, prior treatment
    ├── test_protocol_agent.py      # end-to-end on the real CALIBRATE text, incl. a fake Claude client
    ├── test_patient_agent.py       # record safety checks, NOT_RECORDED, conflicts, scenario dates
    ├── test_eligibility_logic.py   # strict inequalities, calendar months, boundary days, combining
    ├── test_eligibility_agent.py   # expected results for each synthetic patient
    ├── test_verification_agent.py  # honest results pass; 12 kinds of tampering are caught
    ├── test_screening_summary.py   # status headline and 'option not needed' explanations
    └── test_ui_text.py             # every code shown on screen has a plain translation
```

## How the Protocol Agent works (Milestone 2)

The agent turns the eligibility text into 45 structured criteria (5 inclusion and 35 exclusion
criteria, plus their sub-criteria). Each one has an ID such as `EXC-14` or `INC-03.b.1`, its type,
its **exact original text**, a readable version, a normalized interpretation, thresholds, time
windows, the patient data it needs, review flags, and a source reference (JSON path, character
offsets, raw file and SHA-256).

**It runs fully offline. No API key is needed.** The steps are deterministic tools, each one logged
in the **Activity log** tab:

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

## Synthetic patients and the Patient Agent (Milestone 3)

`data/patients/` holds five **fictional** lupus nephritis records. Each one is designed to test
something specific:

| ID | Designed to test |
|---|---|
| SYN-001 | A complete record with values far from every threshold. It still needs human review (e.g. the anemia criterion has no published cutoff) |
| SYN-002 | Strict-inequality boundaries: age 18, ANC 1500, platelets 50,000, IgG 450, IgA 10, eGFR 20, UPCR 1.0, AST exactly 2 × ULN |
| SYN-003 | Missing data (no hemoglobin, IgA, QuantiFERON, HIV, anti-dsDNA …), a spot-urine UPCR, and date windows just outside their limits |
| SYN-004 | Prior-treatment rules (rituximab in 2012, recent cyclophosphamide, IVIG) and a live vaccine just outside 3 months |
| SYN-005 | Contradictions: two same-day ANC values on either side of 1500, a negative hCG vs a reported positive home test, "no prior biologics" vs a belimumab pharmacy record |

Each field holds a **list of observations** (value, unit, date, fictional source document, note).
Nothing is merged or averaged. A record is refused unless it is labelled `SYNTHETIC`, has an ID such as
`SYN-001`, uses only data-dictionary fields, and contains no identifying keys (name, MRN, date of
birth, address …).

The **Patient Agent** is the only component that reads patient data. Its tools (`get_field`,
`get_medication_history`, `list_recorded_fields`, `get_patient_overview`, `list_patients`) return
each observation with an evidence ID such as `SYN-003/labs.upcr/0`, report `NOT_RECORDED` instead of
guessing, and report `CONFLICTING` when same-day values disagree. Every call appears in the
**Activity log** tab. The tools also have JSON schemas, so the Eligibility Agent (Milestone 4)
can call them.

## Pre-screening with the Eligibility Agent (Milestone 4)

The Eligibility Agent checks every criterion separately for one synthetic patient, getting all
data through the Patient Agent's logged tools.

- Inclusion results: **MET**, **NOT_MET**, **UNKNOWN** (information missing), **NEEDS_HUMAN_REVIEW**
- Exclusion results: **PRESENT**, **ABSENT**, **UNKNOWN**, **NEEDS_HUMAN_REVIEW**
- There is **no overall eligibility decision**. Instead, each patient gets a **pre-screening status**
  (*Possible rule-out found*, *Needs review by the study team*, *Information missing*, or *No problems
  found by the app*) and a follow-up list: what may rule the patient out, what information to
  request, and what the study team must decide.
- Some criteria offer **options** (e.g. INC-03.b.1–3: "any one of the following"). An option that
  is not met is labelled *option · not needed* when another option already satisfies the criterion,
  so it is never counted against the patient.

How results are reached:

| Situation | What the app does |
|---|---|
| Clear number with a unit (ANC, platelets, eGFR, IgG/IgA) | Compares exactly; "below 1,500" excludes 1,500 |
| Limit relative to normal range (AST/ALT/ALP ≥ 2 × ULN) | Uses the lab's own upper limit recorded with the result |
| Number or unit missing from the protocol (EXC-04, INC-04) | NEEDS_HUMAN_REVIEW; never fills in a value or assumes a unit |
| Units in the record differ from the protocol | NEEDS_HUMAN_REVIEW; never converts |
| Time limit with a stated start ("within 14 days before visit -1") | Exact calendar arithmetic |
| Time limit without a stated start ("within the past 6 months") | Checked from screening **and** planned enrollment; if they disagree, NEEDS_HUMAN_REVIEW |
| Date exactly on the first day of a window | NEEDS_HUMAN_REVIEW (the protocol does not say if it counts) |
| Information missing from the record | UNKNOWN, naming what to request |
| Sources disagree | NEEDS_HUMAN_REVIEW, showing every conflicting entry |
| Investigator's opinion, "severe", "consistent with" … | NEEDS_HUMAN_REVIEW |

The registry's structured age limit ("18 Years", no maximum) is checked as `REG-AGE`.

**Optional Claude mode:** with an API key, Claude can add an *advisory* suggestion to criteria that
need a person. It must look up evidence with the Patient Agent's tools and cite evidence IDs it
actually retrieved. The status stays NEEDS_HUMAN_REVIEW.

## Verification and the screening report (Milestone 5)

After every pre-screening, the **Verification Agent** re-checks each result on its own. It does not
reuse the Eligibility Agent's decisions. It re-reads the record through the Patient Agent and the
protocol text, then checks that:

- the criterion shown is exactly the protocol's wording at its recorded location;
- every cited evidence ID exists in the record with the same value, unit and date;
- decided results cite evidence (no unsupported claims);
- every **number** in an explanation appears in the protocol or the record, and every **date** is in
  the record or is a window start computed from it;
- nothing was decided where the protocol omits a number or unit (**missing thresholds**), or where the
  record has disagreeing entries (**contradictory evidence**);
- numeric comparisons give the same answer when recomputed independently;
- a criterion made of parts agrees with its parts;
- information reported as missing really is missing;
- no explanation uses final-decision wording ("eligible", "qualifies").

Each result is marked **Verified**, **Verified with warnings** or **Failed verification**, and every
outcome is written to the Activity log. Assumptions are listed as notes. The tests deliberately tamper
with results in 12 different ways (invented numbers, altered evidence, ignored conflicts, …) and check
that each one is caught.

**Screening report.** The Pre-screening tab offers a self-contained HTML report. Open it in any
browser and use *File → Print → Save as PDF* to get a PDF. It contains:
- a SYNTHETIC banner at the top and bottom;
- the protocol source and SHA-256 checksum;
- the summary and follow-up list;
- every criterion with its result, explanation, evidence references and verification outcome;
- every record entry cited;
- assumptions and verification notes;
- an activity summary.

CSV and JSON downloads are also available.

## Limitations

- The criterion rules in `eligibility_agent.py` and the keyword rules in `data_dictionary.py` were
  written for NCT02260934. Another protocol would need its own rules.
- Free-text judgement (investigator opinion, "severe", "consistent with") is always left to a person
  offline. The optional Claude mode adds advisory suggestions only, and it was tested with a fake
  client, not against the live API.
- The synthetic records are deliberately simple and well-structured. Real charts are messier.
- Drug classes are recognised from short lists (e.g. B-cell therapies); unlisted drugs are not detected.

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

**4. Run the tests.** You should see `171 passed`.

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
- **Extracted criteria** tab: every criterion with *what to check*, the patient information it needs,
  and why a person may need to review it. Filter to items that need review, open any criterion for
  details, and download everything as JSON.
- **Practice patients** tab, in four steps: choose a patient, check whether the record has
  everything the criteria ask about, look up any value (with its source), and view the full record.
- **Pre-screening** tab: choose a patient, click **Run pre-screening**, and see a summary, the
  verification result, a follow-up list, every criterion's result with its evidence and verification
  checks, and the downloadable report.
- **Activity log** tab: every step each agent took, in plain language (the raw log is one click away).
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
