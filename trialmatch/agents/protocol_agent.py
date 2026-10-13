"""Protocol Agent: turns eligibility text into structured, verifiable criteria.

Two modes, same output model:

  Offline (default, no API key)
      The agent runs its deterministic tools in a fixed order for every
      criterion: segment -> thresholds -> time windows -> flags -> required
      data -> interpretation -> verification.

  Claude (optional, only if ANTHROPIC_API_KEY is set)
      The offline pass runs first, so criterion IDs, exact text and
      thresholds are always rule-based. Claude is then given tools to read
      the criteria and to SUBMIT plain-language interpretations. The submit
      tool rejects any interpretation that contains a number not found in
      the source text, so Claude cannot introduce thresholds.

Every tool call, in both modes, is written to the ActivityLog.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from trialmatch import config
from trialmatch.activity_log import ActivityLog, summarize
from trialmatch.agents import protocol_tools as tools
from trialmatch.agents.claude_loop import run_tool_loop
from trialmatch.protocol_models import (
    Criterion,
    Flag,
    FlagCode,
    ProtocolExtraction,
    SourceReference,
)
from trialmatch.text_utils import unescape_markdown

AGENT_NAME = "Protocol Agent"
MAX_CLAUDE_TURNS = 25


class ProtocolAgent:
    def __init__(self, log: ActivityLog, client: Any = None, model: str = config.ANTHROPIC_MODEL) -> None:
        """`client` is an `anthropic.Anthropic()` instance, or None for offline mode."""
        self.log = log
        self.client = client
        self.model = model

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def run(
        self,
        nct_id: str,
        eligibility_text: str,
        raw_file: Optional[str] = None,
        raw_sha256: Optional[str] = None,
    ) -> ProtocolExtraction:
        self.log.add(AGENT_NAME, "info", f"Starting extraction for {nct_id} "
                     f"({len(eligibility_text)} characters of eligibility text).")
        criteria = self._rule_based_pass(nct_id, eligibility_text, raw_file, raw_sha256)

        mode, model = "offline (rule-based)", None
        if self.client is not None:
            if self._claude_pass(nct_id, criteria):
                mode, model = "claude", self.model
        else:
            self.log.add(AGENT_NAME, "info", "No Anthropic API key configured: running in offline rule-based mode.")

        issues = self._call("verify_extraction", tools.verify_extraction,
                            {"criteria": f"{len(criteria)} criteria"}, criteria, eligibility_text)
        if issues:
            for issue in issues:
                self.log.add(AGENT_NAME, "verification", f"Problem ({issue.check}) {issue.criterion_id or ''}: {issue.message}")
        else:
            self.log.add(AGENT_NAME, "verification",
                         "Passed: every criterion matches the published text, nothing was skipped, no numbers were added.")

        return ProtocolExtraction(nct_id=nct_id, mode=mode, model=model, criteria=criteria,
                                  verification_issues=issues)

    # ------------------------------------------------------------------ #
    # Tool dispatch with logging
    # ------------------------------------------------------------------ #

    def _call(self, name: str, func: Callable[..., Any], shown_input: Any, *args: Any) -> Any:
        self.log.add(AGENT_NAME, "tool_call", summarize(shown_input), tool=name)
        result = func(*args)
        self.log.add(AGENT_NAME, "tool_result", _describe_result(result), tool=name)
        return result

    # ------------------------------------------------------------------ #
    # Offline pass
    # ------------------------------------------------------------------ #

    def _rule_based_pass(self, nct_id: str, text: str, raw_file: Optional[str],
                         raw_sha256: Optional[str]) -> List[Criterion]:
        segments = self._call("segment_criteria", tools.segment_criteria,
                              {"eligibility_text": f"{len(text):,} characters"}, text)
        child_ids: Dict[str, List[str]] = {}
        for seg in segments:
            if seg.parent_id:
                child_ids.setdefault(seg.parent_id, []).append(seg.criterion_id)

        criteria: List[Criterion] = []
        for seg in segments:
            original = text[seg.char_start:seg.char_end]
            shown = {"criterion_id": seg.criterion_id, "text": unescape_markdown(original)}
            thresholds, t_flags = self._call("extract_thresholds", tools.extract_thresholds, shown, original)
            windows, w_flags = self._call("extract_time_windows", tools.extract_time_windows, shown, original)
            j_flags = self._call("detect_flags", tools.detect_flags, shown, original)
            required, r_flags = self._call("map_required_data", tools.map_required_data, shown, original)

            children = child_ids.get(seg.criterion_id, [])
            if children:
                r_flags = []  # a grouping criterion's data needs come from its sub-criteria
            flags = t_flags + w_flags + j_flags + r_flags
            if seg.structure_note:
                flags.append(Flag(code=FlagCode.STRUCTURE_INFERRED, message=seg.structure_note))

            display = unescape_markdown(original)
            criteria.append(Criterion(
                criterion_id=seg.criterion_id,
                type=seg.type,
                parent_id=seg.parent_id,
                logic=seg.logic,
                original_text=original,
                display_text=display,
                normalized_interpretation=tools.build_interpretation(seg.type, display, thresholds, windows, children),
                thresholds=thresholds,
                time_windows=windows,
                required_data=required,
                flags=flags,
                source=SourceReference(nct_id=nct_id, char_start=seg.char_start, char_end=seg.char_end,
                                       line_number=seg.line_number, raw_file=raw_file, raw_sha256=raw_sha256),
            ))
        self.log.add(AGENT_NAME, "info", f"Rule-based pass produced {len(criteria)} criteria.")
        return criteria

    # ------------------------------------------------------------------ #
    # Optional Claude pass (real tool calling)
    # ------------------------------------------------------------------ #

    def _claude_pass(self, nct_id: str, criteria: List[Criterion]) -> bool:
        """Let Claude refine interpretations via tools. Returns True if it completed."""
        by_id = {c.criterion_id: c for c in criteria}
        handlers = {
            "list_criteria": lambda _inp: [
                {"criterion_id": c.criterion_id, "type": c.type.value, "parent_id": c.parent_id,
                 "text": c.display_text, "flags": [f.code.value for f in c.flags]}
                for c in criteria
            ],
            "get_criterion_details": lambda inp: _criterion_details(by_id[inp["criterion_id"]]),
            "submit_interpretation": lambda inp: _submit(by_id, inp),
        }

        return run_tool_loop(
            client=self.client,
            model=self.model,
            system=CLAUDE_SYSTEM_PROMPT,
            tools=CLAUDE_TOOLS,
            first_message=(
                f"Review the {len(criteria)} eligibility criteria of {nct_id}. Call list_criteria, inspect any "
                "criterion with get_criterion_details, and call submit_interpretation for every criterion that "
                "has flags or no numeric threshold. Finish with a one-sentence summary."),
            handlers=handlers,
            log=self.log,
            agent_name=AGENT_NAME,
            max_turns=MAX_CLAUDE_TURNS,
        )


# ---------------------------------------------------------------------- #
# Claude tool definitions and handlers
# ---------------------------------------------------------------------- #

CLAUDE_SYSTEM_PROMPT = """You assist a clinical research coordinator in an EDUCATIONAL SIMULATION that \
reviews a historical trial protocol. You are given eligibility criteria that were already split, \
located and analysed by deterministic code. Your only job is to write a clear, plain-language \
interpretation of what each criterion requires, so that a coordinator knows what evidence to look for.

Rules:
- Use only the criterion's own wording. Do not add clinical knowledge as if it were protocol text.
- Never introduce a number, threshold, unit or time window that is not in the criterion text. \
If a threshold is missing (flag MISSING_THRESHOLD), say that it is missing; do not suggest a value.
- Where wording is ambiguous or needs investigator judgement, say so in ambiguity_notes.
- Write interpretations as what must be checked, never as an eligibility decision about a patient.
- If submit_interpretation rejects your text, fix the stated problem and resubmit."""

CLAUDE_TOOLS = [
    {
        "name": "list_criteria",
        "description": "List every extracted criterion with its ID, type, parent, readable text and flag codes.",
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "get_criterion_details",
        "description": ("Get one criterion's original text, extracted thresholds, time windows, required data, "
                        "flags with explanations, and the current rule-based interpretation."),
        "input_schema": {
            "type": "object",
            "properties": {"criterion_id": {"type": "string", "description": "e.g. EXC-04 or INC-03.b.1"}},
            "required": ["criterion_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "submit_interpretation",
        "description": ("Replace a criterion's interpretation with a plain-language one. Rejected if it contains "
                        "any number that does not appear in the criterion's original text."),
        "input_schema": {
            "type": "object",
            "properties": {
                "criterion_id": {"type": "string"},
                "interpretation": {"type": "string", "description": "What must be checked, in plain language."},
                "ambiguity_notes": {"type": "array", "items": {"type": "string"},
                                    "description": "Points a human reviewer must resolve; empty if none."},
            },
            "required": ["criterion_id", "interpretation", "ambiguity_notes"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def _criterion_details(c: Criterion) -> Dict[str, Any]:
    return c.model_dump(mode="json", exclude={"source"})


def _submit(by_id: Dict[str, Criterion], inp: Dict[str, Any]) -> Dict[str, Any]:
    c = by_id[inp["criterion_id"]]
    text = inp["interpretation"].strip()
    notes = [n.strip() for n in inp.get("ambiguity_notes", []) if n.strip()]
    if not text:
        return {"status": "rejected", "reason": "Interpretation is empty."}
    bad = tools.unsupported_numbers(text + " " + " ".join(notes), c.original_text)
    if bad:
        return {"status": "rejected",
                "reason": f"Contains numbers not present in the criterion text: {bad}. Remove them."}
    c.normalized_interpretation = text
    c.interpretation_source = "claude"
    c.flags = [f for f in c.flags if f.code != FlagCode.AMBIGUITY_NOTED_BY_CLAUDE]
    c.flags += [Flag(code=FlagCode.AMBIGUITY_NOTED_BY_CLAUDE, message=n) for n in notes]
    return {"status": "accepted", "criterion_id": c.criterion_id}


def _describe_result(result: Any) -> str:
    """Compact description of a tool result for the activity log."""
    if isinstance(result, tuple):
        return " | ".join(_describe_result(r) for r in result)
    if isinstance(result, list):
        if not result:
            return "[]"
        first = result[0]
        if hasattr(first, "code"):  # Flag
            return "flags: " + ", ".join(f.code.value for f in result)
        if hasattr(first, "comparator"):  # Threshold
            return "thresholds: " + ", ".join(
                f"{t.parameter} {t.comparator.value} {t.value if t.value is not None else 'MISSING'}" for t in result)
        if hasattr(first, "anchor"):  # TimeWindow
            return "windows: " + ", ".join(f"{w.amount} {w.unit} / {w.anchor or 'no anchor'}" for w in result)
        if hasattr(first, "field"):  # RequiredData
            return "fields: " + ", ".join(r.field for r in result)
        if hasattr(first, "char_start"):  # Segment
            return f"{len(result)} criteria: " + ", ".join(s.criterion_id for s in result)
        if hasattr(first, "check"):  # VerificationIssue
            return f"{len(result)} verification issue(s)"
        return summarize(result)
    return summarize(result)


def make_claude_client() -> Any:
    """Return an Anthropic client if an API key is configured, else None."""
    if not config.ANTHROPIC_API_KEY:
        return None
    import anthropic

    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
