"""Shared Claude tool-calling loop (optional; only used when an API key is set).

The loop sends the conversation, runs any tools Claude asks for through the
given handlers, logs every step, and stops on end_turn, refusal, max_tokens,
an API error, or the turn limit.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from trialmatch.activity_log import ActivityLog, summarize


def run_tool_loop(
    *,
    client: Any,
    model: str,
    system: str,
    tools: List[Dict[str, Any]],
    first_message: str,
    handlers: Dict[str, Callable[[Dict[str, Any]], Any]],
    log: ActivityLog,
    agent_name: str,
    max_turns: int = 25,
) -> bool:
    """Run Claude with tools until it finishes. Returns True only if it ended normally."""
    import anthropic  # imported here so offline mode never needs the SDK configured

    messages: List[Dict[str, Any]] = [{"role": "user", "content": first_message}]
    for turn in range(1, max_turns + 1):
        log.add(agent_name, "llm_request", f"Turn {turn}: sending {len(messages)} messages to {model}.")
        try:
            response = client.beta.messages.create(
                model=model,
                max_tokens=16000,
                system=system,
                tools=tools,
                messages=messages,
                # Server-side refusal fallback; Anthropic picks the fallback model.
                betas=["server-side-fallback-2026-07-01"],
                extra_body={"fallbacks": "default", "output_config": {"effort": "medium"}},
            )
        except anthropic.APIError as exc:
            log.add(agent_name, "warning", f"Claude API error, keeping rule-based results: {exc}")
            return False

        log.add(agent_name, "llm_response", f"stop_reason={response.stop_reason}")
        if response.stop_reason == "refusal":
            log.add(agent_name, "warning", "Claude declined the request; keeping rule-based results.")
            return False
        if response.stop_reason == "max_tokens":
            log.add(agent_name, "warning", "Claude hit max_tokens; keeping results accepted so far.")
            return False

        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "end_turn":
            summary = " ".join(b.text for b in response.content if getattr(b, "type", "") == "text")
            log.add(agent_name, "info", f"Claude finished: {summary[:300]}")
            return True
        if response.stop_reason == "pause_turn":
            continue

        results = []
        for block in response.content:
            if getattr(block, "type", "") != "tool_use":
                continue
            log.add(agent_name, "tool_call", summarize(block.input), tool=block.name)
            try:
                output = handlers[block.name](block.input)
                is_error = False
            except (KeyError, ValueError, TypeError) as exc:
                output, is_error = {"error": f"{type(exc).__name__}: {exc}"}, True
            log.add(agent_name, "tool_result", summarize(output), tool=block.name)
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(output, default=str), "is_error": is_error})
        messages.append({"role": "user", "content": results})

    log.add(agent_name, "warning", f"Stopped after {max_turns} turns.")
    return False
