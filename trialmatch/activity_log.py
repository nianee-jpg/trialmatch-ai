"""Agent activity log: a visible record of every tool call and check.

Each agent writes to an ActivityLog; the dashboard shows it as a table so a
reviewer can see exactly what ran, with what input, and what came back.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class LogEntry(BaseModel):
    timestamp: str
    agent: str
    event: str = Field(..., description="tool_call | tool_result | llm_request | llm_response | info | warning | verification")
    tool: Optional[str] = None
    detail: str


class ActivityLog:
    def __init__(self) -> None:
        self.entries: List[LogEntry] = []

    def add(self, agent: str, event: str, detail: str, tool: Optional[str] = None) -> None:
        self.entries.append(
            LogEntry(
                timestamp=datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3],
                agent=agent,
                event=event,
                tool=tool,
                detail=detail,
            )
        )

    def as_rows(self) -> List[dict]:
        return [e.model_dump() for e in self.entries]


def summarize(value: Any, limit: int = 160) -> str:
    """Short one-line rendering of a tool input/output for the log."""
    try:
        text = json.dumps(value, default=str, ensure_ascii=False)
    except TypeError:
        text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"
