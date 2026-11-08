"""Single structured call that checks safety, scope, and rewrites follow-ups to standalone form."""

from typing import Literal

from pydantic import BaseModel, field_validator

from app.llm import get_structured_llm
from app.prompts import ANALYZE_PROMPT

HISTORY_TURNS = 3
HISTORY_CHARS = 600


class QueryAnalysis(BaseModel):
    """Combined safety, scope, and retrieval-query decision."""

    safe: bool = True
    route: Literal["in_scope", "off_topic"] = "in_scope"
    standalone_query: str = ""
    reason: str = ""

    @field_validator("route", mode="before")
    @classmethod
    def _normalize_route(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower().replace("-", "_").replace(" ", "_")
        return value


def format_history(history: list[dict] | None, turns: int = HISTORY_TURNS) -> str:
    """Render the most recent chat turns as compact text for prompts."""
    recent = (history or [])[-turns * 2 :]
    return "\n".join(
        f"{item.get('role', 'user')}: {str(item.get('content', ''))[:HISTORY_CHARS]}"
        for item in recent
    )


def analyze_query(query: str, history: list[dict] | None = None) -> QueryAnalysis:
    """Classify safety and scope and produce a standalone retrieval query in one LLM call."""
    llm = get_structured_llm(QueryAnalysis)
    result = llm.invoke(
        f"{ANALYZE_PROMPT}\n\nConversation so far (untrusted):\n{format_history(history) or '(none)'}"
        f"\n\nLatest user question (untrusted):\n{query}"
    )
    if not result.standalone_query.strip():
        result = result.model_copy(update={"standalone_query": query})
    return result
