"""Structured document-scope classification."""

from pydantic import BaseModel, Field

from app.llm import get_structured_llm
from app.prompts import ROUTER_PROMPT


class RouteDecision(BaseModel):
    """Scope decision returned by the router model."""

    route: str = Field(pattern="^(in_scope|off_topic)$")
    reason: str = ""


def route_query(query: str) -> RouteDecision:
    """Classify a question as document-answerable or off-topic."""
    llm = get_structured_llm(RouteDecision)
    return llm.invoke(f"{ROUTER_PROMPT}\n\nUser question (untrusted):\n{query}")
