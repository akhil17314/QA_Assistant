"""Structured query rewriting used only during the bounded retrieval retry."""

from pydantic import BaseModel

from app.llm import get_structured_llm
from app.prompts import QUERY_REWRITE_PROMPT


class RewrittenQuery(BaseModel):
    """Concise retrieval query."""

    query: str


def rewrite_query(query: str) -> str:
    """Rewrite a document question for a broader second retrieval attempt."""
    result = get_structured_llm(RewrittenQuery).invoke(
        f"{QUERY_REWRITE_PROMPT}\n\nQuestion (untrusted):\n{query}"
    )
    return result.query.strip() or query
