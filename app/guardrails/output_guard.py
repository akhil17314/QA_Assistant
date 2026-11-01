"""Schema validation, citation validation, and response secret redaction."""

import re

from app.state import AnswerOutput, Citation

API_KEY_PATTERN = re.compile(
    r"\b(?:AIza[0-9A-Za-z_-]{30,}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})\b"
)
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)


def validate_schema(raw: AnswerOutput | dict | str) -> AnswerOutput:
    """Parse raw model output into the required answer schema."""
    if isinstance(raw, AnswerOutput):
        return raw
    if isinstance(raw, str):
        return AnswerOutput.model_validate_json(raw)
    return AnswerOutput.model_validate(raw)


def check_citations(answer: AnswerOutput, chunks: list[dict]) -> list[Citation]:
    """Return citations whose source and page do not occur in retrieved chunks."""
    valid_locations = {
        (str(chunk.get("source", "")), int(chunk.get("page", 0))) for chunk in chunks
    }
    return [
        citation
        for citation in answer.citations
        if (citation.source, citation.page) not in valid_locations
    ]


def filter_citations(answer: AnswerOutput, chunks: list[dict]) -> AnswerOutput:
    """Drop citations not backed by a retrieved source/page pair."""
    invalid = check_citations(answer, chunks)
    invalid_keys = {(item.source, item.page) for item in invalid}
    return answer.model_copy(
        update={
            "citations": [
                item for item in answer.citations if (item.source, item.page) not in invalid_keys
            ]
        }
    )


def redact_secrets(text: str) -> str:
    """Redact common API-key shapes and email addresses from displayed text."""
    return EMAIL_PATTERN.sub("[REDACTED EMAIL]", API_KEY_PATTERN.sub("[REDACTED KEY]", text))
