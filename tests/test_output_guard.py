import pytest
from pydantic import ValidationError

from app.agents.grounding import check_grounding
from app.guardrails.output_guard import check_citations, redact_secrets, validate_schema
from app.state import AnswerOutput, Citation


def test_invalid_json_is_rejected():
    with pytest.raises(ValidationError):
        validate_schema('{"answer": 1, "citations": [], "confidence": "certain"}')


def test_invented_citation_is_flagged():
    answer = AnswerOutput(
        answer="The result is 42.",
        citations=[Citation(source="invented.pdf", page=9, snippet="42")],
        confidence="medium",
    )
    assert check_citations(answer, [{"source": "real.pdf", "page": 1}]) == answer.citations


def test_secrets_and_emails_are_redacted():
    redacted = redact_secrets(
        "Contact user@example.com; key=AIza12345678901234567890123456789012345"
    )
    assert "user@example.com" not in redacted
    assert "AIza" not in redacted


def test_explicit_abstention_is_grounded_without_a_judge():
    answer = AnswerOutput(answer="I do not know from these documents.", confidence="low")

    grounded, report = check_grounding(answer, [])

    assert grounded is True
    assert "abstains" in report
