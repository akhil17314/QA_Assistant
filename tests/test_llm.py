from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from app.agents.answerer import answer_question
from app.llm import _call_counter, get_structured_llm, is_invalid_api_key_error, track_llm_calls


def test_model_callback_counts_calls_only_inside_request_scope():
    with track_llm_calls() as counter:
        _call_counter.on_llm_start({}, ["prompt"])
        _call_counter.on_chat_model_start({}, [[]])

    assert counter == [2]
    with track_llm_calls() as next_counter:
        assert next_counter == [0]


def test_structured_output_is_configured_without_broad_retry(monkeypatch):
    class OutputSchema(BaseModel):
        result: str

    runnable, model = Mock(), Mock()
    runnable.with_config.return_value = runnable
    model.with_structured_output.return_value = runnable
    monkeypatch.setattr("app.llm._create_chat_model", lambda *_, **__: model)

    assert get_structured_llm(OutputSchema) is runnable
    model.with_structured_output.assert_called_once_with(OutputSchema)
    runnable.with_config.assert_called_once()
    runnable.with_retry.assert_not_called()



def test_invalid_api_key_is_detected_through_wrapped_exception():
    wrapped = RuntimeError("request failed")
    wrapped.__cause__ = ValueError(
        "Unauthenticated: 401 API key not valid; reason: ACCESS_TOKEN_TYPE_UNSUPPORTED"
    )

    assert is_invalid_api_key_error(wrapped)


def test_answerer_propagates_invalid_api_key(monkeypatch):
    runnable = Mock()
    runnable.invoke.side_effect = RuntimeError("API key not valid. Please pass a valid API key.")
    monkeypatch.setattr("app.agents.answerer.get_structured_llm", lambda *_: runnable)

    with pytest.raises(RuntimeError, match="API key not valid"):
        answer_question("question", [{"text": "context"}])
