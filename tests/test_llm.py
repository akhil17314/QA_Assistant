from pydantic import BaseModel

from app.llm import _call_counter, get_structured_llm, track_llm_calls


def test_model_callback_counts_calls_only_inside_request_scope():
    with track_llm_calls() as counter:
        _call_counter.on_llm_start({}, ["prompt"])
        _call_counter.on_chat_model_start({}, [[]])

    assert counter == [2]
    with track_llm_calls() as next_counter:
        assert next_counter == [0]


def test_structured_output_is_configured_before_retry(monkeypatch):
    events = []

    class StructuredRunnable:
        def with_config(self, **kwargs):
            events.append("config")
            return self

        def with_retry(self, **kwargs):
            events.append("retry")
            return self

    class ChatModel:
        def with_structured_output(self, schema):
            events.append(("structured", schema))
            return StructuredRunnable()

    class OutputSchema(BaseModel):
        result: str

    monkeypatch.setattr("app.llm._create_chat_model", lambda model, temperature: ChatModel())

    get_structured_llm(OutputSchema)

    assert events == [("structured", OutputSchema), "config", "retry"]
