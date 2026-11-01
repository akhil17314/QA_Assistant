from app.llm import _call_counter, track_llm_calls


def test_model_callback_counts_calls_only_inside_request_scope():
    with track_llm_calls() as counter:
        _call_counter.on_llm_start({}, ["prompt"])
        _call_counter.on_chat_model_start({}, [[]])

    assert counter == [2]
    with track_llm_calls() as next_counter:
        assert next_counter == [0]
