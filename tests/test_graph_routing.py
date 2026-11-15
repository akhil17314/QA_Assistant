from app.agents.router import QueryAnalysis
from app.graph import GraphServices, build_graph
from app.state import AnswerOutput
from evals.metrics import hit_rate_at_k, mean_reciprocal_rank


def _base_services(**overrides):
    services = {
        "input_check": lambda _: ("safe", ""),
        "analyze": lambda query, _history: QueryAnalysis(standalone_query=query),
        "has_documents": lambda _collection: True,
        "retrieve": lambda query, k, _collection: [
            {"text": query, "source": "doc.pdf", "page": 1}
        ],
        "answer": lambda _query, _chunks, _history: AnswerOutput(
            answer="The document says yes.", citations=[], confidence="high"
        ),
        "ground": lambda _answer, _chunks: (True, "supported"),
    }
    services.update(overrides)
    return GraphServices(**services)


def test_blocked_branch_skips_agents():
    result = build_graph(_base_services(input_check=lambda _: ("blocked", "injection"))).invoke(
        {"query": "unsafe", "retries": 0}
    )
    assert result["final_response"].startswith("I can't help")
    assert "route" not in result


def test_analysis_flagged_unsafe_is_refused():
    services = _base_services(
        analyze=lambda *_: QueryAnalysis(safe=False, reason="jailbreak attempt")
    )
    result = build_graph(services).invoke({"query": "sneaky", "retries": 0})
    assert result["final_response"] == "I can't help with that request. jailbreak attempt"
    assert "chunks" not in result


def test_off_topic_branch_declines():
    services = _base_services(analyze=lambda *_: QueryAnalysis(route="off_topic"))
    result = build_graph(services).invoke({"query": "Tell me a joke", "retries": 0})
    assert "questions about the documents" in result["final_response"]


def test_no_documents_skips_llm_analysis():
    def fail(*_):
        raise AssertionError("analysis must not run without documents")

    result = build_graph(_base_services(has_documents=lambda _c: False, analyze=fail)).invoke(
        {"query": "What does it say?", "retries": 0}
    )
    assert "upload and ingest a PDF" in result["final_response"]


def test_grounded_branch_finalizes():
    result = build_graph(_base_services()).invoke({"query": "What does it say?", "retries": 0})
    assert result["final_response"] == "The document says yes."
    assert result["grounded"] is True


def test_control_characters_are_removed_before_analysis():
    seen = []
    services = _base_services(
        analyze=lambda query, _h: seen.append(query) or QueryAnalysis(route="off_topic")
    )

    build_graph(services).invoke({"query": "What\x00 is documented?", "retries": 0})

    assert seen == ["What is documented?"]


def test_history_and_collection_reach_agents_and_standalone_query_drives_retrieval():
    calls = {}
    services = _base_services(
        analyze=lambda query, history: calls.update(history=history)
        or QueryAnalysis(standalone_query="standalone about timeouts"),
        retrieve=lambda query, k, collection: calls.update(query=query, collection=collection)
        or [{"text": "t", "source": "doc.pdf", "page": 1}],
        answer=lambda query, _chunks, history: calls.update(answer_history=history)
        or AnswerOutput(answer="ok", citations=[], confidence="high"),
    )
    history = [{"role": "user", "content": "What about timeouts?"}]

    build_graph(services).invoke(
        {"query": "and retries?", "retries": 0, "history": history, "collection": "session-1"}
    )

    assert calls["history"] == calls["answer_history"] == history
    assert calls["query"] == "standalone about timeouts"
    assert calls["collection"] == "session-1"


def test_retry_then_grounded():
    verdicts = iter([(False, "unsupported"), (True, "supported after retry")])
    result = build_graph(_base_services(ground=lambda _answer, _chunks: next(verdicts))).invoke(
        {"query": "What does it say?", "retries": 0}
    )
    assert result["retries"] == 1
    assert result["grounded"] is True


def test_retry_widens_retrieval_without_extra_llm_call():
    ks = []
    verdicts = iter([(False, "unsupported"), (True, "ok")])
    services = _base_services(
        retrieve=lambda query, k, _c: ks.append(k) or [{"text": "t", "source": "d", "page": 1}],
        ground=lambda _a, _c: next(verdicts),
        analyze=lambda query, _h: QueryAnalysis(standalone_query=query),
    )
    build_graph(services).invoke({"query": "q", "retries": 0})
    assert ks[1] > ks[0]


def test_retry_exhausted_returns_fixed_fallback(monkeypatch):
    monkeypatch.setattr("app.graph.settings.max_retries", 1)
    result = build_graph(
        _base_services(ground=lambda _answer, _chunks: (False, "unsupported"))
    ).invoke({"query": "What does it say?", "retries": 0})
    assert result["retries"] == 1
    assert (
        result["final_response"]
        == "I could not find a well-supported answer in the uploaded documents."
    )


def test_retrieval_metrics_accept_multiple_gold_pages():
    retrieved = [[{"source": "doc.pdf", "page": 3}, {"source": "doc.pdf", "page": 5}]]
    expected = [[("doc.pdf", 4), ("doc.pdf", 5)]]

    assert hit_rate_at_k(retrieved, expected) == 1.0
    assert mean_reciprocal_rank(retrieved, expected) == 0.5
