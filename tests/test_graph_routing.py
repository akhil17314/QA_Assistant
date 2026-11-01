from app.graph import GraphServices, build_graph
from app.state import AnswerOutput
from evals.metrics import hit_rate_at_k, mean_reciprocal_rank


def _base_services(**overrides):
    services = {
        "input_check": lambda _: ("safe", ""),
        "route": lambda _: "in_scope",
        "has_documents": lambda: True,
        "retrieve": lambda query, k: [{"text": query, "source": "doc.pdf", "page": 1}],
        "answer": lambda _query, _chunks: AnswerOutput(
            answer="The document says yes.", citations=[], confidence="high"
        ),
        "ground": lambda _answer, _chunks: (True, "supported"),
        "rewrite": lambda query: query + " detail",
    }
    services.update(overrides)
    return GraphServices(**services)


def test_blocked_branch_skips_agents():
    result = build_graph(_base_services(input_check=lambda _: ("blocked", "injection"))).invoke(
        {"query": "unsafe", "retries": 0}
    )
    assert result["final_response"].startswith("I can't help")
    assert "route" not in result


def test_off_topic_branch_declines():
    result = build_graph(_base_services(route=lambda _: "off_topic")).invoke(
        {"query": "Tell me a joke", "retries": 0}
    )
    assert "questions about the documents" in result["final_response"]


def test_grounded_branch_finalizes():
    result = build_graph(_base_services()).invoke({"query": "What does it say?", "retries": 0})
    assert result["final_response"] == "The document says yes."
    assert result["grounded"] is True


def test_control_characters_are_removed_before_routing():
    seen = []
    services = _base_services(route=lambda query: seen.append(query) or "off_topic")

    build_graph(services).invoke({"query": "What\x00 is documented?", "retries": 0})

    assert seen == ["What is documented?"]


def test_retry_then_grounded():
    verdicts = iter([(False, "unsupported"), (True, "supported after retry")])
    result = build_graph(_base_services(ground=lambda _answer, _chunks: next(verdicts))).invoke(
        {"query": "What does it say?", "retries": 0}
    )
    assert result["retries"] == 1
    assert result["grounded"] is True


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
