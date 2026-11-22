from unittest.mock import Mock

import pytest
from app.agents.retriever import has_documents, index_status, indexed_sources, retrieve


def test_index_status_uses_one_read_for_availability_and_sources(monkeypatch):
    collection = Mock()
    collection.get.return_value = {
        "ids": ["chunk-1", "chunk-2"],
        "metadatas": [{"source": "b.pdf"}, {"source": "a.pdf"}],
    }
    monkeypatch.setattr(
        "app.agents.retriever.get_vector_store", lambda _: Mock(_collection=collection)
    )

    assert index_status("session-test") == (True, ["a.pdf", "b.pdf"])
    collection.get.assert_called_once_with(include=["metadatas"])


def test_index_and_retrieval_errors_propagate(monkeypatch):
    def fail(_collection):
        raise RuntimeError("index unavailable")

    monkeypatch.setattr("app.agents.retriever.get_vector_store", fail)

    for operation in (
        lambda: has_documents("session-test"),
        lambda: indexed_sources("session-test"),
        lambda: retrieve("question", collection="session-test"),
    ):
        with pytest.raises(RuntimeError, match="index unavailable"):
            operation()