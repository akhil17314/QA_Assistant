from app.ingest import Chunk, Page, build_index, chunk_pages


def test_chunking_preserves_page_metadata():
    chunks = chunk_pages([Page(text="alpha " * 30, page=3, source="notes.pdf")], size=40, overlap=5)

    assert chunks
    assert all(chunk.page == 3 for chunk in chunks)
    assert all(chunk.source == "notes.pdf" for chunk in chunks)


def test_reingesting_uses_deterministic_ids(monkeypatch):
    calls = []

    class FakeStore:
        def add_documents(self, documents, ids):
            calls.append((documents, ids))

    monkeypatch.setattr("app.ingest.get_vector_store", lambda _: FakeStore())
    chunks = [Chunk(text="text", source="notes.pdf", page=2)]

    build_index(chunks, "unused")
    build_index(chunks, "unused")

    assert calls[0][1] == calls[1][1] == ["notes.pdf-2-0"]
