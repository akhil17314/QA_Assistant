"""Streamlit interface for document ingestion and cited Q&A."""

import logging
import tempfile
from pathlib import Path

import streamlit as st

from app.agents.retriever import has_documents, indexed_sources
from app.config import settings
from app.graph import run_query
from app.ingest import ingest_file
from app.state import AnswerOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Document Q&A", page_icon="D", layout="wide")
st.title("Document Q&A")
st.caption("Ask questions grounded in your uploaded PDFs.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def ingest_samples_once() -> None:
    """Seed an empty index from PDFs shipped in data/sample, when available."""
    if st.session_state.get("skip_sample_seed") or has_documents():
        return
    sample_dir = Path(__file__).resolve().parents[1] / "data" / "sample"
    for pdf_path in sorted(sample_dir.glob("*.pdf")):
        try:
            ingest_file(pdf_path)
        except Exception as error:
            logger.warning(
                "Could not ingest sample PDF %s: %s", pdf_path.name, type(error).__name__
            )


ingest_samples_once()

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Ingest documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Choose at least one PDF first.")
        else:
            progress = st.progress(0)
            for index, uploaded in enumerate(uploaded_files, start=1):
                safe_name = Path(uploaded.name).name
                with tempfile.TemporaryDirectory(prefix="doc-qa-upload-") as temporary_directory:
                    temporary_path = Path(temporary_directory) / safe_name
                    temporary_path.write_bytes(uploaded.getvalue())
                    try:
                        chunk_count = ingest_file(temporary_path)
                        st.success(f"Indexed {safe_name}: {chunk_count} chunks")
                    except Exception as error:
                        st.error(f"Could not ingest {safe_name}: {error}")
                progress.progress(index / len(uploaded_files))

    sources = indexed_sources()
    st.subheader("Indexed files")
    if sources:
        for source in sources:
            st.write(f"- {source}")
    else:
        st.caption("No PDFs indexed yet.")

    if st.button("Clear index", use_container_width=True):
        try:
            st.session_state.skip_sample_seed = True
            from app.ingest import get_vector_store

            collection = get_vector_store()._collection
            indexed_ids = collection.get().get("ids", [])
            if indexed_ids:
                collection.delete(ids=indexed_ids)
            st.session_state.messages = []
            st.rerun()
        except Exception as error:
            st.error(f"Could not clear the index: {error}")

if not settings.google_api_key:
    st.error("GOOGLE_API_KEY is missing. Add it to your .env file to enable chat.")
if not has_documents():
    st.info("Upload and ingest a PDF to start asking questions.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            with st.expander("Sources"):
                for citation in message.get("citations", []):
                    st.markdown(
                        f"**{citation['source']}**, page {citation['page']}  \n"
                        f"> {citation['snippet']}"
                    )
            with st.expander("Trace"):
                st.write(f"Route: {message.get('route', 'unknown')}")
                st.write(f"Retries: {message.get('retries', 0)}")
                st.write(f"Grounded: {'Yes' if message.get('grounded') else 'No'}")
                if message.get("block_reason"):
                    st.write(f"Block reason: {message['block_reason']}")

question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            result = run_query(question)
            response = result.get("final_response", "No response was produced.")
            st.markdown(response)
            answer = result.get("answer")
            citations = answer.citations if isinstance(answer, AnswerOutput) else []
            citations_data = [citation.model_dump() for citation in citations]
            with st.expander("Sources"):
                if citations_data:
                    for citation in citations_data:
                        st.markdown(
                            f"**{citation['source']}**, page {citation['page']}  \n"
                            f"> {citation['snippet']}"
                        )
                else:
                    st.caption("No document citations were returned.")
            with st.expander("Trace"):
                st.write(f"Route: {result.get('route', 'unknown')}")
                st.write(f"Retries: {result.get('retries', 0)}")
                st.write(f"Grounded: {'Yes' if result.get('grounded') else 'No'}")
                if result.get("block_reason"):
                    st.write(f"Block reason: {result['block_reason']}")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                    "citations": citations_data,
                    "route": result.get("route", "unknown"),
                    "retries": result.get("retries", 0),
                    "grounded": result.get("grounded", False),
                    "block_reason": result.get("block_reason", ""),
                }
            )
        except Exception as error:
            if "429" in str(error) or "rate" in str(error).lower():
                st.error(
                    "Gemini's free-tier rate limit was reached. A document question can use "
                    "several model requests; wait about a minute before retrying and avoid "
                    "resubmitting while the request is still running. Check the quota for "
                    "your selected model if this continues."
                )
            else:
                st.error(f"The request could not be completed: {error}")
