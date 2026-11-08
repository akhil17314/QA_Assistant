"""Streamlit interface for document ingestion and cited Q&A."""

import logging
import sys
import tempfile
import uuid
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st

from app.agents.retriever import has_documents, indexed_sources
from app.config import settings
from app.graph import run_query
from app.ingest import clear_collection, ingest_file
from app.state import AnswerOutput

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Document Q&A", page_icon="D", layout="wide")
st.title("Document Q&A")
st.caption("Ask questions grounded in your uploaded PDFs.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection" not in st.session_state:
    # A fresh collection per browser session keeps sessions isolated from each other.
    st.session_state.collection = f"session-{uuid.uuid4().hex}"
collection = st.session_state.collection


def chat_history() -> list[dict]:
    """Prior turns used as conversation context, excluding refused messages."""
    messages = st.session_state.messages
    history = []
    for index, message in enumerate(messages):
        if message["role"] == "assistant" and message.get("block_reason"):
            continue
        if message["role"] == "user" and index + 1 < len(messages):
            reply = messages[index + 1]
            if reply["role"] == "assistant" and reply.get("block_reason"):
                continue
        history.append({"role": message["role"], "content": message["content"]})
    return history


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
                        chunk_count = ingest_file(temporary_path, collection)
                        st.success(f"Indexed {safe_name}: {chunk_count} chunks")
                    except Exception as error:
                        st.error(f"Could not ingest {safe_name}: {error}")
                progress.progress(index / len(uploaded_files))

    sources = indexed_sources(collection)
    st.subheader("Indexed files")
    if sources:
        for source in sources:
            st.write(f"- {source}")
    else:
        st.caption("No PDFs indexed yet.")

    if st.button("Clear index", use_container_width=True):
        try:
            clear_collection(collection)
            st.session_state.messages = []
            st.rerun()
        except Exception as error:
            st.error(f"Could not clear the index: {error}")

if not settings.google_api_key:
    st.error("GOOGLE_API_KEY is missing. Add it to your .env file to enable chat.")
if not has_documents(collection):
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
    history = chat_history()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            result = run_query(question, history=history, collection=collection)
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
            logger.exception("Query failed for question: %r", question)
            if "429" in str(error) or "rate" in str(error).lower():
                st.error(
                    "Gemini's free-tier rate limit was reached. A document question can use "
                    "several model requests; wait about a minute before retrying and avoid "
                    "resubmitting while the request is still running. Check the quota for "
                    "your selected model if this continues."
                )
            else:
                st.error(f"The request could not be completed: {error}")
