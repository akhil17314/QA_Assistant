"""Streamlit interface for document ingestion and cited Q&A."""

import logging
import sys
import tempfile
import time
import uuid
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st  # noqa: E402

from app.agents.retriever import index_status  # noqa: E402
from app.config import settings  # noqa: E402
from app.graph import run_query  # noqa: E402
from app.ingest import clear_collection, ingest_file  # noqa: E402
from app.llm import is_invalid_api_key_error  # noqa: E402
from app.state import AnswerOutput  # noqa: E402

MAX_UPLOAD_MB = 20
MIN_SECONDS_BETWEEN_QUESTIONS = 3

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
        if message["role"] == "assistant" and message.get("error"):
            continue
        if message["role"] == "user" and index + 1 < len(messages):
            reply = messages[index + 1]
            if reply["role"] == "assistant" and (
                reply.get("block_reason") or reply.get("error")
            ):
                continue
        history.append({"role": message["role"], "content": message["content"]})
    return history


def render_chat_message(message: dict) -> None:
    """Render a chat message and any assistant details consistently."""
    if message.get("error"):
        st.error(message["content"])
        return
    st.markdown(message["content"])
    if message["role"] != "assistant":
        return

    with st.expander("Sources"):
        for citation in message.get("citations", []):
            st.markdown(
                f"**{citation['source']}**, page {citation['page']}  \n"
                f"> {citation['snippet']}"
            )
        if not message.get("citations"):
            st.caption("No document citations were returned.")
    with st.expander("Trace"):
        st.write(f"Route: {message.get('route', 'unknown')}")
        st.write(f"Retries: {message.get('retries', 0)}")
        st.write(f"Grounded: {'Yes' if message.get('grounded') else 'No'}")
        if message.get("block_reason"):
            st.write(f"Block reason: {message['block_reason']}")


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
                if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
                    st.error(f"{safe_name} exceeds the {MAX_UPLOAD_MB} MB limit.")
                    progress.progress(index / len(uploaded_files))
                    continue
                with tempfile.TemporaryDirectory(prefix="doc-qa-upload-") as temporary_directory:
                    temporary_path = Path(temporary_directory) / safe_name
                    temporary_path.write_bytes(uploaded.getvalue())
                    try:
                        with st.spinner(f"Reading and indexing {safe_name}..."):
                            chunk_count = ingest_file(temporary_path, collection)
                        st.success(f"Indexed {safe_name}: {chunk_count} chunks")
                    except Exception as error:
                        logger.exception("Ingest failed for %s", safe_name)
                        message = (
                            str(error)
                            if isinstance(error, ValueError)
                            else "The file could not be read as a valid PDF."
                        )
                        st.error(f"Could not ingest {safe_name}: {message}")
                progress.progress(index / len(uploaded_files))

    try:
        documents_available, sources = index_status(collection)
    except Exception:
        documents_available, sources = None, None
        st.error("Could not load indexed files. Please try again.")
    st.subheader("Indexed files")
    if sources:
        for source in sources:
            st.write(f"- {source}")
    elif sources is not None:
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
if documents_available is False:
    st.info("Upload and ingest a PDF to start asking questions.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render_chat_message(message)

question = st.chat_input("Ask a question about your documents")
if question and time.monotonic() - st.session_state.get("last_query", 0.0) < (
    MIN_SECONDS_BETWEEN_QUESTIONS
):
    st.warning("Please wait a few seconds between questions.")
    question = None
if question:
    st.session_state.last_query = time.monotonic()
    history = chat_history()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your documents and checking the answer..."):
                result = run_query(question, history=history, collection=collection)
            response = result.get("final_response", "No response was produced.")
            answer = result.get("answer")
            citations = answer.citations if isinstance(answer, AnswerOutput) else []
            citations_data = [citation.model_dump() for citation in citations]
            assistant_message = {
                "role": "assistant",
                "content": response,
                "citations": citations_data,
                "route": result.get("route", "unknown"),
                "retries": result.get("retries", 0),
                "grounded": result.get("grounded", False),
                "block_reason": result.get("block_reason", ""),
            }
            st.session_state.messages.append(assistant_message)
            render_chat_message(assistant_message)
        except Exception as error:
            logger.exception("Query failed for question: %r", question)
            error_text = str(error).lower()
            if is_invalid_api_key_error(error):
                error_message = (
                    "Gemini rejected the API key. Check GOOGLE_API_KEY in your .env file, "
                    "then restart the Streamlit app and try again."
                )
            elif "429" in error_text or "rate" in error_text:
                error_message = (
                    "Gemini's free-tier rate limit was reached. A document question can use "
                    "several model requests; wait about a minute before retrying and avoid "
                    "resubmitting while the request is still running. Check the quota for "
                    "your selected model if this continues."
                )
            else:
                error_message = "The request could not be completed. Please try again."
            assistant_message = {"role": "assistant", "content": error_message, "error": True}
            st.session_state.messages.append(assistant_message)
            render_chat_message(assistant_message)
