"""LangGraph orchestration with conditional guard, scope, and grounding routes."""

import logging
from dataclasses import dataclass
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from app.agents.answerer import answer_question
from app.agents.grounding import check_grounding
from app.agents.retriever import has_documents, retrieve
from app.agents.router import analyze_query
from app.config import settings
from app.guardrails.input_guard import sanitize_input, validate_input_rules
from app.guardrails.output_guard import redact_secrets
from app.ingest import DEFAULT_COLLECTION
from app.llm import track_llm_calls
from app.state import AnswerOutput, GraphState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphServices:
    """Injectable agent functions, primarily to keep graph tests offline."""

    input_check: Callable = validate_input_rules
    analyze: Callable = analyze_query
    has_documents: Callable = has_documents
    retrieve: Callable = retrieve
    answer: Callable = answer_question
    ground: Callable = check_grounding


def build_graph(services: GraphServices | None = None) -> Any:
    """Construct and compile the document QA state graph."""
    agents = services or GraphServices()

    def input_guard(state: GraphState) -> dict:
        query = sanitize_input(state.get("query", ""))
        verdict, reason = agents.input_check(query)
        return {"query": query, "input_verdict": verdict, "block_reason": reason}


    def refuse(state: GraphState) -> dict:
        reason = state.get("block_reason", "")
        response = f"I can't help with that request. {reason}".strip()
        return {"final_response": redact_secrets(response)}

    def router(state: GraphState) -> dict:
        collection = state.get("collection", DEFAULT_COLLECTION)
        if not agents.has_documents(collection):
            return {
                "route": "off_topic",
                "final_response": "Please upload and ingest a PDF before asking a document question.",
            }
        query = state.get("query", "")
        decision = agents.analyze(query, state.get("history", []))
        if not decision.safe:
            return {
                "input_verdict": "blocked",
                "block_reason": decision.reason or "The message was blocked by the safety check.",
            }
        return {
            "route": decision.route,
            "retrieval_query": decision.standalone_query.strip() or query,
        }

    def decline(state: GraphState) -> dict:
        response = state.get(
            "final_response", "I can answer questions about the documents indexed in this session."
        )
        return {"final_response": redact_secrets(response)}

    def retrieve_initial(state: GraphState) -> dict:
        query = state.get("retrieval_query", state.get("query", ""))
        collection = state.get("collection", DEFAULT_COLLECTION)
        return {"chunks": agents.retrieve(query, settings.top_k, collection)}

    def answer(state: GraphState) -> dict:
        response = agents.answer(
            state.get("query", ""), state.get("chunks", []), state.get("history", [])
        )
        if not isinstance(response, AnswerOutput):
            response = AnswerOutput.model_validate(response)
        return {"answer": response}

    def grounding(state: GraphState) -> dict:
        answer_output = state.get("answer")
        if answer_output is None:
            return {"grounded": False, "grounding_report": "No validated answer was produced."}
        grounded, report = agents.ground(answer_output, state.get("chunks", []))
        return {"grounded": grounded, "grounding_report": report}

    def widen_retrieval(state: GraphState) -> dict:
        query = state.get("retrieval_query", state.get("query", ""))
        collection = state.get("collection", DEFAULT_COLLECTION)
        return {
            "retries": state.get("retries", 0) + 1,
            "chunks": agents.retrieve(query, settings.top_k_retry, collection),
        }

    def finalize(state: GraphState) -> dict:
        answer_output = state.get("answer")
        response = (
            answer_output.answer if answer_output else "I could not produce a validated answer."
        )
        return {"final_response": redact_secrets(response)}

    def fallback(_: GraphState) -> dict:
        return {
            "final_response": redact_secrets(
                "I could not find a well-supported answer in the uploaded documents."
            )
        }

    def input_route(state: GraphState) -> str:
        return "blocked" if state.get("input_verdict") == "blocked" else "safe"

    def scope_route(state: GraphState) -> str:
        if state.get("input_verdict") == "blocked":
            return "blocked"
        return "in_scope" if state.get("route") == "in_scope" else "off_topic"

    def grounding_route(state: GraphState) -> str:
        if state.get("grounded"):
            return "grounded"
        if state.get("retries", 0) < settings.max_retries:
            return "retry"
        return "exhausted"

    graph = StateGraph(GraphState)
    graph.add_node("input_guard", input_guard)
    graph.add_node("refuse", refuse)
    graph.add_node("router", router)
    graph.add_node("decline", decline)
    graph.add_node("retrieve", retrieve_initial)
    graph.add_node("answer", answer)
    graph.add_node("grounding_check", grounding)
    graph.add_node("widen_retrieval", widen_retrieval)
    graph.add_node("finalize", finalize)
    graph.add_node("fallback", fallback)

    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges("input_guard", input_route, {"blocked": "refuse", "safe": "router"})
    graph.add_edge("refuse", END)
    graph.add_conditional_edges(
        "router",
        scope_route,
        {"in_scope": "retrieve", "off_topic": "decline", "blocked": "refuse"},
    )
    graph.add_edge("decline", END)
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "grounding_check")
    graph.add_conditional_edges(
        "grounding_check",
        grounding_route,
        {"grounded": "finalize", "retry": "widen_retrieval", "exhausted": "fallback"},
    )
    graph.add_edge("widen_retrieval", "answer")
    graph.add_edge("finalize", END)
    graph.add_edge("fallback", END)
    return graph.compile()


def run_query(
    query: str,
    history: list[dict] | None = None,
    collection: str = DEFAULT_COLLECTION,
    services: GraphServices | None = None,
) -> GraphState:
    """Execute a user query against a session's collection and return the final graph state."""
    logger.info("run_query start: %r", query)
    with track_llm_calls() as call_count:
        result = build_graph(services).invoke(
            {"query": query, "retries": 0, "history": history or [], "collection": collection}
        )
    result["llm_calls"] = call_count[0]
    logger.info(
        "run_query done: route=%s grounded=%s retries=%s llm_calls=%s",
        result.get("route"),
        result.get("grounded"),
        result.get("retries"),
        call_count[0],
    )
    return result
