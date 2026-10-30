
from typing import TypedDict, Any

from langgraph.graph import StateGraph, END

from vector_store import search_chunks
from web_search import search_web
from answer_generator import answer_from_documents


# --------------------------------------------------
# 1. Define agent state
# --------------------------------------------------

class AgentState(TypedDict, total=False):
    question: str
    documents: list
    web_results: list[dict]
    evaluation: str
    retry_count: int
    retrieval_failed: bool
    answer: str
    sources: list[dict]
    error: str


# --------------------------------------------------
# 2. Retrieve relevant passages from ChromaDB
# --------------------------------------------------

def retrieve_documents(state: AgentState) -> dict:
    question = state["question"]
    retry_count = state.get("retry_count", 0)

    k = 8 if retry_count > 0 else 4

    try:
        documents = search_chunks(question, k=k)

        print(f"Retrieved {len(documents)} passages from ChromaDB.")

        if not documents:
            return {
                "documents": [],
                "evaluation": "INSUFFICIENT",
                "retrieval_failed": False,
            }

        return {
            "documents": documents,
            "evaluation": "PENDING",
            "retrieval_failed": False,
        }

    except Exception as exc:
        print(f"Local retrieval failed: {exc}")

        return {
            "documents": [],
            "evaluation": "INSUFFICIENT",
            "retrieval_failed": True,
            "error": str(exc),
        }


# --------------------------------------------------
# 3. Evaluate whether the PDF contains enough evidence
# --------------------------------------------------

def evaluate_evidence(state: AgentState) -> dict:
    if state.get("retrieval_failed"):
        return {"evaluation": "INSUFFICIENT"}

    documents = state.get("documents", [])

    if not documents:
        return {"evaluation": "INSUFFICIENT"}

    # Keep the evaluator simple and deterministic.
    # This checks whether retrieval returned meaningful text;
    # it is not a semantic entailment evaluation.
    usable_text = []

    for document in documents:
        if isinstance(document, dict):
            text = document.get("text", "")
        else:
            text = getattr(document, "page_content", str(document))

        if text and text.strip():
            usable_text.append(text.strip())

    if not usable_text:
        return {"evaluation": "INSUFFICIENT"}

    try:
        # Use the existing answer generator to evaluate relevance
        # through a separate Gemini call only if your project
        # already provides a dedicated evaluator.
        from answer_generator import evaluate_documents

        verdict = evaluate_documents(
            state["question"],
            usable_text,
        )

        verdict = str(verdict).strip().upper()

        if "INSUFFICIENT" in verdict:
            return {"evaluation": "INSUFFICIENT"}

        if "SUFFICIENT" in verdict:
            return {"evaluation": "SUFFICIENT"}

        return {"evaluation": "INSUFFICIENT"}

    except ImportError:
        # If evaluate_documents has not been implemented yet,
        # do not pretend that retrieved text is sufficient.
        print(
            "No evaluate_documents() function found; "
            "using web fallback after retrieval retry."
        )
        return {"evaluation": "INSUFFICIENT"}

    except Exception as exc:
        print(f"Evidence evaluation failed: {exc}")
        return {"evaluation": "EVALUATION_ERROR"}


# --------------------------------------------------
# 4. Decide the next step
# --------------------------------------------------

def route_after_evaluation(state: AgentState) -> str:
    if state.get("retrieval_failed"):
        return "web_search"

    evaluation = state.get("evaluation", "INSUFFICIENT")

    if evaluation == "SUFFICIENT":
        return "local_answer"

    if evaluation == "EVALUATION_ERROR":
        # Preserve the existing PDF-only fallback behavior.
        return "local_answer"

    if state.get("retry_count", 0) == 0:
        return "retry"

    return "web_search"


# --------------------------------------------------
# 5. Retry local retrieval once
# --------------------------------------------------

def retry_retrieval(state: AgentState) -> dict:
    print("Evidence insufficient. Retrying local retrieval...")

    return {
        "retry_count": state.get("retry_count", 0) + 1,
    }


# --------------------------------------------------
# 6. Search the web
# --------------------------------------------------

def perform_web_search(state: AgentState) -> dict:
    question = state["question"]

    print("Local evidence insufficient. Searching the web...")

    try:
        results = search_web(question, max_results=5)
    except Exception as exc:
        print(f"Web search failed: {exc}")
        results = []

    print(f"Received {len(results)} web results.")

    return {"web_results": results}


# --------------------------------------------------
# 7. Generate an answer from PDF evidence
# --------------------------------------------------

def generate_local_answer(state: AgentState) -> dict:
    question = state["question"]
    documents = state.get("documents", [])

    try:
        result = answer_from_documents(
            question,
            documents=documents,
        )

        # Support either a string or a structured return value.
        if isinstance(result, dict):
            answer = result.get("answer", "")
            sources = result.get("sources", [])
        else:
            answer = str(result)
            sources = []

        if not sources:
            sources = extract_pdf_sources(documents)

        return {
            "answer": answer,
            "sources": sources,
        }

    except Exception as exc:
        print(f"Local answer generation failed: {exc}")

        return {
            "answer": (
                "I couldn't generate an answer from the project "
                f"report because of an error: {exc}"
            ),
            "sources": extract_pdf_sources(documents),
        }


# --------------------------------------------------
# 8. Generate an answer using web fallback
# --------------------------------------------------

def generate_web_answer(state: AgentState) -> dict:
    question = state["question"]
    pdf_documents = state.get("documents", [])
    web_results = state.get("web_results", [])

    print("Generating an evidence-grounded answer with Gemini...")

    # Convert search results to the document structure expected
    # by answer_from_documents().
    web_documents = []

    for item in web_results:
        title = item.get("title", "Untitled")
        snippet = item.get("snippet", "")
        url = item.get("url", "")

        if not snippet.strip():
            continue

        web_documents.append({
            "text": (
                f"Web result title: {title}\n"
                f"Web result snippet: {snippet}\n"
                f"Web URL: {url}"
            ),
            "source": url or "Web search",
            "page": "Web",
        })

    # Include the PDF passages as context, but explicitly label
    # web results so the answer generator can distinguish them.
    combined_documents = list(pdf_documents) + web_documents

    try:
        result = answer_from_documents(
            question,
            documents=combined_documents,
        )

        if isinstance(result, dict):
            answer = result.get("answer", "")
        else:
            answer = str(result)

        # Only include web sources actually cited in the answer.
        cited_web_sources = []

        for item in web_results:
            url = item.get("url", "")

            if url and url in answer and url not in cited_web_sources:
                cited_web_sources.append(url)

        # If the answer generator cites no URLs, retain the
        # sources that were supplied to it rather than unrelated
        # PDF pages. This fallback is deliberately conservative.
        if not cited_web_sources:
            cited_web_sources = [
                item.get("url", "")
                for item in web_results
                if item.get("url")
            ]

        sources = [
            {
                "source": url,
                "page": "Web",
            }
            for url in cited_web_sources
        ]

        return {
            "answer": answer,
            "sources": sources,
        }

    except Exception as exc:
        print(f"Web answer generation failed: {exc}")

        # Do not invent a factual answer if Gemini is unavailable.
        fallback_lines = []

        for item in web_results:
            title = item.get("title", "Untitled")
            snippet = item.get("snippet", "")
            url = item.get("url", "")

            if snippet.strip():
                fallback_lines.append(
                    f"- {title}: {snippet} ({url})"
                )

        if fallback_lines:
            answer = (
                "I couldn't generate a synthesized answer because "
                "the language model request failed. Here are the "
                "retrieved web snippets to review:\n\n"
                + "\n".join(fallback_lines)
            )
        else:
            answer = (
                "I couldn't find usable web results or generate "
                "an answer. Please try again later."
            )

        return {
            "answer": answer,
            "sources": [
                {
                    "source": item.get("url", "Web search"),
                    "page": "Web",
                }
                for item in web_results
                if item.get("url")
            ],
        }


# --------------------------------------------------
# 9. Extract PDF source references
# --------------------------------------------------

def extract_pdf_sources(documents: list) -> list[dict]:
    sources = []
    seen = set()

    for document in documents:
        if isinstance(document, dict):
            metadata = document.get("metadata", {})
            source = document.get(
                "source",
                metadata.get("source", "Project_Report.pdf"),
            )
            page = document.get(
                "page",
                metadata.get("page", "Unknown"),
            )
        else:
            metadata = getattr(document, "metadata", {}) or {}
            source = metadata.get("source", "Project_Report.pdf")
            page = metadata.get("page", "Unknown")

        key = (str(source), str(page))

        if key not in seen:
            seen.add(key)
            sources.append({
                "source": str(source),
                "page": page,
            })

    return sources


# --------------------------------------------------
# 10. Build the LangGraph workflow
# --------------------------------------------------

workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("evaluate", evaluate_evidence)
workflow.add_node("retry", retry_retrieval)
workflow.add_node("web_search", perform_web_search)
workflow.add_node("local_answer", generate_local_answer)
workflow.add_node("web_answer", generate_web_answer)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "evaluate")

workflow.add_conditional_edges(
    "evaluate",
    route_after_evaluation,
    {
        "retry": "retry",
        "web_search": "web_search",
        "local_answer": "local_answer",
    },
)

workflow.add_edge("retry", "retrieve")
workflow.add_edge("web_search", "web_answer")
workflow.add_edge("local_answer", END)
workflow.add_edge("web_answer", END)

agent = workflow.compile()


# --------------------------------------------------
# 11. Public function used by test_agent.py
# --------------------------------------------------

def ask_agent(question: str) -> dict:
    if not question or not question.strip():
        return {
            "answer": "Please enter a question.",
            "sources": [],
        }

    initial_state: AgentState = {
        "question": question.strip(),
        "documents": [],
        "web_results": [],
        "evaluation": "PENDING",
        "retry_count": 0,
        "retrieval_failed": False,
        "answer": "",
        "sources": [],
    }

    result = agent.invoke(initial_state)

    return {
        "answer": result.get("answer", "No answer was generated."),
        "sources": result.get("sources", []),
    }
