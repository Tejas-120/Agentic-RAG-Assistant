
import sys
from pathlib import Path

import streamlit as st

# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent_graph import ask_agent


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Agentic RAG | Project Assistant",
    page_icon="📚",
    layout="wide",
)

# --------------------------------------------------
# Styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #777;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">📚 Agentic RAG Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about your project report. '
    'If the report does not contain enough evidence, '
    'the agent can search the web.'
    '</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("About this application")

    st.write(
        "An Agentic RAG assistant that retrieves passages from "
        "your project PDF, evaluates evidence, retries retrieval "
        "when needed, and can use web search as a fallback."
    )

    st.divider()

    st.subheader("Capabilities")
    st.write("🔎 Semantic search with ChromaDB")
    st.write("🤖 Evidence evaluation")
    st.write("🔄 Retrieval retry")
    st.write("🌐 Web-search fallback")
    st.write("📄 Source references")
    st.write("💬 Conversational interface")

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("pending_question", None)
        st.rerun()

# --------------------------------------------------
# Initialize conversation
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# Source rendering
# --------------------------------------------------

def display_sources(sources):
    if not sources:
        st.info("No sources were returned.")
        return

    with st.expander("View sources", expanded=False):
        for index, source in enumerate(sources, start=1):
            source_name = source.get("source", "Unknown source")
            page = source.get("page", "Unknown")
            text = source.get("text", "")

            if page == "Web":
                st.markdown(f"**Source {index}: Web**")
                st.markdown(f"[Open source]({source_name})")

                if text:
                    st.write(text)

            else:
                st.markdown(
                    f"**Source {index}: {source_name} — Page {page}**"
                )

                if text:
                    st.write(text)
                else:
                    st.caption(
                        "The agent returned a source reference "
                        "but no passage text."
                    )

            st.divider()


# --------------------------------------------------
# Suggested questions
# --------------------------------------------------

st.subheader("Try a question")

suggestions = [
    "Which deep learning models were used in the project?",
    "What classification results were reported?",
    "How does the soft voting ensemble work?",
]

columns = st.columns(len(suggestions))

for index, suggestion in enumerate(suggestions):
    with columns[index]:
        if st.button(
            suggestion,
            key=f"suggestion_{index}",
            use_container_width=True,
        ):
            st.session_state.pending_question = suggestion

# --------------------------------------------------
# Display chat history
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            display_sources(message.get("sources", []))

# --------------------------------------------------
# Get user question
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about your project report..."
)

if not question and "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

# --------------------------------------------------
# Run the agent
# --------------------------------------------------

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(
            "Retrieving evidence, evaluating results, "
            "and preparing your answer..."
        ):
            try:
                result = ask_agent(question)

                answer = result.get(
                    "answer",
                    "No answer was generated.",
                )

                sources = result.get("sources", [])

                st.markdown(answer)
                display_sources(sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as exc:
                st.error(
                    "Something went wrong while answering your "
                    "question. Check the VS Code terminal for details."
                )

                print(f"Agent error: {exc}")
