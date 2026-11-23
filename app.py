"""Streamlit page and session flow for MediBot."""

from __future__ import annotations

import hashlib
import os
from typing import Any

import streamlit as st

from rag import answer_question, build_vectorstore
from ui import (
    EXAMPLE_QUESTIONS,
    count_pdf_pages,
    render_disclaimer,
    render_examples,
    render_page_header,
    render_sources,
)


# Set the browser tab title, icon, and wide layout before drawing the page.
st.set_page_config(
    page_title="MediBot | Medical Knowledge Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading the document search model…")
def load_embeddings() -> Any:
    """Load the sentence embedding model once and reuse it in this app process."""
    # This model turns text into vectors that FAISS can search by meaning.
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_hf_token() -> str:
    """Read the app owner's private token from deployment secrets or environment."""
    # Streamlit secrets are used after deployment; an environment variable works locally.
    try:
        token = st.secrets.get("HF_TOKEN", "")
    except Exception:
        token = ""
    return str(token or os.getenv("HF_TOKEN", "")).strip()


def initialize_session() -> None:
    """Set default values used to remember the current chat and PDF."""
    # Streamlit reruns this script after interactions, so session state keeps data between reruns.
    defaults = {
        "messages": [],
        "active_document_key": None,
        "active_pdf_name": None,
        "active_page_range": None,
        "vectorstore": None,
        "question_count": 0,
        "pending_question": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clear_current_document() -> None:
    """Forget the current PDF and chat when the user removes the upload."""
    st.session_state["pdf_upload"] = None
    st.session_state.active_document_key = None
    st.session_state.active_pdf_name = None
    st.session_state.active_page_range = None
    st.session_state.vectorstore = None
    st.session_state.messages = []
    st.session_state.pending_question = None


def queue_question(question: str) -> None:
    """Remember a sample question until the next Streamlit rerun."""
    st.session_state.pending_question = question


def render_upload_section() -> tuple[Any, tuple[int, int] | None]:
    """Show the PDF uploader and return the upload plus selected page range."""
    # The upload panel returns the selected PDF to the rest of the app.
    with st.container(border=True):
        st.markdown('<div class="step-label">Step 1 · Add your reference</div>', unsafe_allow_html=True)
        st.subheader("Start with a reference PDF")
        st.markdown(
            '<div class="muted-copy">Choose a medical encyclopedia or another trusted reference. '
            "MediBot prepares it for search in this session.</div>",
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Upload a PDF", type=["pdf"], label_visibility="collapsed", key="pdf_upload"
        )
        file_col, action_col = st.columns([3, 1])
        if uploaded:
            size_mb = uploaded.size / (1024 * 1024)
            file_col.markdown(
                f'<div class="document-meta">📄 <b>{uploaded.name}</b> · {size_mb:.2f} MB</div>',
                unsafe_allow_html=True,
            )
            action_col.button("Remove PDF", on_click=clear_current_document, use_container_width=True)
            page_count = count_pdf_pages(uploaded.getvalue())
            page_range = None
            if page_count and page_count > 50:
                file_key = hashlib.sha256(uploaded.getvalue()).hexdigest()[:12]
                choice = st.radio(
                    f"Choose how much of this {page_count}-page PDF to prepare",
                    [
                        "Quick start · first 50 pages",
                        "Choose a page range",
                        f"Full document · all {page_count} pages",
                    ],
                    horizontal=True,
                    key=f"pdf_scope_{file_key}",
                )
                if choice.startswith("Quick start"):
                    page_range = (1, 50)
                    st.caption(
                        "Quick start searches pages 1–50 only. Choose Full document if your answer "
                        "may be later in the PDF."
                    )
                elif choice.startswith("Choose"):
                    page_range = st.slider(
                        "Select the pages to search",
                        min_value=1,
                        max_value=page_count,
                        value=(1, min(50, page_count)),
                        key=f"pdf_pages_{file_key}",
                    )
                    st.caption(
                        f"MediBot will search pages {page_range[0]}–{page_range[1]} only. "
                        "Use the full-document option to include every page."
                    )
                else:
                    page_range = (1, page_count)
                    st.caption("The full document is selected; scanned PDFs may take longer to prepare.")
            elif page_count:
                page_range = (1, page_count)
                st.caption(f"{page_count} pages will be prepared.")
        else:
            page_range = None
            file_col.caption("Text PDFs and scanned PDFs are supported. Scanned pages may take longer to read.")
        st.caption("Your PDF is processed for this session. Choose another file above to replace it.")
    return uploaded, page_range


def render_get_started_section() -> None:
    """Explain the two steps in plain language."""
    with st.container(border=True):
        st.markdown('<div class="step-label">Step 2 · Ask</div>', unsafe_allow_html=True)
        st.subheader("Get started")
        st.markdown(
            '<div class="muted-copy">After the PDF is ready, ask your own question or choose an '
            "example below. Answers include the passages and page numbers used.</div>",
            unsafe_allow_html=True,
        )


def prepare_document(uploaded: Any, page_range: tuple[int, int] | None) -> None:
    """Build a search index when the selected PDF changes."""
    # A content hash lets us reuse the index while the user asks follow-up questions.
    pdf_bytes = uploaded.getvalue()
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
    document_key = f"{pdf_hash}:{page_range or 'all'}"
    if st.session_state.active_document_key == document_key:
        return

    # Drop the previous document before processing so it cannot be used by mistake.
    st.session_state.vectorstore = None
    st.session_state.active_document_key = None
    st.session_state.active_pdf_name = None
    st.session_state.active_page_range = None
    st.session_state.messages = []

    # Build a fresh index and show which real processing step is underway.
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        with st.status("Preparing your PDF…", expanded=True) as status:
            status.update(label="Loading the document search model…", state="running")
            embeddings = load_embeddings()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

            def show_stage(label: str) -> None:
                status.update(label=label, state="running")

            if page_range:
                status.update(
                    label=f"Preparing PDF pages {page_range[0]}–{page_range[1]}…",
                    state="running",
                )
            vectorstore = build_vectorstore(
                pdf_bytes,
                uploaded.name,
                embeddings,
                splitter,
                page_range=page_range,
                on_stage=show_stage,
            )
            status.update(label="PDF ready to use", state="complete", expanded=False)

        st.session_state.vectorstore = vectorstore
        st.session_state.active_document_key = document_key
        st.session_state.active_pdf_name = uploaded.name
        st.session_state.active_page_range = page_range
    except ValueError as exc:
        st.error(f"Couldn't prepare this PDF: {exc}")
        st.stop()
    except Exception:
        st.error(
            "Couldn't read this PDF. Check that it opens normally, isn't password-protected, "
            "and try uploading it again."
        )
        st.stop()


def create_model_client() -> Any:
    """Connect to the configured Hugging Face model."""
    # Visitors don't provide a token; the app owner configures it privately once.
    token = get_hf_token()
    if not token:
        st.error(
            "The chat service is not configured yet. The app owner needs to add the Hugging Face "
            "token in the app’s private settings."
        )
        st.stop()

    try:
        from huggingface_hub import InferenceClient

        # Route this model through Featherless AI via the Hugging Face router.
        # With an HF token, huggingface_hub sends provider requests
        # to router.huggingface.co and uses the HF_TOKEN for authentication.
        return InferenceClient(provider="featherless-ai", token=token)
    except Exception as exc:
        st.error(f"Could not set up the Hugging Face model: {exc}")
        st.stop()


def render_chat(client: Any) -> None:
    """Display chat history, sample prompts, and the question input."""
    # The simple demo cap is tracked for this visitor's current Streamlit session.
    remaining = max(0, 10 - st.session_state.question_count)
    title_col, limit_col, clear_col = st.columns([3, 1.5, 1])
    page_range = st.session_state.active_page_range
    page_label = f" · pages {page_range[0]}–{page_range[1]}" if page_range else ""
    title_col.markdown(f"### Chat about {st.session_state.active_pdf_name}{page_label}")
    limit_col.caption(f"{remaining} / 10 questions left")
    if st.session_state.messages:
        clear_col.button(
            "Clear chat", on_click=st.session_state.messages.clear, use_container_width=True
        )

    # Repaint saved turns after each Streamlit rerun.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                render_sources(message["sources"])

    if not st.session_state.messages and remaining:
        st.markdown("**Choose a sample question to get started**")
        columns = st.columns(3)
        for index, (column, sample) in enumerate(zip(columns, EXAMPLE_QUESTIONS)):
            column.button(
                sample,
                key=f"sample_{index}",
                on_click=queue_question,
                args=(sample,),
                use_container_width=True,
            )

    # Read either typed text or a sample button, then clear the queued sample.
    question = st.chat_input("Ask a question about the uploaded reference…") if remaining else None
    queued_question = st.session_state.pending_question
    st.session_state.pending_question = None
    question = question or queued_question
    if question and not remaining:
        st.info("You've used the 10-question demo limit for this session.")
        return
    if not question:
        if not remaining:
            st.info("You've used the 10-question demo limit for this session.")
        return

    # Retrieve relevant PDF passages and ask the model to answer from those passages.
    st.session_state.question_count += 1
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the reference and preparing an answer…"):
                answer, sources = answer_question(question, st.session_state.vectorstore, client)
            st.markdown(answer)
            render_sources(sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as exc:
            st.error(
                "The model couldn't answer right now. Check that the Hugging Face token and "
                "model are available, then try again."
            )
            st.caption(str(exc))


def main() -> None:
    """Run the page in order: layout, upload, index, then chat."""
    # Page order: welcome and upload, then PDF preparation, then chat.
    initialize_session()
    render_page_header()
    uploaded, page_range = render_upload_section()
    render_get_started_section()
    render_examples()
    render_disclaimer()

    if not uploaded:
        st.info("Upload a PDF to prepare the reference and start asking questions.")
        return

    prepare_document(uploaded, page_range)
    st.success(f"Document ready: **{st.session_state.active_pdf_name}**")
    render_chat(create_model_client())


if __name__ == "__main__":
    main()
