"""Page styling and small display helpers for MediBot."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import streamlit as st


EXAMPLE_QUESTIONS = [
    "Summarize the main points",
    "What symptoms are described?",
    "What treatments are mentioned?",
]


@st.cache_data(show_spinner=False)
def count_pdf_pages(pdf_bytes: bytes) -> int | None:
    """Read and cache the number of pages so it isn't recounted on every rerun."""
    try:
        from pypdf import PdfReader

        return len(PdfReader(BytesIO(pdf_bytes)).pages)
    except Exception:
        # Let the PDF preparation step show the more useful error for a broken file.
        return None


def render_page_header() -> None:
    """Show the shared styles and the MediBot hero banner."""
    # CSS keeps visual styling in one place instead of mixing it with app logic.
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp, [data-testid="stAppViewContainer"] { background: #11191c; color: #eaf2f3; }
        .block-container { max-width: 1120px; padding: 2rem 2rem 4rem; }
        header[data-testid="stHeader"] { background: transparent; }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #f4f8f8; border: 1px solid #d9e5e6; border-radius: 18px;
        }
        [data-testid="stVerticalBlockBorderWrapper"] .stMarkdown,
        [data-testid="stVerticalBlockBorderWrapper"] .stMarkdown p,
        [data-testid="stVerticalBlockBorderWrapper"] h2,
        [data-testid="stVerticalBlockBorderWrapper"] label,
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stWidgetLabel"] {
            color: #1d3337;
        }
        .hero { padding: 2.1rem 2.2rem; border-radius: 22px; margin-bottom: 1.4rem;
                background: #1b292d; border: 1px solid #2d4246; border-left: 4px solid #48c4ad; }
        .hero .eyebrow { text-transform: uppercase; letter-spacing: .13em; font-size: .72rem;
                         font-weight: 700; color: #64d6bd; margin-bottom: .65rem; }
        .hero h1 { margin: 0 0 .45rem 0; color: #f1f8f7; font-size: 2.55rem; }
        .hero p { margin: 0; color: #b7c9cb; font-size: 1.02rem; max-width: 700px; line-height: 1.6; }
        .hero-note { margin-top: 1rem; color: #8fa8aa; font-size: .86rem; }
        [data-testid="stFileUploader"] section { border-radius: 14px; background: #eaf2f2; border-color: #b8d3d2; }
        [data-testid="stFileUploader"] section, [data-testid="stFileUploader"] section * { color: #23383c; }
        [data-testid="stChatInput"] textarea { background: #f5f9f9; color: #203438; border-radius: 14px; }
        .step-label { color: #188a7b; font-size: .74rem; text-transform: uppercase;
                      letter-spacing: .12em; font-weight: 700; margin-bottom: .1rem; }
        .muted-copy { color: #506a6f; line-height: 1.55; }
        .example-card { min-height: 118px; padding: 1.05rem; border-radius: 15px;
                        background: #1b292d; border: 1px solid #304448; border-top: 3px solid #48c4ad; }
        .example-number { color: #65d5bd; font-size: .7rem; font-weight: 700;
                          letter-spacing: .1em; text-transform: uppercase; margin-bottom: .65rem; }
        .example-question { color: #eaf2f3; font-weight: 600; line-height: 1.45; }
        .document-meta { color: #d4e1e2; font-size: .9rem; padding: .5rem 0; }
        .stButton > button { border-radius: 11px; }
        @media (max-width: 700px) { .block-container { padding: 1.1rem 1rem 3rem; }
            .hero { padding: 1.5rem; } .hero h1 { font-size: 2rem; } }
        </style>
        <div class="hero"><div class="eyebrow">Medical knowledge, made easier</div>
        <h1>🩺 MediBot</h1>
        <p>Ask questions about a medical reference and get clear answers grounded in the document’s pages.</p>
        <div class="hero-note">Upload a PDF · Ask in plain language · Check the cited source pages</div></div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(docs: list[Any]) -> None:
    """Show source names, page numbers, and retrieved text excerpts."""
    # Each retrieved chunk carries its original PDF page in its metadata.
    with st.expander(f"📚 Retrieved sources ({len(docs)})", expanded=False):
        for index, doc in enumerate(docs, start=1):
            source = Path(str(doc.metadata.get("source", "Uploaded PDF"))).name
            page = doc.metadata.get("page")
            if isinstance(page, int):
                st.markdown(f"**Source {index} · PDF page {int(page) + 1}**")
            else:
                st.markdown(f"**Source {index}**")
            st.caption(f"{source}{' · OCR text' if doc.metadata.get('ocr') else ''}")
            st.caption(doc.page_content)
            if index < len(docs):
                st.divider()


def render_examples() -> None:
    """Show the static sample-question cards on the welcome page."""
    st.markdown("#### Example questions")
    columns = st.columns(3)
    for number, (column, question) in enumerate(zip(columns, EXAMPLE_QUESTIONS), start=1):
        with column:
            st.markdown(
                f'<div class="example-card"><div class="example-number">Example 0{number}</div>'
                f'<div class="example-question">{question}</div></div>',
                unsafe_allow_html=True,
            )


def render_disclaimer() -> None:
    with st.expander("A quick note about medical information"):
        st.write(
            "MediBot is an educational reference tool, not a substitute for professional medical advice, "
            "diagnosis, or treatment. For emergencies, contact local emergency services."
        )
