
import hashlib
from pathlib import Path

import streamlit as st

from app.rag import BioRAG


# ==================================================
# CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="BioRAG | Biomedical Research Assistant",
    page_icon="🧬",
    layout="wide",
)

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# RAG INITIALIZATION
# ==================================================

@st.cache_resource
def load_rag():
    return BioRAG()


def rebuild_search_index():
    """
    Rebuild the dense FAISS index and BM25 index.
    BioRAG -> PassageReranker -> HybridRetriever -> Retriever
    """
    rag = load_rag()
    chunk_count = rag.retriever.rebuild_index()

    # Discard the old RAG instance so subsequent questions
    # use the refreshed indexes.
    load_rag.clear()

    return chunk_count


# ==================================================
# HEADER
# ==================================================

st.title("🧬 BioRAG")
st.subheader("Biomedical Research Assistant")

st.write(
    "Upload biomedical research papers and ask questions about "
    "their findings, methodology, results, and limitations. "
    "BioRAG uses local Ollama generation and displays supporting sources."
)

st.divider()


# ==================================================
# PDF UPLOAD
# ==================================================

st.markdown("## 📄 Research Papers")

uploaded_pdf = st.file_uploader(
    "Upload a biomedical research paper (PDF)",
    type=["pdf"],
    help="PDFs are saved in the local data/raw folder.",
)

if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.getvalue()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    existing_file = None

    # Avoid saving identical PDF content twice.
    for pdf_path in DATA_DIR.glob("*.pdf"):
        try:
            existing_hash = hashlib.sha256(
                pdf_path.read_bytes()
            ).hexdigest()

            if existing_hash == file_hash:
                existing_file = pdf_path
                break
        except OSError:
            continue

    if existing_file:
        st.info(
            f"This PDF is already saved as **{existing_file.name}**."
        )
    else:
        # Use only the filename, not any uploaded directory path.
        safe_name = Path(uploaded_pdf.name).name

        if not safe_name.lower().endswith(".pdf"):
            safe_name += ".pdf"

        save_path = DATA_DIR / safe_name

        # Do not overwrite a different PDF with the same filename.
        if save_path.exists():
            save_path = DATA_DIR / (
                f"{Path(safe_name).stem}_{file_hash[:8]}.pdf"
            )

        try:
            save_path.write_bytes(pdf_bytes)
            st.success(f"PDF saved: **{save_path.name}**")
            st.info(
                "Rebuild the search index below to make this paper "
                "available for retrieval."
            )
        except OSError as exc:
            st.error(f"Could not save the PDF: {exc}")


# ==================================================
# SAVED PAPERS
# ==================================================

saved_pdfs = sorted(DATA_DIR.glob("*.pdf"))

with st.expander(
    f"View saved research papers ({len(saved_pdfs)})"
):
    if saved_pdfs:
        for pdf_path in saved_pdfs:
            st.write(f"📄 {pdf_path.name}")
    else:
        st.info("No research PDFs have been saved yet.")


# ==================================================
# INDEX MANAGEMENT
# ==================================================

st.divider()
st.markdown("## 🔄 Search Index")

st.write(
    "Rebuild the indexes after uploading a PDF. "
    "This refreshes the FAISS vector index and BM25 keyword index "
    "used by the hybrid retrieval pipeline."
)

if st.button(
    "Rebuild Search Index",
    type="secondary",
    disabled=(len(saved_pdfs) == 0),
):
    try:
        with st.spinner(
            "Rebuilding FAISS and BM25 indexes. Please wait..."
        ):
            chunk_count = rebuild_search_index()

        st.success(
            f"Search index rebuilt successfully. "
            f"Indexed {chunk_count} chunks."
        )

    except Exception as exc:
        st.error(f"Index rebuilding failed: {exc}")


# ==================================================
# QUESTION-ANSWERING
# ==================================================

st.divider()
st.markdown("## 💬 Ask a Research Question")

question = st.text_area(
    "Enter your research question",
    placeholder=(
        "Example: What are the main findings of the study?"
    ),
    height=110,
    key="research_question",
)

if st.button("Ask BioRAG", type="primary"):
    if not question.strip():
        st.warning("Please enter a research question.")

    else:
        try:
            with st.spinner(
                "Searching research evidence and generating an answer..."
            ):
                rag = load_rag()
                result = rag.ask(question.strip())

            answer = result.get(
                "answer",
                "No answer was returned.",
            )
            sources = result.get("sources", [])

            st.markdown("### 📝 Answer")
            st.markdown(answer)

            st.markdown("### 📚 Supporting Sources")

            if not sources:
                st.info(
                    "No supporting sources were returned. "
                    "Try another question or rebuild the index "
                    "if you recently uploaded a PDF."
                )

            else:
                for i, source in enumerate(sources, start=1):
                    source_name = source.get(
                        "source",
                        "Unknown document",
                    )
                    page_number = source.get(
                        "page",
                        "Unknown",
                    )

                    heading = (
                        f"Source {i}: {source_name} "
                        f"— Page {page_number}"
                    )

                    with st.expander(heading):
                        st.write(
                            source.get(
                                "text",
                                "No source text available.",
                            )
                        )

                        if source.get("score") is not None:
                            st.caption(
                                f"Dense similarity: "
                                f"{source['score']:.4f}"
                            )

                        if source.get("rrf_score") is not None:
                            st.caption(
                                f"Hybrid RRF score: "
                                f"{source['rrf_score']:.5f}"
                            )

                        if source.get("rerank_score") is not None:
                            st.caption(
                                f"Reranker score: "
                                f"{source['rerank_score']:.4f}"
                            )

        except Exception as exc:
            st.error(f"Question processing failed: {exc}")


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "BioRAG | Local biomedical research assistant. "
    "Verify answers against the original paper. "
    "This application does not replace professional medical advice."
)
