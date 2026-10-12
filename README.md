# 🧬 BioRAG — Biomedical Research Assistant

A Retrieval-Augmented Generation (RAG) application that helps users explore biomedical research papers through natural-language questions, hybrid information retrieval, and source-grounded answers.

BioRAG combines semantic search, keyword retrieval, and cross-encoder reranking with a locally hosted language model to help users find relevant evidence in research documents.

## ✨ Features

- **PDF ingestion:** Extracts text from biomedical research papers.
- **Semantic search:** Uses Sentence Transformers embeddings and FAISS.
- **Hybrid retrieval:** Combines dense vector search with BM25 keyword retrieval.
- **Reciprocal Rank Fusion:** Combines rankings from semantic and keyword search.
- **Passage reranking:** Uses a Cross-Encoder to rerank candidate passages.
- **Local LLM inference:** Uses Ollama with the `qwen2.5:3b` model.
- **Source inspection:** Displays supporting document passages and page references.
- **Interactive interface:** Provides a Streamlit web application.
- **Retrieval evaluation:** Includes scripts for basic evaluation and retrieval metrics.

## 🏗️ Architecture

```text
Biomedical Research PDFs
          |
          v
     PDF Extraction
          |
          v
      Text Chunking
          |
          v
   Sentence Transformers
          |
          v
       FAISS Index
          |
          +------------------+
          |                  |
          v                  v
    Dense Retrieval      BM25 Search
          |                  |
          +--------+---------+
                   |
                   v
      Reciprocal Rank Fusion
                   |
                   v
       Cross-Encoder Reranking
                   |
                   v
         Relevant Passages
                   |
                   v
           Ollama LLM
                   |
                   v
       Answer + Source Passages
                   |
                   v
          Streamlit Interface
```

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| User interface | Streamlit |
| Embeddings | Sentence Transformers |
| Vector search | FAISS |
| Keyword retrieval | BM25 (`rank-bm25`) |
| Reranking | Cross-Encoder |
| Local language model | Ollama, Qwen 2.5 3B |
| PDF processing | PyMuPDF |
| Evaluation | Custom Python scripts |

## 📂 Project Structure

```text
BioRAG/
├── app/
│   ├── __init__.py
│   ├── parser.py
│   ├── chunker.py
│   ├── retriever.py
│   ├── hybrid.py
│   ├── reranker.py
│   ├── rag.py
│   └── ui.py
├── data/
│   └── raw/
├── indexes/
├── .gitignore
├── evaluate_rag.py
├── evaluate_retrieval.py
├── test_rag.py
└── README.md
```

## ⚙️ Prerequisites

- Python 3.10 or a compatible Python version
- Git
- Ollama
- Sufficient RAM and disk space for embedding and reranking models

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Abhinav-ai01/BioRAG.git
cd BioRAG
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run commands directly with:

```powershell
.\.venv\Scripts\python.exe --version
```

### 3. Install dependencies

Install the dependencies used by the project:

```powershell
pip install pymupdf sentence-transformers faiss-cpu rank-bm25 numpy ollama streamlit
```

If the project has a `requirements.txt`, you can instead run:

```powershell
pip install -r requirements.txt
```

### 4. Install and start Ollama

Install Ollama from [ollama.com](https://ollama.com/).

Download the model:

```powershell
ollama pull qwen2.5:3b
```

If the `ollama` command is not on your PATH, use the installed executable or configure your PATH.

### 5. Add research papers

Place your own readable biomedical PDF files in:

```text
data/raw/
```

Do not commit copyrighted or confidential research documents unless you have permission to distribute them.

### 6. Launch the application

From the repository root:

```powershell
python -m streamlit run app/ui.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

Upload a research paper, rebuild the search index, and ask questions about the paper.

## 🧪 Evaluation

The project includes:

- `test_rag.py` — manually tests question answering.
- `evaluate_rag.py` — records basic answer-generation and latency results.
- `evaluate_retrieval.py` — calculates Recall@5 and reciprocal-rank metrics for annotated relevant pages.

Retrieval metrics depend on correctly annotated ground-truth pages. Run the evaluation scripts locally and report actual measured results rather than assuming a particular accuracy or performance level.

## 🔒 Privacy and Security

- Ollama provides local LLM inference.
- Do not commit API keys, `.env` files, private PDFs, virtual environments, or generated indexes.
- Verify that your `.gitignore` excludes sensitive and machine-generated files.
- Review source documents and generated answers before relying on research conclusions.

## ⚠️ Limitations

- Answer quality depends on PDF extraction, chunking, retrieval quality, and the selected language model.
- Source passages help users verify answers but do not guarantee factual correctness.
- Evaluation is currently based on custom scripts and requires more extensive validation.
- This tool supports research exploration; it is not a clinical
