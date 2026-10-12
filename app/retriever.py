
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from app.parser import extract_pdfs
from app.chunker import create_chunks


INDEX_DIR = Path("indexes")
INDEX_PATH = INDEX_DIR / "biomedical.faiss"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = None
        self.chunks = []
        self.rebuild_index()

    def rebuild_index(self):
        """Re-extract PDFs and rebuild the FAISS index."""
        pages = extract_pdfs()
        chunks = create_chunks(pages)

        if not chunks:
            raise ValueError(
                "No text found. Add a readable PDF to data/raw."
            )

        texts = [chunk["text"] for chunk in chunks]

        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        INDEX_DIR.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(INDEX_PATH))

        with open(CHUNKS_PATH, "w", encoding="utf-8") as file:
            json.dump(
                chunks,
                file,
                ensure_ascii=False,
                indent=2,
            )

        # Replace in-memory data only after building succeeds.
        self.index = index
        self.chunks = chunks

        return len(self.chunks)

    def search(self, question, top_k=5):
        if not question or not question.strip():
            return []

        if self.index is None or not self.chunks:
            return []

        query_vector = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_vector, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append({
                    **self.chunks[int(idx)],
                    "score": float(score),
                })

        return results


if __name__ == "__main__":
    retriever = Retriever()

    print(f"\nIndexed {len(retriever.chunks)} chunks.")

    question = "What is the main finding of this research paper?"
    results = retriever.search(question, top_k=3)

    for result in results:
        print("\nSource:", result["source"])
        print("Page:", result["page"])
        print("Similarity:", round(result["score"], 3))
        print(result["text"][:300])
