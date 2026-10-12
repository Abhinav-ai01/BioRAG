
import re
from rank_bm25 import BM25Okapi

from app.retriever import Retriever


def tokenize(text):
    return re.findall(
        r"[A-Za-z0-9]+(?:[._:/-][A-Za-z0-9]+)*",
        text.lower(),
    )


class HybridRetriever:
    def __init__(self):
        self.dense = Retriever()
        self.chunks = self.dense.chunks
        self.bm25 = None
        self._build_bm25()

    def _build_bm25(self):
        """Build the BM25 keyword index from current chunks."""
        self.chunks = self.dense.chunks

        if not self.chunks:
            raise ValueError(
                "No indexed chunks found. Add a readable PDF to data/raw."
            )

        tokenized_chunks = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_chunks)

    def rebuild_index(self):
        """Refresh both the dense and keyword indexes."""
        count = self.dense.rebuild_index()
        self.chunks = self.dense.chunks
        self._build_bm25()
        return count

    def search(self, question, top_k=5, candidate_k=20):
        if not question or not question.strip():
            return []

        candidate_k = min(candidate_k, len(self.chunks))

        if candidate_k == 0:
            return []

        # Dense vector search
        dense_results = self.dense.search(
            question,
            top_k=candidate_k,
        )

        # BM25 keyword search
        keyword_scores = self.bm25.get_scores(tokenize(question))

        keyword_ids = sorted(
            range(len(keyword_scores)),
            key=lambda idx: keyword_scores[idx],
            reverse=True,
        )[:candidate_k]

        # Reciprocal Rank Fusion
        fused_scores = {}

        for rank, item in enumerate(dense_results, start=1):
            chunk_id = item["chunk_id"]
            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + 1 / (60 + rank)
            )

        for rank, idx in enumerate(keyword_ids, start=1):
            chunk_id = self.chunks[idx]["chunk_id"]
            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + 1 / (60 + rank)
            )

        chunk_by_id = {
            chunk["chunk_id"]: chunk
            for chunk in self.chunks
        }

        ranked_ids = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )[:top_k]

        return [
            {
                **chunk_by_id[chunk_id],
                "rrf_score": fused_scores[chunk_id],
            }
            for chunk_id in ranked_ids
        ]


if __name__ == "__main__":
    searcher = HybridRetriever()

    question = "What is the main finding of this research paper?"
    results = searcher.search(question, top_k=5)

    for item in results:
        print(f"\n{item['source']} — page {item['page']}")
        print("RRF score:", round(item["rrf_score"], 5))
        print(item["text"][:300])
