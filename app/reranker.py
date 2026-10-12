
from sentence_transformers import CrossEncoder

from app.hybrid import HybridRetriever


class PassageReranker:
    def __init__(self):
        self.retriever = HybridRetriever()

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rebuild_index(self):
        """Rebuild the underlying vector index and refresh hybrid search."""
        count = self.retriever.rebuild_index()
        return count

    def search(self, question, top_k=5, candidate_k=20):
        candidates = self.retriever.search(
            question,
            top_k=candidate_k,
        )

        if not candidates:
            return []

        pairs = [
            (question, item["text"])
            for item in candidates
        ]

        scores = self.model.predict(pairs)

        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(score)

        candidates.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return candidates[:top_k]


if __name__ == "__main__":
    reranker = PassageReranker()

    question = "What is the main finding of this research paper?"
    results = reranker.search(question, top_k=5)

    for item in results:
        print(f"\n{item['source']} — page {item['page']}")
        print("Reranker score:", round(item["rerank_score"], 4))
        print(item["text"][:300])
