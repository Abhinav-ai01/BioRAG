
from app.reranker import PassageReranker


TEST_CASES = [
    {
        "question": "What is the title of the research paper?",
        "relevant_pages": [1],
    },
    {
        "question": "Who are the authors of the research paper?",
        "relevant_pages": [1],
    },
    {
        "question": "What are the main findings of the study?",
        "relevant_pages": [],  # Annotate after checking the paper.
    },
    {
        "question": "What limitations did the researchers mention?",
        "relevant_pages": [],  # Annotate after checking the paper.
    },
]


def recall_at_k(results, relevant_pages, k=5):
    if not relevant_pages:
        return None

    retrieved_pages = {
        item["page"] for item in results[:k]
    }

    return len(retrieved_pages & set(relevant_pages)) / len(
        set(relevant_pages)
    )


def reciprocal_rank(results, relevant_pages):
    if not relevant_pages:
        return None

    relevant_pages = set(relevant_pages)

    for rank, item in enumerate(results, start=1):
        if item["page"] in relevant_pages:
            return 1.0 / rank

    return 0.0


def main():
    retriever = PassageReranker()
    recall_scores = []
    reciprocal_ranks = []

    for case in TEST_CASES:
        question = case["question"]
        relevant_pages = case["relevant_pages"]

        results = retriever.search(
            question,
            top_k=5,
            candidate_k=20,
        )

        recall = recall_at_k(results, relevant_pages)
        mrr = reciprocal_rank(results, relevant_pages)

        print("\n" + "=" * 60)
        print("Question:", question)
        print(
            "Retrieved pages:",
            [
                (item["source"], item["page"])
                for item in results
            ],
        )

        if recall is None:
            print("Recall@5: Not scored — annotate relevant pages.")
            print("Reciprocal rank: Not scored.")
        else:
            print(f"Recall@5: {recall:.2f}")
            print(f"Reciprocal rank: {mrr:.2f}")

            recall_scores.append(recall)
            reciprocal_ranks.append(mrr)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("Scored questions:", len(recall_scores))

    if recall_scores:
        print(
            f"Mean Recall@5: "
            f"{sum(recall_scores) / len(recall_scores):.3f}"
        )
        print(
            f"MRR: "
            f"{sum(reciprocal_ranks) / len(reciprocal_ranks):.3f}"
        )

    print(
        "\nNote: These metrics use page numbers only. "
        "If multiple PDFs are indexed, use source-and-page pairs "
        "to avoid matching the wrong document."
    )


if __name__ == "__main__":
    main()
