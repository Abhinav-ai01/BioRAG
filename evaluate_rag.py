
import json
import time
from pathlib import Path

from app.rag import BioRAG


TEST_QUESTIONS = [
    {
        "question": "What is the title of the research paper?",
        "expected_page": 1,
    },
    {
        "question": "Who are the authors of the research paper?",
        "expected_page": 1,
    },
    {
        "question": "What are the main findings of the study?",
        "expected_page": None,
    },
    {
        "question": "What limitations did the researchers mention?",
        "expected_page": None,
    },
]


def main():
    rag = BioRAG()
    results = []

    for item in TEST_QUESTIONS:
        question = item["question"]
        start = time.perf_counter()

        try:
            result = rag.ask(question)
            latency = round(time.perf_counter() - start, 2)

            answer = result.get("answer", "")
            sources = result.get("sources", [])

            source_pages = [
                source.get("page")
                for source in sources
                if source.get("page") is not None
            ]

            record = {
                "question": question,
                "answer": answer,
                "latency_seconds": latency,
                "source_count": len(sources),
                "source_pages": source_pages,
                "answer_generated": bool(answer.strip()),
                "sources_returned": bool(sources),
                "expected_page": item["expected_page"],
            }

        except Exception as exc:
            record = {
                "question": question,
                "error": str(exc),
                "answer_generated": False,
                "sources_returned": False,
            }

        results.append(record)

        print("\n" + "=" * 60)
        print(f"Question: {question}")
        print(f"Answer: {record.get('answer', record.get('error', ''))}")
        print(f"Latency: {record.get('latency_seconds', 'N/A')} seconds")
        print(f"Sources returned: {record['sources_returned']}")
        print(f"Source pages: {record.get('source_pages', [])}")

    output_path = Path("evaluation_results.json")
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    successful = [
        result for result in results
        if "latency_seconds" in result
    ]

    if successful:
        average_latency = sum(
            result["latency_seconds"] for result in successful
        ) / len(successful)

        print("\n" + "=" * 60)
        print(f"Questions evaluated: {len(results)}")
        print(f"Average latency: {average_latency:.2f} seconds")
        print(
            "Answers generated: "
            f"{sum(r['answer_generated'] for r in successful)}"
            f"/{len(successful)}"
        )
        print(
            "Questions with sources: "
            f"{sum(r['sources_returned'] for r in successful)}"
            f"/{len(successful)}"
        )

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
