
from dotenv import load_dotenv
from app.rag import BioRAG

load_dotenv()


def main():
    rag = BioRAG()

    question = input(
        "Ask a question about your research paper: "
    )

    result = rag.ask(question)

    print("\n========== ANSWER ==========\n")
    print(result["answer"])

    print("\n========== SOURCES ==========\n")

    for source in result["sources"]:
        print(
            f"{source['id']} | {source['source']} "
            f"| Page {source['page']}"
        )
        print(source["text"][:300])
        print("-" * 60)


if __name__ == "__main__":
    main()
