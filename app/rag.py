import ollama

from app.parser import extract_title_pages
from app.reranker import PassageReranker


class BioRAG:
    def __init__(self):
        self.retriever = PassageReranker()
        self.model = "qwen2.5:3b"

    def answer_metadata_question(self, question: str):
        """Answer paper metadata questions using the first page of each PDF."""
        pages = extract_title_pages()

        if not pages:
            return None

        context = "\n\n".join(
            f"[S{i} | {page['source']} | page 1]\n{page['text']}"
            for i, page in enumerate(pages, start=1)
        )

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are BioRAG, a biomedical research assistant. "
                        "Answer metadata questions using only the supplied "
                        "first-page text. For author questions, identify "
                        "the authors as printed on the title page. Do not "
                        "confuse author names with author contributions. "
                        "Cite sources using IDs such as [S1]. If the "
                        "information is absent, say so. Never invent facts."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Title-page evidence:\n{context}"
                    ),
                },
            ],
        )

        return {
            "answer": response["message"]["content"],
            "sources": [
                {
                    "id": f"S{i}",
                    "source": page["source"],
                    "page": 1,
                    "text": page["text"],
                }
                for i, page in enumerate(pages, start=1)
            ],
        }

    def ask(self, question: str) -> dict:
        if not question or not question.strip():
            raise ValueError("Please enter a question.")

        question = question.strip()

        # Route metadata questions to the paper's first page.
        metadata_keywords = (
            "author",
            "authors",
            "written by",
            "paper title",
            "article title",
            "doi",
        )

        if any(keyword in question.lower() for keyword in metadata_keywords):
            result = self.answer_metadata_question(question)
            if result is not None:
                return result

        # Retrieve and rerank evidence for other questions.
        passages = self.retriever.search(
            question,
            top_k=5,
            candidate_k=20,
        )

        if not passages:
            return {
                "answer": "No supporting evidence was found.",
                "sources": [],
            }

        context = "\n\n".join(
            f"[S{i} | {p['source']} | page {p['page']}]\n{p['text']}"
            for i, p in enumerate(passages, start=1)
        )

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are BioRAG, a biomedical research assistant. "
                        "Answer using only the supplied passages. "
                        "Cite factual claims with IDs such as [S1]. "
                        "If the evidence is insufficient, say so. "
                        "Never invent research findings or citations. "
                        "Do not provide a clinical diagnosis."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Retrieved evidence:\n{context}"
                    ),
                },
            ],
        )

        return {
            "answer": response["message"]["content"],
            "sources": [
                {
                    "id": f"S{i}",
                    "source": p["source"],
                    "page": p["page"],
                    "text": p["text"],
                    "rerank_score": p.get("rerank_score"),
                }
                for i, p in enumerate(passages, start=1)
            ],
        }


if __name__ == "__main__":
    rag = BioRAG()
    question = input("Ask a question about your research paper: ")
    result = rag.ask(question)

    print("\n--- ANSWER ---\n")
    print(result["answer"])

    print("\n--- SOURCES ---\n")

    for source in result["sources"]:
        print(
            f"{source['id']}: "
            f"{source['source']}, page {source['page']}"
        )
        print(source["text"][:300])
        print("-" * 50)
