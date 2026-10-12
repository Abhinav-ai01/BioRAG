
def create_chunks(pages, chunk_size=350, overlap=50):
    """
    Split extracted PDF text into overlapping chunks.
    Preserve source filename and page number for citations.
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk_size or overlap")

    chunks = []
    step = chunk_size - overlap

    for page in pages:
        words = page["text"].split()

        for start in range(0, len(words), step):
            text = " ".join(words[start:start + chunk_size])

            if text.strip():
                chunks.append({
                    "chunk_id": len(chunks),
                    "text": text,
                    "source": page["source"],
                    "page": page["page"],
                })

            if start + chunk_size >= len(words):
                break

    return chunks
