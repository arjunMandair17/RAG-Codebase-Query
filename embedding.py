from config import collection


def embed_chunks(chunks: list[dict]) -> bool:
    """Embed and store chunks in Chroma."""
    for chunk in chunks:
        try:
            collection.add(
                documents=[chunk["text"]],
                metadatas=[
                    {
                        "path": chunk["path"],
                        "language": chunk["language"],
                        "type": chunk["type"],
                    }
                ],
            )
        except Exception as e:
            print(f"Error embedding chunk: {e}")
            continue
    if collection.count() != len(chunks):
        raise Exception("Error embedding chunks")
    return True


def query_chunks(query: str) -> list[dict]:
    """Query Chroma for chunks similar to the given text."""
    results = collection.query(
        query_texts=[query],
        n_results=8,
        include=["distances", "documents", "metadatas"],
    )
    return results["documents"]
