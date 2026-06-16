import hashlib

from config import collection


def _chunk_id(chunk: dict) -> str:
    """Build a stable id for upserting chunks without duplicates on re-ingest."""
    key = f"{chunk['path']}|{chunk['text']}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def embed_chunks(chunks: list[dict]) -> bool:
    """Embed and store chunks in Chroma."""
    if not chunks:
        return True

    failed = 0
    for chunk in chunks:
        try:
            collection.upsert(
                ids=[_chunk_id(chunk)],
                documents=[chunk["text"]],
                metadatas=[
                    {
                        "path": chunk["path"],
                        "language": chunk.get("language") or "unknown",
                        "type": chunk["type"],
                    }
                ],
            )
        except Exception as e:
            print(f"Error embedding chunk: {e}")
            failed += 1
    return failed == 0


def clear_collection() -> bool:
    """Remove all documents from the collection."""
    try:
        ids = collection.get(include=[])["ids"]
        if ids:
            collection.delete(ids=ids)
        return True
    except Exception as e:
        print(f"Error clearing collection: {e}")
        return False


def query_chunks(query: str) -> list[str]:
    """Query Chroma for chunks similar to the given text."""
    results = collection.query(
        query_texts=[query],
        n_results=8,
        include=["distances", "documents", "metadatas"],
    )
    return results["documents"][0] if results["documents"] else []
