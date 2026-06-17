import hashlib
import os
from config import collection


def _chunk_id(chunk: dict) -> str:
    """Build a stable id for upserting chunks without duplicates on re-ingest."""
    key = f"{chunk['path']}|{chunk['text']}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def embed_chunks(chunks: list[dict]) -> bool:
    """Embed and store chunks in Chroma."""
    if not chunks:
        return True

    failed = False
    batch_size = int(os.getenv("BATCH_SIZE", "50"))

    for i in range(0, len(chunks), batch_size):
        curBatch = chunks[i:i + batch_size]
        try:
            collection.upsert(
                ids=[_chunk_id(chunk) for chunk in curBatch],
                documents=[chunk["text"] for chunk in curBatch],
                metadatas=[
                    {
                        "path": chunk["path"],
                        "language": chunk.get("language") or "unknown",
                        "type": chunk["type"],
                    }
                    for chunk in curBatch
                ],
            )
        except Exception as e:
            print(f"Error embedding batch: {e}")
            failed = True
    return not failed


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
        n_results=10,
        include=["distances", "documents", "metadatas"],
    )
    return results["documents"][0] if results["documents"] else []
