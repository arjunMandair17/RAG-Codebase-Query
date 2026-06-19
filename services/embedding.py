import hashlib
import os
import time

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
    batch_size = int(os.getenv("BATCH_SIZE", "30"))
    batch_delay = float(os.getenv("EMBED_BATCH_DELAY", "1"))

    for i in range(0, len(chunks), batch_size):
        curBatch = chunks[i:i + batch_size]
        for attempt in range(5):
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
                break
            except Exception as e:
                err = str(e).lower()
                if attempt < 4 and any(x in err for x in ("429", "rate", "quota", "too many")):
                    time.sleep(batch_delay * (2 ** attempt))
                    continue
                print(f"Error embedding batch: {e}")
                failed = True
                break
        if not failed and i + batch_size < len(chunks):
            time.sleep(batch_delay)

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
