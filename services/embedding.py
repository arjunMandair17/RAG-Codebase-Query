import hashlib
import os
import time

from config import collection


def _chunk_id(chunk: dict) -> str:
    """Build a stable id for upserting chunks without duplicates on re-ingest."""
    key = "|".join(
        [
            chunk["path"],
            chunk.get("type", ""),
            str(chunk.get("key", "")),
            str(chunk.get("second_key", "")),
            chunk["text"],
        ]
    )
    return hashlib.sha256(key.encode()).hexdigest()


def _dedupe_chunks(chunks: list[dict], stats: dict | None = None) -> list[dict]:
    """Drop chunks that share the same id (identical content); Chroma upsert requires unique ids per batch."""
    seen: set[str] = set()
    unique: list[dict] = []
    for chunk in chunks:
        cid = _chunk_id(chunk)
        if cid in seen:
            if stats is not None:
                stats["duplicates_skipped"] = stats.get("duplicates_skipped", 0) + 1
            continue
        seen.add(cid)
        unique.append(chunk)
    return unique


def embed_chunks(chunks: list[dict], stats: dict | None = None) -> bool:
    """Embed and store chunks in Chroma."""
    if not chunks:
        return True

    chunks = _dedupe_chunks(chunks, stats=stats)
    if not chunks:
        return True

    failed = False
    batch_size = int(os.getenv("BATCH_SIZE", "30"))
    batch_delay = float(os.getenv("EMBED_BATCH_DELAY", "1"))

    for i in range(0, len(chunks), batch_size):
        curBatch = chunks[i:i + batch_size]
        batch_ok = False
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
                batch_ok = True
                if stats is not None:
                    stats["batches_ok"] = stats.get("batches_ok", 0) + 1
                    stats["chunks_embedded"] = stats.get("chunks_embedded", 0) + len(curBatch)
                break
            except Exception as e:
                err = str(e).lower()
                if attempt < 4 and any(x in err for x in ("429", "rate", "quota", "too many")):
                    if stats is not None:
                        stats["rate_limit_retries"] = stats.get("rate_limit_retries", 0) + 1
                    time.sleep(batch_delay * (2 ** attempt))
                    continue
                print(f"Error embedding batch: {e}")
                if stats is not None:
                    stats["other_errors"] = stats.get("other_errors", 0) + 1
                failed = True
                break
        if not batch_ok and not failed:
            failed = True
        if failed:
            if stats is not None:
                stats["batches_failed"] = stats.get("batches_failed", 0) + 1
            break
        if i + batch_size < len(chunks):
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
