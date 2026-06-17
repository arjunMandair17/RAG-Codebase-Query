import asyncio

from fastapi import APIRouter, HTTPException
from models.ingest import IngestRequest

from chunk import chunk_code, parse_code
from embedding import clear_collection, embed_chunks

router = APIRouter(
    prefix="/ingest",
    tags=["ingest"]
)


def _chunk_and_embed(files: list[dict]) -> int:
    """Chunk files and store embeddings; returns number of chunks stored."""
    chunks = []
    for file in files:
        chunks.extend(
            chunk_code(file["content"], file["language"], file["path"], file["extension"])
        )
    if not embed_chunks(chunks):
        raise RuntimeError("Error embedding chunks")
    return len(chunks)


@router.post("/")
async def ingest(body: IngestRequest):
    """Fetch a GitHub repo, chunk it, and store embeddings in Chroma."""
    github_url = body.github_url

    if not github_url or github_url == "" or not github_url.startswith("https://github.com/"):
        raise HTTPException(status_code=400, detail="GitHub URL is invalid")

    try:
        files = await parse_code(github_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch repo: {e}")

    try:
        chunks_stored = await asyncio.to_thread(_chunk_and_embed, files)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": "Chunks embedded successfully", "chunks_stored": chunks_stored}


@router.delete("/")
async def delete():
    """Clear all embedded chunks from the collection."""
    cleared = await asyncio.to_thread(clear_collection)
    if not cleared:
        raise HTTPException(status_code=500, detail="Error clearing collection")
    return {"message": "Collection cleared successfully"}
