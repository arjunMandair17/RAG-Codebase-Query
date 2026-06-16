from fastapi import APIRouter, HTTPException
from models.ingest import IngestRequest

from chunk import parse_code, chunk_code
from embedding import clear_collection, embed_chunks

router = APIRouter(
    prefix="/ingest",
    tags=["ingest"]
)


@router.post("/")
async def ingest(body: IngestRequest):
    """Fetch a GitHub repo, chunk it, and store embeddings in Chroma."""
    github_url = body.github_url

    if not github_url or github_url == "" or not github_url.startswith("https://github.com/"):
        raise HTTPException(status_code=400, detail="GitHub URL is invalid")

    try:
        files = parse_code(github_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch repo: {e}")

    chunks = []
    for file in files:
        chunks.extend(chunk_code(file["content"], file["language"], file["path"], file["extension"]))

    if not embed_chunks(chunks):
        raise HTTPException(status_code=500, detail="Error embedding chunks")

    return {"message": "Chunks embedded successfully", "chunks_stored": len(chunks)}


@router.delete("/")
async def delete():
    """Clear all embedded chunks from the collection."""
    if not clear_collection():
        raise HTTPException(status_code=500, detail="Error clearing collection")
    return {"message": "Collection cleared successfully"}
