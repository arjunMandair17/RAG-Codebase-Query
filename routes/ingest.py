from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from models.ingest import IngestRequest

from services import ingest_job
from services.embedding import clear_collection
import asyncio

router = APIRouter(
    prefix="/ingest",
    tags=["ingest"]
)


@router.post("/")
async def ingest(body: IngestRequest, sync: bool = False):
    """Start ingesting a GitHub repo (background by default) or block until finished."""
    github_url = body.github_url

    if not github_url or github_url == "" or not github_url.startswith("https://github.com/"):
        raise HTTPException(status_code=400, detail="GitHub URL is invalid")

    if ingest_job.is_running():
        raise HTTPException(status_code=409, detail="Ingest already in progress")

    if sync:
        try:
            chunks_stored = await ingest_job.run_ingest_blocking(github_url)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {"message": "Chunks embedded successfully", "chunks_stored": chunks_stored}

    ingest_job.start(github_url)
    return JSONResponse(status_code=202, content={"status": "started"})


@router.get("/status")
async def ingest_status():
    """Return progress for the current or most recent background ingest job."""
    return ingest_job.get_status()


@router.delete("/")
async def delete():
    """Clear all embedded chunks from the collection."""
    if ingest_job.is_running():
        raise HTTPException(status_code=409, detail="Cannot clear collection while ingest is running")

    cleared = await asyncio.to_thread(clear_collection)
    if not cleared:
        raise HTTPException(status_code=500, detail="Error clearing collection")
    return {"message": "Collection cleared successfully"}
