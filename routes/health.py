from fastapi import APIRouter, HTTPException
from config import collection

router = APIRouter(
    prefix="/health",
    tags=["health"]
)

@router.get("/")
async def health():
    """Check the health of the application."""
    if not collection.client.is_running():
        raise HTTPException(status_code=500, detail="Chroma client is not running")
    return {"embedded_chunks": collection.count(), "chroma_client": collection.client.is_running()}