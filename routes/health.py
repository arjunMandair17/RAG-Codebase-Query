from fastapi import APIRouter, HTTPException

from config import collection

router = APIRouter(
    prefix="/health",
    tags=["health"]
)


@router.get("/")
async def health():
    """Check the health of the application and Chroma connectivity."""
    try:
        return {"status": "ok", "embedded_chunks": collection.count()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chroma unavailable: {e}")
