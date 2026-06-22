import asyncio

from services.chunk import chunk_code, parse_code
from services.embedding import embed_chunks

_status: dict = {
    "state": "idle",
    "phase": None,
    "chunks_total": 0,
    "chunks_embedded": 0,
    "chunks_stored": 0,
    "error": None,
}
_task: asyncio.Task | None = None


def get_status() -> dict:
    """Return a snapshot of the current background ingest job."""
    return {**_status}


def is_running() -> bool:
    """Return True when a background ingest is in progress."""
    return _status["state"] == "running"


def start(github_url: str) -> None:
    """Kick off ingest in the background; poll get_status() for progress."""
    global _task
    _status.update(
        state="running",
        phase="fetching",
        chunks_total=0,
        chunks_embedded=0,
        chunks_stored=0,
        error=None,
    )
    _task = asyncio.create_task(_run_ingest(github_url))


async def run_ingest_blocking(github_url: str) -> int:
    """Run ingest to completion and return the number of chunks stored."""
    await _run_ingest(github_url)
    if _status["state"] == "failed":
        raise RuntimeError(_status["error"] or "Error embedding chunks")
    return _status["chunks_stored"]


async def _run_ingest(github_url: str) -> None:
    """Fetch, chunk, and embed a repo while updating shared job status."""
    try:
        _status["phase"] = "fetching"
        files = await parse_code(github_url)

        _status["phase"] = "chunking"
        chunks = []
        for file in files:
            chunks.extend(
                chunk_code(file["content"], file["language"], file["path"], file["extension"])
            )

        _status["phase"] = "embedding"
        _status["chunks_total"] = len(chunks)

        ok = await asyncio.to_thread(embed_chunks, chunks, _status)
        if not ok:
            raise RuntimeError("Error embedding chunks")

        _status["state"] = "complete"
        _status["chunks_stored"] = _status.get("chunks_embedded", len(chunks))
        _status["phase"] = None
    except Exception as e:
        _status["state"] = "failed"
        _status["error"] = str(e)
        _status["phase"] = None
