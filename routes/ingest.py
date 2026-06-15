from fastapi import APIRouter, HTTPException
from models.ingest import IngestRequest

from chunk import parse_code, chunk_code

router = APIRouter(
    prefix="/ingest",
    tags=["ingest"]
)


@router.post("/")
async def ingest(body: IngestRequest):

    github_url = body.github_url

    if not github_url or github_url == "" or not github_url.startswith("https://github.com/"):
        raise HTTPException(status_code=400, detail="GitHub URL is invalid")


    ## parse the codebase
    files = parse_code(github_url)
    
    ## chunk the codebase
    chunks = []
    for file in files:
        chunks.extend(chunk_code(file["content"], file["language"], file["path"], file["extension"]))

    ## embed the chunks
    embeddings = embed_chunks(chunks)
