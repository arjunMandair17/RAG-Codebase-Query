from fastapi import FastAPI
from fastapi.routing import APIRouter

from routes.health import router as health_router
from routes.ingest import router as ingest_router
from routes.retrieve import router as retrieve_router

app = FastAPI(
    title="RAG Codebase Query",
    description="A system that passes codebase queries to an LLM using RAG and returns an answer",
    version="0.1.0"
)

app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(retrieve_router)