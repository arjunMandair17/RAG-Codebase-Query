import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent

# Local persistent storage (default). Mount this path as a volume in Docker.
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(PROJECT_ROOT / "db"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "codebase")

# Remote Chroma (set CHROMA_HOST when running backend in Docker against a Chroma container)
CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

GEMINI_EMBEDDING_KEY = os.getenv("GEMINI_EMBEDDING_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")


def get_embedding_function():
    """Build the Gemini embedding function used by the Chroma collection."""
    model_name = EMBEDDING_MODEL.removeprefix("models/")
    return embedding_functions.GoogleGeminiEmbeddingFunction(
        model_name=model_name,
        api_key_env_var="GEMINI_EMBEDDING_KEY",
        task_type="RETRIEVAL_DOCUMENT"
    )


def get_chroma_client():
    """Return a Chroma client: HTTP when CHROMA_HOST is set, else local persistent storage."""
    if CHROMA_HOST:
        return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


embedding_function = get_embedding_function()
chroma_client = get_chroma_client()
collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME,
    embedding_function=embedding_function,
)
