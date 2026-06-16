import chromadb
import os
from chromadb.utils import embedding_functions
from dotenv import load_dotenv


load_dotenv()


gemini_embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    model_name="models/gemini-embedding-001",
    api_key=os.getenv("GEMINI_EMBEDDING_KEY")
)

chroma_client = chromadb.PersistentClient(path="db")
collection = chroma_client.get_or_create_collection(
    name="codebase", 
    embedding_function=gemini_embedding_function
)

def embed_chunks(chunks: list[dict]) -> list[dict]:
    return [
        collection.add(
            documents=[chunk["text"]],
            metadatas=[
                {
                    "path": chunk["path"], 
                    "language": chunk["language"], 
                    "type": chunk["type"], 
                    "extension": chunk["extension"]
                }
            ]
        )
        for chunk in chunks
    ]


def create_embedding(query: str) -> list[float]:
    results = collection.query(
        query_texts=[query],
        n_results=8,
        include=["distances", "documents", "metadatas"]
    )
    return results