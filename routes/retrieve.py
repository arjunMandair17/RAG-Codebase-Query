from fastapi import APIRouter, HTTPException
from models.retrieve import RetrieveRequest
from llm import generate_response

router = APIRouter(
    prefix="/retrieve",
    tags=["retrieve"]
)

@router.post("/")
async def retrieve(body: RetrieveRequest):
    query = body.query

    if not query or query == "" or len(query) < 3:
        raise HTTPException(status_code=400, detail="Query is invalid")

    response = generate_response(query)
    if not response or response == "" or len(response) < 3:
        raise HTTPException(status_code=500, detail="I'm sorry, I'm having trouble answering your query. Please try again.")

    return {"answer": response}