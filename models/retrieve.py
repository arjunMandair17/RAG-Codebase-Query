from pydantic import BaseModel

class RetrieveRequest(BaseModel):
    query: str