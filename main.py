from chunk import parse_code, chunk_code
from embedding import clear_collection
import requests

github_url = "https://github.com/arjunMandair17/RAG-Codebase-Query"

clear_collection()

response = requests.delete("http://localhost:8000/ingest")
print(response.json())

response = requests.post("http://localhost:8000/ingest", json={"github_url": github_url})
print(response.json())

while True:
    query = input("Enter a query: ")
    response = requests.post("http://localhost:8000/retrieve", json={"query": query})
    print(response.json()["answer"])
clear_collection()