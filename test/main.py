from services.chunk import parse_code, chunk_code
from services.embedding import clear_collection
import requests

github_url = "https://github.com/arjunMandair17/Job-Application-Tracker"



confirm = input("Clear collection? (y/n): ")
if confirm == "y":
    clear_collection()

confirm = input("Ingest codebase? (y/n): ")
if confirm == "n":
    exit()

response = requests.delete("http://localhost:8000/ingest")
print(response.json())

response = requests.post("http://localhost:8000/ingest", json={"github_url": github_url})
print(response.json())

while True:
    query = input("Enter a query: ")
    response = requests.post("http://localhost:8000/retrieve", json={"query": query})
    print(response.json()["answer"])