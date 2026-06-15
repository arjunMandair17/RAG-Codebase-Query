from chunk import parse_code, chunk_code

github_url = "https://github.com/arjunMandair17/RAG-Codebase-Query"

files = parse_code(github_url)

for file in files:
    chunks = chunk_code(file["content"], file["language"], file["path"])
    print(chunks)