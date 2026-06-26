import os

import google.generativeai as genai
from dotenv import load_dotenv

from services.embedding import query_chunks

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_EMBEDDING_KEY"))


def generate_response(query: str) -> str:
    """Retrieve relevant chunks and generate an answer with Gemini."""
    chunks = query_chunks(query)
    context = "\n\n".join(chunks) if chunks else "No relevant context found."

    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
You are a helpful code assistant who is an expert in the codebase a user provides to you.
Answer based on the retrieved source files below. Each block starts with "File:" showing where the code lives.
Synthesize a clear answer that explains how the code works, especially for workflows that span multiple files.

The user's query is:
{query}

Retrieved source from the codebase:
{context}

Answer using the context above. Mention relevant file paths when helpful.
If the context does not contain enough information, say so.
"""

    try:
        response = model.generate_content(prompt)
        if not response.candidates:
            return "I'm sorry, I'm having trouble answering your query. Please try again."
        return response.text
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I'm sorry, I'm having trouble answering your query. Please try again."
