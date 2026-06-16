import os

import google.generativeai as genai
from dotenv import load_dotenv

from embedding import query_chunks

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_EMBEDDING_KEY"))


def generate_response(query: str) -> str:
    """Retrieve relevant chunks and generate an answer with Gemini."""
    chunks = query_chunks(query)
    context = "\n\n".join(chunks) if chunks else "No relevant context found."

    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
You are a helpful code assistant who is an expert in the codebase a user provides to you.
You will be given a query from a user and you will need to answer it based on your knowledge of the codebase
provided to you.

The user's query is:
{query}

Here is some relevant context from the codebase that relate to the user's query:
{context}

Please answer the user's query based on the context provided and your own reasoning.
If you don't know the answer, please say so.
If the user's query is not related to the codebase, please say so.
If the user's query is not clear, please ask for more information.
Please provide the answer in a concise and to the point manner.
"""

    try:
        response = model.generate_content(prompt)
        if not response.candidates:
            return "I'm sorry, I'm having trouble answering your query. Please try again."
        return response.text
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I'm sorry, I'm having trouble answering your query. Please try again."
