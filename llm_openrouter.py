"""
OpenRouter-based LLM wrapper for RAG comparison.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment 
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

MODEL_NAME = "mistralai/mistral-7b-instruct"


def generate_openrouter_answer(prompt: str) -> str:
    """
    Generates an answer using OpenRouter API.
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()
