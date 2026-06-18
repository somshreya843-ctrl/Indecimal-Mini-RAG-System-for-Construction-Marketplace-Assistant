"""
rag.py
------
Features:
- FAISS semantic retrieval with distance + confidence
- Cached local Hugging Face LLM (Phi-2)
- Optional OpenRouter LLM (Mistral-7B)
- Strict grounding (no hallucinations)
- Clean answer output (no prompt echo)
- Latency tracking
"""

from typing import Dict
from dotenv import load_dotenv
import os
import re
import time
import streamlit as st

from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from vector import get_retriever


# Cached Local Phi-2
# @st.cache_resource(show_spinner="Loading local Phi-2 model...")
# def load_local_llm():
#     tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

#     model = AutoModelForCausalLM.from_pretrained(
#         HF_MODEL_NAME,
#         torch_dtype="auto",
#         trust_remote_code=True
#     )

#     hf_pipeline = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         device=0 if DEVICE == "cuda" else -1,
#         max_new_tokens=256,
#         do_sample=False,
#         use_cache=True
#     )

#     llm = HuggingFacePipeline(pipeline=hf_pipeline)
#     # Warm up
#     llm.invoke("Warm up")

#     return llm
# Environment
load_dotenv()

HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "./models/phi-2")
DEVICE = "cpu"

@st.cache_resource(show_spinner="Loading local Phi-2 model...")
def load_local_llm():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_NAME,
        torch_dtype="auto",
        trust_remote_code=True
    )

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if DEVICE == "cuda" else -1,
        max_new_tokens=256,
        do_sample=False,
        use_cache=True
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    llm.invoke("Warm up")

    return llm


# llm = load_local_llm()
llm = None

#Retriever
retriever = get_retriever(k=3)


#  Utilities
def compute_confidence(distance: float) -> float:
    """Lower FAISS distance → higher confidence."""
    return round(1 / (1 + distance), 2)


def clean_answer(text: str) -> str:
    """Remove prompt echo and return only final answer."""
    if "Answer:" in text:
        text = text.split("Answer:")[-1]

    text = re.sub(
        r"(System:|Human:|Context:|Question:).*",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    return text.strip()


def is_context_echo(answer: str, context: str) -> bool:
    """Detect when model is copying context instead of answering."""
    return len(answer) > 0 and answer[:200] in context

#  RAG Inference
def generate_answer(query: str, model_type: str = "local") -> Dict:
    """
    Returns:
    - retrieved_context (with confidence)
    - grounded answer
    - latency
    """

    from llm_openrouter import generate_openrouter_answer

    global llm

    # Auto-fallback if local model is unavailable
    if model_type == "local":
        if llm is None:
            if os.path.exists(HF_MODEL_NAME):
                llm = load_local_llm()
            else:
                model_type = "openrouter"

    #Retrieve documents WITH distance
    docs_and_scores = retriever.vectorstore.similarity_search_with_score(
        query, k=3
    )

    retrieved_context = []
    context_chunks = []

    for doc, distance in docs_and_scores:
        confidence = compute_confidence(distance)
        retrieved_context.append({
            "source": doc.metadata.get("source"),
            "content": doc.page_content,
            "distance": round(distance, 3),
            "confidence": confidence
        })
        context_chunks.append(doc.page_content)

    # Rank by confidence
    retrieved_context.sort(key=lambda x: x["confidence"], reverse=True)

    context_text = "\n\n".join(context_chunks)
    # LOCAL Phi-2 
    if model_type == "local":
        phi2_prompt = f"""
You must answer the question using ONLY the information below.

Context:
{context_text}

Question:
{query}

If the answer is not explicitly stated, reply exactly:
"I do not have enough information in the provided documents."

Answer:
""".strip()

        start = time.time()
        raw = llm.invoke(phi2_prompt)
        latency = time.time() - start

        answer = clean_answer(raw)

        if not answer or is_context_echo(answer, context_text):
            answer = (
                "The provided documents describe Indecimal’s policies and processes. "
                "However, they do not specify the information required to answer the question. "
                "I do not have enough information in the provided documents."
            )
    # OpenRouter (Mistral-7B)
    elif model_type == "openrouter":
        prompt_text = f"""
You are an AI assistant for a construction marketplace.

Answer ONLY using the context below.
If not present, say:
"I do not have enough information in the provided documents."

Context:
{context_text}

Question:
{query}

Answer:
""".strip()

        start = time.time()
        answer = generate_openrouter_answer(prompt_text)
        latency = time.time() - start

    else:
        raise ValueError("model_type must be 'local' or 'openrouter'")

    return {
        "query": query,
        "retrieved_context": retrieved_context,
        "answer": answer,
        "latency": round(latency, 2)
    }
