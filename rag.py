import os
import pickle
import faiss
import numpy as np

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq


# ==========================================
# LOAD ENVIRONMENT
# ==========================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. "
        "Please add it to your .env file."
    )


# ==========================================
# CONFIGURATION
# ==========================================

INDEX_PATH = "vectorstore/corvit.index"
CHUNKS_PATH = "vectorstore/chunks.pkl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_MODEL = "openai/gpt-oss-20b"

TOP_K = 5


# ==========================================
# LOAD COMPONENTS
# ==========================================

if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(
        "FAISS index not found.\n"
        "Run: python ingest.py"
    )

if not os.path.exists(CHUNKS_PATH):
    raise FileNotFoundError(
        "chunks.pkl not found.\n"
        "Run: python ingest.py"
    )


embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

index = faiss.read_index(INDEX_PATH)

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

client = Groq(
    api_key=GROQ_API_KEY
)


# ==========================================
# RETRIEVE DOCUMENTS
# ==========================================

def retrieve_documents(query, top_k=TOP_K):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        if idx >= len(chunks):
            continue

        result = {
            "text": chunks[idx]["text"],
            "page": chunks[idx]["page"],
            "score": float(score)
        }

        results.append(result)

    return results


# ==========================================
# BUILD CONTEXT
# ==========================================

def build_context(results):

    context_parts = []

    for i, result in enumerate(results, start=1):

        context_parts.append(
            f"""
SOURCE {i}
Page: {result['page']}

{result['text']}
"""
        )

    return "\n".join(context_parts)


# ==========================================
# GENERATE ANSWER
# ==========================================

def generate_answer(question, conversation_history=None):

    results = retrieve_documents(
        question,
        TOP_K
    )

    if not results:
        return (
            "I could not find relevant information "
            "in the Corvit knowledge base."
        ), results

    context = build_context(results)

    system_prompt = """
You are Corvit Assistant, a helpful RAG chatbot
for Corvit Systems.

Your job is to answer questions using ONLY the
provided knowledge-base context.

IMPORTANT RULES:

1. Do not invent facts.
2. Do not use information that is not supported
   by the retrieved context.
3. If the answer is not available in the context,
   clearly say that the information is not available
   in the current knowledge base.
4. Do not guess course fees, dates, timings,
   phone numbers, eligibility requirements,
   discounts, or course availability.
5. If the knowledge base says information may change,
   mention that the user should verify the latest
   information with Corvit.
6. Answer naturally and conversationally.
7. Keep answers concise but useful.
8. If the user asks about multiple things, answer
   each part separately.
9. If a source page number is available, mention
   it at the end as a source.
10. Never claim that you accessed a website unless
    that information is actually present in the context.

The knowledge base is the source of truth for this chatbot.
"""

    user_prompt = f"""
KNOWLEDGE BASE CONTEXT:

{context}

USER QUESTION:

{question}

Answer the user's question using the knowledge base.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # Add limited conversation history
    if conversation_history:

        for message in conversation_history[-6:]:

            if message["role"] in [
                "user",
                "assistant"
            ]:

                messages.append({
                    "role": message["role"],
                    "content": message["content"]
                })

    messages.append({
        "role": "user",
        "content": user_prompt
    })

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        max_completion_tokens=700
    )

    answer = response.choices[0].message.content

    return answer, results