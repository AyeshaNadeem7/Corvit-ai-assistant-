import os
import pickle
import faiss
import streamlit as st

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# GET GROQ API KEY
#
# Priority:
# 1. Streamlit Cloud Secrets
# 2. Environment variable
# 3. Local .env
# ============================================================

def get_groq_api_key():

    # --------------------------------------------------------
    # 1. STREAMLIT SECRETS
    # --------------------------------------------------------

    try:
        if "GROQ_API_KEY" in st.secrets:

            secret_key = st.secrets["GROQ_API_KEY"]

            if secret_key:

                secret_key = str(secret_key).strip()

                if secret_key:
                    return secret_key

    except Exception:
        pass


    # --------------------------------------------------------
    # 2. ENVIRONMENT VARIABLE
    # --------------------------------------------------------

    env_key = os.environ.get("GROQ_API_KEY")

    if env_key:

        env_key = env_key.strip()

        if env_key:
            return env_key


    # --------------------------------------------------------
    # 3. LOCAL .ENV
    # --------------------------------------------------------

    load_dotenv()

    dotenv_key = os.getenv("GROQ_API_KEY")

    if dotenv_key:

        dotenv_key = dotenv_key.strip()

        if dotenv_key:
            return dotenv_key


    # --------------------------------------------------------
    # API KEY NOT FOUND
    # --------------------------------------------------------

    raise ValueError(
        """
GROQ_API_KEY was not found.

LOCAL:
Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key

STREAMLIT CLOUD:
Go to:

Manage app -> Settings -> Secrets

and add:

GROQ_API_KEY = "your_groq_api_key"
"""
    )


GROQ_API_KEY = get_groq_api_key()


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_PATH = "vectorstore/corvit.index"

CHUNKS_PATH = "vectorstore/chunks.pkl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_MODEL = "openai/gpt-oss-20b"

TOP_K = 5


# ============================================================
# CHECK VECTOR STORE
# ============================================================

if not os.path.isfile(INDEX_PATH):

    raise FileNotFoundError(
        f"""
FAISS index not found.

Expected:
{INDEX_PATH}

Make sure this file exists in your GitHub repository.
"""
    )


if not os.path.isfile(CHUNKS_PATH):

    raise FileNotFoundError(
        f"""
Chunks file not found.

Expected:
{CHUNKS_PATH}

Make sure this file exists in your GitHub repository.
"""
    )


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


embedding_model = load_embedding_model()


# ============================================================
# LOAD FAISS INDEX
# ============================================================

@st.cache_resource
def load_faiss_index():

    return faiss.read_index(
        INDEX_PATH
    )


index = load_faiss_index()


# ============================================================
# LOAD CHUNKS
# ============================================================

@st.cache_resource
def load_chunks():

    with open(
        CHUNKS_PATH,
        "rb"
    ) as f:

        return pickle.load(f)


chunks = load_chunks()


# ============================================================
# GROQ CLIENT
# ============================================================

@st.cache_resource
def load_groq_client():

    return Groq(
        api_key=GROQ_API_KEY
    )


client = load_groq_client()


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    query,
    top_k=TOP_K
):

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


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for i, result in enumerate(
        results,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {i}

Page: {result['page']}

{result['text']}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    conversation_history=None
):

    results = retrieve_documents(
        question,
        TOP_K
    )

    if not results:

        return (
            "I could not find relevant information "
            "in the Corvit knowledge base."
        ), results

    context = build_context(
        results
    )


    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

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

8. If the user asks about multiple things,
   answer each part separately.

9. If a source page number is available,
   mention it at the end as a source.

10. Never claim that you accessed a website unless
    that information is actually present in the context.

The knowledge base is the source of truth
for this chatbot.
"""


    # ========================================================
    # USER PROMPT
    # ========================================================

    user_prompt = f"""
KNOWLEDGE BASE CONTEXT:

{context}

USER QUESTION:

{question}

Answer the user's question using the knowledge base.
"""


    # ========================================================
    # MESSAGES
    # ========================================================

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]


    # ========================================================
    # CONVERSATION HISTORY
    # ========================================================

    if conversation_history:

        for message in conversation_history[-6:]:

            if message.get("role") in [
                "user",
                "assistant"
            ]:

                messages.append(
                    {
                        "role": message["role"],
                        "content": message["content"]
                    }
                )


    # ========================================================
    # CURRENT QUESTION
    # ========================================================

    messages.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )


    # ========================================================
    # GROQ API CALL
    # ========================================================

    response = client.chat.completions.create(

        model=GROQ_MODEL,

        messages=messages,

        temperature=0.2,

        max_completion_tokens=700
    )


    # ========================================================
    # ANSWER
    # ========================================================

    answer = response.choices[0].message.content


    return answer, results

