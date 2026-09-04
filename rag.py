
# rag.py

import os
import pickle
import faiss
import streamlit as st

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# PATHS
# =========================================================

VECTORSTORE_DIR = "vectorstore"

INDEX_PATH = os.path.join(
    VECTORSTORE_DIR,
    "corvit.index"
)

CHUNKS_PATH = os.path.join(
    VECTORSTORE_DIR,
    "chunks.pkl"
)


# =========================================================
# SETTINGS
# =========================================================

TOP_K = 5

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_MODEL = "openai/gpt-oss-20b"


# =========================================================
# GET GROQ API KEY
# =========================================================

def get_groq_api_key():

    # -----------------------------------------------------
    # 1. Streamlit Cloud Secrets
    # -----------------------------------------------------

    try:
        api_key = st.secrets["GROQ_API_KEY"]

        if api_key:
            return str(api_key).strip()

    except Exception:
        pass


    # -----------------------------------------------------
    # 2. Environment Variable
    # -----------------------------------------------------

    api_key = os.environ.get("GROQ_API_KEY")

    if api_key:
        return api_key.strip()


    # -----------------------------------------------------
    # 3. Local .env file
    # -----------------------------------------------------

    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        return api_key.strip()


    # Nothing found
    return None


# =========================================================
# GROQ CLIENT
# =========================================================

def get_groq_client():

    api_key = get_groq_api_key()

    if not api_key:

        st.error(
            "GROQ_API_KEY was not found. "
            "Please add it to Streamlit Secrets or your local .env file."
        )

        return None

    try:

        client = Groq(
            api_key=api_key
        )

        return client

    except Exception as e:

        st.error(
            f"Unable to initialize Groq client: {str(e)}"
        )

        return None


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    try:

        model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        return model

    except Exception as e:

        st.error(
            f"Error loading embedding model: {str(e)}"
        )

        return None


# =========================================================
# LOAD FAISS INDEX
# =========================================================

@st.cache_resource
def load_faiss_index():

    if not os.path.exists(INDEX_PATH):

        st.error(
            f"FAISS index not found: {INDEX_PATH}"
        )

        return None

    try:

        index = faiss.read_index(
            INDEX_PATH
        )

        return index

    except Exception as e:

        st.error(
            f"Error loading FAISS index: {str(e)}"
        )

        return None


# =========================================================
# LOAD CHUNKS
# =========================================================

@st.cache_resource
def load_chunks():

    if not os.path.exists(CHUNKS_PATH):

        st.error(
            f"Chunks file not found: {CHUNKS_PATH}"
        )

        return None

    try:

        with open(
            CHUNKS_PATH,
            "rb"
        ) as file:

            chunks = pickle.load(file)

        return chunks

    except Exception as e:

        st.error(
            f"Error loading chunks.pkl: {str(e)}"
        )

        return None


# =========================================================
# LOAD ALL RAG COMPONENTS
# =========================================================

def load_rag_components():

    model = load_embedding_model()

    index = load_faiss_index()

    chunks = load_chunks()

    return model, index, chunks


# =========================================================
# RETRIEVE RELEVANT DOCUMENTS
# =========================================================

def retrieve_documents(
    question,
    top_k=TOP_K
):

    model, index, chunks = load_rag_components()


    # Check components

    if model is None:
        return []

    if index is None:
        return []

    if chunks is None:
        return []


    try:

        # Create query embedding

        query_embedding = model.encode(
            [question],
            convert_to_numpy=True
        )

        # Make sure FAISS gets float32

        query_embedding = query_embedding.astype(
            "float32"
        )


        # Search FAISS

        distances, indices = index.search(
            query_embedding,
            top_k
        )


        retrieved = []


        for distance, idx in zip(
            distances[0],
            indices[0]
        ):

            if idx < 0:
                continue

            if idx >= len(chunks):
                continue

            retrieved.append(
                {
                    "text": chunks[idx],
                    "score": float(distance)
                }
            )


        return retrieved


    except Exception as e:

        st.error(
            f"Error during document retrieval: {str(e)}"
        )

        return []


# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(documents):

    if not documents:
        return "No relevant information was found."


    context_parts = []


    for i, document in enumerate(
        documents,
        start=1
    ):

        text = document["text"]

        context_parts.append(
            f"Document {i}:\n{text}"
        )


    return "\n\n".join(
        context_parts
    )


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(
    question,
    chat_history=None
):

    # -----------------------------------------------------
    # Validate question
    # -----------------------------------------------------

    if not question or not question.strip():

        return (
            "Please enter a question.",
            []
        )


    question = question.strip()


    # -----------------------------------------------------
    # Retrieve documents
    # -----------------------------------------------------

    documents = retrieve_documents(
        question,
        TOP_K
    )


    # -----------------------------------------------------
    # Build context
    # -----------------------------------------------------

    context = build_context(
        documents
    )


    # -----------------------------------------------------
    # Get Groq client
    # -----------------------------------------------------

    client = get_groq_client()


    if client is None:

        return (
            "The AI service is not configured correctly. "
            "Please check the GROQ_API_KEY configuration.",
            documents
        )


    # -----------------------------------------------------
    # Prepare conversation history
    # -----------------------------------------------------

    history_text = ""


    if chat_history:

        recent_messages = chat_history[-6:]


        for message in recent_messages:

            if not isinstance(
                message,
                dict
            ):
                continue


            role = message.get(
                "role",
                ""
            )

            content = message.get(
                "content",
                ""
            )


            if not content:
                continue


            if role == "user":

                history_text += (
                    f"User: {content}\n"
                )

            elif role == "assistant":

                history_text += (
                    f"Assistant: {content}\n"
                )


    # -----------------------------------------------------
    # SYSTEM PROMPT
    # -----------------------------------------------------

    system_prompt = """
You are Corvit AI Assistant.

You are a helpful assistant for Corvit Systems.

Answer the user's question using the provided knowledge base context.

Important rules:

1. Give accurate and concise answers.
2. Prefer information from the provided context.
3. Do not invent campus locations, courses, fees, timings, contacts,
   policies, or other Corvit information.
4. If the answer is clearly available in the context, answer directly.
5. If the context does not contain the answer, politely say that
   the information is not available in the current knowledge base.
6. Do not mention FAISS, embeddings, RAG, vector databases,
   retrieval systems, or internal technical details to the user.
7. Maintain a professional and friendly tone.
8. If the user asks a simple question, give a simple answer.
"""


    # -----------------------------------------------------
    # USER PROMPT
    # -----------------------------------------------------

    user_prompt = f"""
Knowledge Base Context:

{context}


Previous Conversation:

{history_text}


Current User Question:

{question}


Answer the user's question based on the knowledge base.
"""


    # -----------------------------------------------------
    # CALL GROQ
    # -----------------------------------------------------

    try:

        response = client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            temperature=0.2,

            max_tokens=700
        )


        answer = response.choices[0].message.content


        if not answer:

            answer = (
                "Sorry, I could not generate an answer."
            )


        return (
            answer.strip(),
            documents
        )


    except Exception as e:

        error_message = str(e)


        # Do not expose API key or sensitive details

        st.error(
            f"Groq API error: {error_message}"
        )


        return (
            "Sorry, I could not connect to the AI service. "
            "Please try again.",
            documents
        )


# =========================================================
# OPTIONAL: SIMPLE TEST FUNCTION
# =========================================================

def test_groq_connection():

    client = get_groq_client()

    if client is None:

        return False

    try:

        response = client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[
                {
                    "role": "user",
                    "content": "Say OK"
                }
            ],

            max_tokens=10
        )

        return bool(response)

    except Exception:

        return False

