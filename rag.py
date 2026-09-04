
# ============================================================
# rag.py
# Corvit AI Assistant - RAG Backend
# ============================================================

import os
import pickle

import faiss
import streamlit as st

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# LOAD LOCAL .ENV
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

VECTORSTORE_DIR = "vectorstore"

INDEX_PATH = os.path.join(
    VECTORSTORE_DIR,
    "corvit.index"
)

CHUNKS_PATH = os.path.join(
    VECTORSTORE_DIR,
    "chunks.pkl"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

GROQ_MODEL = "openai/gpt-oss-20b"

TOP_K = 5


# ============================================================
# GET GROQ API KEY
# ============================================================

def get_groq_api_key():

    # --------------------------------------------------------
    # 1. STREAMLIT CLOUD SECRETS
    # --------------------------------------------------------

    try:

        if "GROQ_API_KEY" in st.secrets:

            api_key = st.secrets["GROQ_API_KEY"]

            if api_key:

                api_key = str(api_key).strip()

                if api_key:

                    print(
                        "GROQ_API_KEY FOUND IN STREAMLIT SECRETS"
                    )

                    return api_key

    except Exception as e:

        print(
            "STREAMLIT SECRETS ERROR:",
            str(e)
        )


    # --------------------------------------------------------
    # 2. ENVIRONMENT VARIABLE
    # --------------------------------------------------------

    try:

        api_key = os.environ.get(
            "GROQ_API_KEY"
        )

        if api_key:

            api_key = str(api_key).strip()

            if api_key:

                print(
                    "GROQ_API_KEY FOUND IN ENVIRONMENT"
                )

                return api_key

    except Exception as e:

        print(
            "ENVIRONMENT VARIABLE ERROR:",
            str(e)
        )


    # --------------------------------------------------------
    # 3. LOCAL .ENV FILE
    # --------------------------------------------------------

    try:

        load_dotenv()

        api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if api_key:

            api_key = str(api_key).strip()

            if api_key:

                print(
                    "GROQ_API_KEY FOUND IN .ENV"
                )

                return api_key

    except Exception as e:

        print(
            ".ENV ERROR:",
            str(e)
        )


    # --------------------------------------------------------
    # NOTHING FOUND
    # --------------------------------------------------------

    print(
        "GROQ_API_KEY NOT FOUND"
    )

    return None


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client():

    api_key = get_groq_api_key()


    # --------------------------------------------------------
    # DEBUG INFORMATION
    # --------------------------------------------------------

    if api_key:

        print(
            "GROQ_API_KEY FOUND: True"
        )

        print(
            "GROQ_API_KEY LENGTH:",
            len(api_key)
        )

    else:

        print(
            "GROQ_API_KEY FOUND: False"
        )

        try:

            print(
                "AVAILABLE STREAMLIT SECRETS:",
                list(st.secrets.keys())
            )

        except Exception as e:

            print(
                "Could not read Streamlit secrets:",
                str(e)
            )


    # --------------------------------------------------------
    # KEY NOT AVAILABLE
    # --------------------------------------------------------

    if not api_key:

        st.error(
            "GROQ_API_KEY was not found. "
            "Please check your Streamlit Cloud Secrets "
            "or local .env file."
        )

        return None


    # --------------------------------------------------------
    # CREATE GROQ CLIENT
    # --------------------------------------------------------

    try:

        client = Groq(
            api_key=api_key
        )

        print(
            "GROQ CLIENT CREATED: True"
        )

        return client

    except Exception as e:

        print(
            "GROQ CLIENT ERROR:",
            str(e)
        )

        st.error(
            "Could not initialize the Groq AI service."
        )

        return None


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    try:

        print(
            "Loading embedding model..."
        )

        model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        print(
            "Embedding model loaded successfully."
        )

        return model

    except Exception as e:

        print(
            "EMBEDDING MODEL ERROR:",
            str(e)
        )

        st.error(
            "Could not load the embedding model."
        )

        return None


# ============================================================
# LOAD FAISS INDEX
# ============================================================

@st.cache_resource
def load_faiss_index():

    if not os.path.exists(INDEX_PATH):

        st.error(
            f"FAISS index not found: {INDEX_PATH}"
        )

        print(
            "FAISS INDEX NOT FOUND:",
            INDEX_PATH
        )

        return None


    try:

        print(
            "Loading FAISS index..."
        )

        index = faiss.read_index(
            INDEX_PATH
        )

        print(
            "FAISS index loaded successfully."
        )

        print(
            "FAISS INDEX SIZE:",
            index.ntotal
        )

        return index

    except Exception as e:

        print(
            "FAISS ERROR:",
            str(e)
        )

        st.error(
            "Could not load the FAISS vector store."
        )

        return None


# ============================================================
# LOAD CHUNKS
# ============================================================

@st.cache_resource
def load_chunks():

    if not os.path.exists(CHUNKS_PATH):

        st.error(
            f"Chunks file not found: {CHUNKS_PATH}"
        )

        print(
            "CHUNKS FILE NOT FOUND:",
            CHUNKS_PATH
        )

        return None


    try:

        print(
            "Loading chunks..."
        )

        with open(
            CHUNKS_PATH,
            "rb"
        ) as file:

            chunks = pickle.load(
                file
            )


        print(
            "Chunks loaded successfully."
        )

        print(
            "NUMBER OF CHUNKS:",
            len(chunks)
        )

        return chunks

    except Exception as e:

        print(
            "CHUNKS ERROR:",
            str(e)
        )

        st.error(
            "Could not load chunks.pkl."
        )

        return None


# ============================================================
# LOAD ALL RAG COMPONENTS
# ============================================================

def load_rag_components():

    model = load_embedding_model()

    index = load_faiss_index()

    chunks = load_chunks()

    return (
        model,
        index,
        chunks
    )


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    question,
    top_k=TOP_K
):

    model, index, chunks = (
        load_rag_components()
    )


    # --------------------------------------------------------
    # CHECK COMPONENTS
    # --------------------------------------------------------

    if model is None:

        return []


    if index is None:

        return []


    if chunks is None:

        return []


    # --------------------------------------------------------
    # CREATE QUERY EMBEDDING
    # --------------------------------------------------------

    try:

        query_embedding = model.encode(
            [question],
            convert_to_numpy=True
        )

        query_embedding = (
            query_embedding.astype(
                "float32"
            )
        )


    except Exception as e:

        print(
            "QUERY EMBEDDING ERROR:",
            str(e)
        )

        return []


    # --------------------------------------------------------
    # SEARCH FAISS
    # --------------------------------------------------------

    try:

        distances, indices = (
            index.search(
                query_embedding,
                top_k
            )
        )

    except Exception as e:

        print(
            "FAISS SEARCH ERROR:",
            str(e)
        )

        return []


    # --------------------------------------------------------
    # COLLECT RESULTS
    # --------------------------------------------------------

    retrieved_documents = []


    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx < 0:

            continue


        if idx >= len(chunks):

            continue


        chunk = chunks[idx]


        # ----------------------------------------------------
        # Handle different chunk formats
        # ----------------------------------------------------

        if isinstance(
            chunk,
            dict
        ):

            text = (
                chunk.get("text")
                or chunk.get("content")
                or chunk.get("page_content")
                or str(chunk)
            )

        else:

            text = str(chunk)


        retrieved_documents.append(
            {
                "text": text,
                "score": float(distance),
                "index": int(idx)
            }
        )


    print(
        "RETRIEVED DOCUMENTS:",
        len(retrieved_documents)
    )

    return retrieved_documents


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    documents
):

    if not documents:

        return (
            "No relevant information was found "
            "in the knowledge base."
        )


    context_parts = []


    for i, document in enumerate(
        documents,
        start=1
    ):

        text = document.get(
            "text",
            ""
        )


        if not text:

            continue


        context_parts.append(
            f"Document {i}:\n{text}"
        )


    if not context_parts:

        return (
            "No relevant information was found "
            "in the knowledge base."
        )


    return "\n\n".join(
        context_parts
    )


# ============================================================
# FORMAT CHAT HISTORY
# ============================================================

def format_chat_history(
    chat_history
):

    if not chat_history:

        return ""


    history_parts = []


    # Only use recent messages

    recent_messages = (
        chat_history[-6:]
    )


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

            history_parts.append(
                f"User: {content}"
            )

        elif role == "assistant":

            history_parts.append(
                f"Assistant: {content}"
            )


    return "\n".join(
        history_parts
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    chat_history=None
):

    # --------------------------------------------------------
    # VALIDATE QUESTION
    # --------------------------------------------------------

    if not question:

        return (
            "Please enter a question.",
            []
        )


    question = str(
        question
    ).strip()


    if not question:

        return (
            "Please enter a question.",
            []
        )


    # --------------------------------------------------------
    # RETRIEVE RELEVANT DOCUMENTS
    # --------------------------------------------------------

    documents = retrieve_documents(
        question,
        TOP_K
    )


    # --------------------------------------------------------
    # BUILD KNOWLEDGE CONTEXT
    # --------------------------------------------------------

    context = build_context(
        documents
    )


    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    history = format_chat_history(
        chat_history
    )


    # --------------------------------------------------------
    # GET GROQ CLIENT
    # --------------------------------------------------------

    client = get_groq_client()


    if client is None:

        return (
            "The AI service is not configured correctly. "
            "Please check the GROQ_API_KEY configuration.",
            documents
        )


    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    system_prompt = """
You are Corvit AI Assistant.

You answer questions about Corvit Systems using
the provided knowledge base.

RULES:

1. Use the provided knowledge base as the primary source.

2. Give direct, helpful and concise answers.

3. Do not invent information about Corvit.

4. If the answer is available in the knowledge base,
   answer the question directly.

5. If the answer cannot be found in the knowledge base,
   politely say that the information is not available
   in the current knowledge base.

6. Do not mention technical details such as:
   FAISS, embeddings, vector stores, RAG, chunks,
   retrieval, or internal implementation.

7. Do not expose API keys or other secrets.

8. Keep the answer professional and friendly.

9. If the user asks about a campus, provide the campus
   location/details found in the knowledge base.

10. Never make up an address or location.
"""


    # ========================================================
    # USER PROMPT
    # ========================================================

    user_prompt = f"""
KNOWLEDGE BASE:

{context}


PREVIOUS CONVERSATION:

{history}


CURRENT QUESTION:

{question}


Answer the current question using the knowledge base.
"""


    # ========================================================
    # CALL GROQ
    # ========================================================

    try:

        print(
            "Sending request to Groq..."
        )

        response = (
            client.chat.completions.create(

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
        )


        # ----------------------------------------------------
        # GET ANSWER
        # ----------------------------------------------------

        answer = (
            response
            .choices[0]
            .message
            .content
        )


        if not answer:

            answer = (
                "Sorry, I could not generate "
                "an answer."
            )


        print(
            "Groq response received successfully."
        )


        return (
            answer.strip(),
            documents
        )


    # ========================================================
    # GROQ ERROR
    # ========================================================

    except Exception as e:

        print(
            "GROQ API ERROR:",
            str(e)
        )


        # Do not expose technical error to user

        return (
            "Sorry, I could not connect to the AI service. "
            "Please try again.",
            documents
        )


# ============================================================
# OPTIONAL GROQ CONNECTION TEST
# ============================================================

def test_groq_connection():

    client = get_groq_client()


    if client is None:

        return False


    try:

        response = (
            client.chat.completions.create(

                model=GROQ_MODEL,

                messages=[
                    {
                        "role": "user",
                        "content": "Reply with OK."
                    }
                ],

                max_tokens=10
            )
        )


        if response:

            print(
                "GROQ CONNECTION TEST: SUCCESS"
            )

            return True


    except Exception as e:

        print(
            "GROQ CONNECTION TEST FAILED:",
            str(e)
        )


    return False

