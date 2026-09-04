import os
import re
import pickle
import pymupdf
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


# ==============================
# CONFIGURATION
# ==============================

PDF_PATH = "data/corvit.pdf"
VECTORSTORE_DIR = "vectorstore"

INDEX_PATH = os.path.join(VECTORSTORE_DIR, "corvit.index")
CHUNKS_PATH = os.path.join(VECTORSTORE_DIR, "chunks.pkl")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120


# ==============================
# CREATE DIRECTORIES
# ==============================

os.makedirs(VECTORSTORE_DIR, exist_ok=True)


# ==============================
# LOAD PDF
# ==============================

def load_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            "Please put your PDF inside the data folder."
        )

    doc = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text")

        if text and text.strip():
            pages.append({
                "page": page_number,
                "text": text
            })

    doc.close()

    if not pages:
        raise ValueError("No readable text was found in the PDF.")

    return pages


# ==============================
# CLEAN TEXT
# ==============================

def clean_text(text):
    text = text.replace("\x00", " ")

    # Remove excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


# ==============================
# CHUNK TEXT
# ==============================

def chunk_text(text, page_number):
    text = clean_text(text)

    # Split mainly by paragraphs
    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    chunks = []

    current = ""

    for paragraph in paragraphs:

        # If paragraph itself is very large
        if len(paragraph) > CHUNK_SIZE:

            words = paragraph.split()

            while words:
                available_words = max(1, CHUNK_SIZE // 5)

                part_words = words[:available_words]
                words = words[available_words:]

                part = " ".join(part_words).strip()

                if part:
                    chunks.append({
                        "text": part,
                        "page": page_number
                    })

            current = ""
            continue

        candidate = (
            current + "\n\n" + paragraph
            if current
            else paragraph
        )

        if len(candidate) <= CHUNK_SIZE:
            current = candidate

        else:
            if current:
                chunks.append({
                    "text": current.strip(),
                    "page": page_number
                })

            # Add small overlap
            overlap_words = current.split()[-25:] if current else []
            overlap = " ".join(overlap_words)

            current = (
                overlap + "\n\n" + paragraph
                if overlap
                else paragraph
            )

    if current:
        chunks.append({
            "text": current.strip(),
            "page": page_number
        })

    return chunks


# ==============================
# MAIN INGESTION
# ==============================

def main():

    print("=" * 60)
    print("CORVIT RAG - DOCUMENT INGESTION")
    print("=" * 60)

    print("\n[1/4] Loading PDF...")

    pages = load_pdf(PDF_PATH)

    print(f"Loaded {len(pages)} pages.")

    print("\n[2/4] Creating chunks...")

    all_chunks = []

    for page in pages:
        page_chunks = chunk_text(
            page["text"],
            page["page"]
        )

        all_chunks.extend(page_chunks)

    if not all_chunks:
        raise ValueError("No chunks were created from the PDF.")

    print(f"Created {len(all_chunks)} chunks.")

    print("\n[3/4] Loading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = embeddings.astype("float32")

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    print("\n[4/4] Building FAISS index...")

    dimension = embeddings.shape[1]

    # Inner Product + normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    faiss.write_index(
        index,
        INDEX_PATH
    )

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print("\n" + "=" * 60)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"FAISS index : {INDEX_PATH}")
    print(f"Chunks file : {CHUNKS_PATH}")
    print(f"Vectors     : {index.ntotal}")
    print(f"Dimension   : {dimension}")


if __name__ == "__main__":
    main()