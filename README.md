# 🤖 Corvit AI Assistant — RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built for answering questions about Corvit Systems, including courses, campuses, fees, training programs, and related information.

## 🚀 Features

- RAG-based question answering
  
- FAISS vector database for semantic search
  
- Sentence Transformers for embeddings
  
- Groq API for LLM responses
  
- Streamlit chatbot interface
  
- Multiple chat sessions
  
- Chat history
  
- Suggested questions
  
- Attractive light-theme interface
  
- Knowledge-base-grounded responses

## 📁 Project Structure

```text
RAG corvit system/
│
├── data/
│   └── corvit.pdf
│
├── vectorstore/
│   ├── corvit.index
│   └── chunks.pkl
│
├── .env
├── .gitignore
├── app.py
├── ingest.py
├── rag.py
├── requirements.txt
└── README.md
```

## 🛠️ Technologies Used

- Python

- Streamlit
  
- FAISS
  
- Sentence Transformers
  
- Groq API
  
- PyMuPDF
  
- NumPy
  
- python-dotenv

## ⚙️ Installation

### 1. Clone the Repository

```bash

git clone https://github.com/YOUR_USERNAME/corvit-rag-chatbot.git

cd corvit-rag-chatbot

```

### 2. Create Virtual Environment

For Windows:


python -m venv venv


### 3. Activate Virtual Environment

PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install Requirements

```bash
pip install -r requirements.txt
```

## 🔑 Groq API Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_actual_api_key_here
```

Keep your API key private.



## 📚 Build the FAISS Vector Store

The vector store is generated automatically from the Corvit PDF.

Make sure the PDF is located at:

```text
data/corvit.pdf
```

Then run:

```bash
python ingest.py
```

This automatically creates:

```text
vectorstore/corvit.index
vectorstore/chunks.pkl
```

You do not need to manually create these files.

## ▶️ Run the Application

Start Streamlit with:

```bash
streamlit run app.py
```

The Corvit AI Assistant will open in your browser.

## 🧠 RAG Pipeline

The chatbot works through the following process:

```text
Corvit PDF
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Sentence Transformer
     ↓
Embeddings
     ↓
FAISS Vector Database
     ↓
User Question
     ↓
Semantic Search
     ↓
Relevant Context
     ↓
Groq LLM
     ↓
Final Answer
```

## 📌 Important Notes

- The chatbot uses the provided Corvit knowledge base for its answers.
- The FAISS vector store is generated using `ingest.py`.
- Fees, schedules, phone numbers, course availability, and other changing information should be verified using the latest official Corvit information.
- Do not commit your `.env` file or Groq API key.
- If the vector store is missing, run:

```bash
python ingest.py
```

before starting the application.

## 🎯 Project Purpose

This project demonstrates how Retrieval-Augmented Generation can be used to build a domain-specific AI chatbot.

Instead of relying only on the language model's general knowledge, the system retrieves relevant information from a custom knowledge base and provides that context to the LLM before generating an answer.

## 👩‍💻 Project

**Corvit AI Assistant**

Built with:

**Python • Streamlit • FAISS • Sentence Transformers • Groq API**
