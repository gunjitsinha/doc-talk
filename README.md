# Retrieval-based Chatbot

## Overview

Simple retrieval-augmented assistant that answers questions from a local document knowledge base. All embeddings and retrieval run locally.

## Features

- Document chunking and text splitting for context preservation.
- Local embeddings using HuggingFace sentence-transformers.
- FAISS vector store with persistence to disk.
- Retrieval of relevant chunks and LLM answers grounded in retrieved context.
- Streamlit UI for chat and document ingestion.
- Startup indexing of files placed in the `Documents/` folder.
- Optional persistence of uploaded files into `Documents/` via a checkbox.
- Interaction logging to `logs/log.jsonl` (one JSON object per query).

## How it works (short)

- Put `.pdf` or `.txt` files into `Documents/` to have them indexed at startup.
- Uploads are processed temporarily by default and do not survive a restart.
- To persist an uploaded file, enable the "Save uploaded files to Documents" checkbox during upload.
- Queries retrieve top chunks from FAISS and ask the LLM to answer using only that context.
- Each interaction is appended to `logs/log.jsonl` with question, answer, and retrieved chunk snippets.

## Quickstart

1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set the required environment variable for the LLM:

```bash
# Windows PowerShell
set GROQ_API_KEY=your_groq_key_here
# macOS / Linux
export GROQ_API_KEY=your_groq_key_here
```

4. (Optional) Add persistent documents:

- Place files in `Documents/` to have them indexed on startup.

5. Run the app:

```bash
streamlit run app.py
```

6. Upload files in the UI:

- Use the uploader for ad-hoc queries.
- Check "Save uploaded files to Documents" to persist a file into `Documents/` before indexing.

## Logs

- Interactions are written to `logs/log.jsonl` with one JSON object per line.
- Each entry contains `timestamp`, `question`, `answer`, `retrieved_chunks` (rank, source, snippet, score), and `llm_model`.

## Project layout (brief)

- `core/` - ingestion, embeddings, vector store, RAG chain, logger.
- `ui/` - Streamlit components and chat interface.
- `Documents/` - admin-managed folder for persistent documents.
- `logs/` - JSONL interaction logs.

## Tech Stack

* **Language**: Python 3.11+
* **LLM Inference**: Groq (Llama 3.1 70B)
* **Embeddings**: HuggingFace sentence‑transformers (local execution)
* **Vector Store**: FAISS
* **RAG Orchestration**: LangChain
* **UI**: Streamlit

All core retrieval and embedding operations run locally, minimizing external dependencies.

## Notes

- Add files to `Documents/` and restart the app to include them in the default index.
- Uploaded files processed without checking persistence are temporary and removed after processing.
- The FAISS index is persisted to `data/faiss_index` by the vector store.

Generated to reflect current workspace: see `app.py` for the main entrypoint.
