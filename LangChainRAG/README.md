# 📚 LangChain + Ollama RAG WebUI  Sample

A minimal Streamlit web‑app that demonstrates **Retrieval‑Augmented Generation (RAG)** using:

- **LangChain** – high‑level abstractions for LLMs, embeddings, and retrieval chains.
- **Ollama** – a local, open‑source LLM runtime (no Azure/Cloud required).
- **Chroma** – an efficient vector‑store for embeddings.

Upload a plain‑text document, let Ollama embed it, and ask a question.  
The app will retrieve the most relevant passages and generate an answer with the specified LLM.

## Features

| Feature | Details |
|---------|---------|
| **Local LLM** | Uses Ollama models (`llama3.2:3b`, `mxbai-embed-large`). No API key or internet connection required. |
| **Embedding + Retrieval** | `OllamaEmbeddings` → `Chroma` → `RetrievalChain`. |
| **Simple UI** | Upload documents, ask questions, view answers – all in one Streamlit page. |
| **Persistent DB** | Vector store is saved to `./chroma_db`, so uploads survive restarts. |

## Prerequisites

- **Python 3.10+** (tested on 3.11)
- **Ollama** installed locally and running on `http://localhost:11434/`.  
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull llama3.2:3b
  ollama pull mxbai-embed-large
  ```
- **pip** (or `pipx` for isolated environments)

## Installation

```bash
# clone the repo (if you haven't already)
git clone https://github.com/maishiro/python-repo.git
cd python-repo/LangChainRAG

# (optional) create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py
```

The app will open in your default browser.  
Upload a `.txt` file → type a question → hit **検索＆回答** to see the answer.

## File Structure

```
├── app.py            # Streamlit app + LangChain logic
├── requirements.txt  # Python dependencies
└── chroma_db/        # (generated) Chroma vector store
```

## Customization

| Setting | File | How to change |
|---------|------|---------------|
| Embedding model | `app.py` | `EMBED_MODEL = "your-embedding-model"` |
| LLM model | `app.py` | `LLM_MODEL = "your-llm-model"` |
| Chunk size | `app.py` | `RecursiveCharacterTextSplitter(chunk_size=…)` |
| Number of retrieved docs | `app.py` | `search_kwargs={"k": <number>}` |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| **Ollama not found** | Make sure `ollama` is in your `$PATH` and running (`ollama serve`). |
| **Embedding/LLM errors** | Verify the model names match the ones you've pulled (`ollama list`). |
| **Streamlit crashes** | Ensure you have the latest `streamlit` and `langchain` releases. |

```