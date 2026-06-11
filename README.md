# LLM-RAG Terminal App

This app builds a Retrieval-Augmented Generation (RAG) pipeline using:
- LangChain
- ChromaDB
- Ollama (`ChatOllama` + `OllamaEmbeddings`) — runs locally, no API key

It ingests the Alice PDF from `documents/`, chunks the text, stores embeddings in Chroma, retrieves relevant chunks for a user prompt, augments the prompt with context, and queries the LLM.

## 1) Install Ollama

Install [Ollama](https://ollama.com/) and start it, then pull the models used by the app (or change `OLLAMA_EMBED_MODEL` / `OLLAMA_CHAT_MODEL` in `rag_app.py`):

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

## 2) Python setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3) Run

```bash
python rag_app.py Who is Alice?
```

Optional flags:
- `--k 6` — retrieve more chunks
- `--c 1500` — The size of the chunks
- `--o 300` — The overlap size of the chunks
- `--rebuild` — rebuild the Chroma collection from the PDF (use after switching embed models or the source PDF)

Example:

```bash
python rag_app.py --rebuild --k 6 --c 1500 --o 300 What advice does the Caterpillar give Alice?
```

## 4) Evaluation

Run to get an evaluation of the RAG correctness and groundedness using LLM-as-judge.
After running the script, it will print in the terminal the Langsmith link.

```bash
ollama pull mistral
```

```bash
python evaluation\evaluate.py
```


## Notes

- Chroma DB is persisted in `chroma_db/`.
- First run can take longer while Ollama embeds all chunks.
- Evaluation requires a Langsmith account.
- Run the app once before running the evaluation to make it create the Chroma DB