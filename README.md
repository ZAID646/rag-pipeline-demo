# RAG Pipeline Demo

End-to-end Retrieval-Augmented Generation pipeline — ingest documents, chunk and embed them, then retrieve relevant context for LLM-powered question answering.

**[→ Live Demo](https://zaid646-rag-pipeline-demo.hf.space)**

---

## Overview

Built with FAISS for vector search and NVIDIA embeddings, this RAG system processes unstructured text into searchable chunks and answers questions using retrieved context.

### Pipeline

```
1. Ingest → 2. Chunk → 3. Embed → 4. Index → 5. Retrieve → 6. Generate
```

| Stage | Detail |
|---|---|
| **Chunking** | Recursive split (paragraphs → sentences → words), merge to ~500 chars |
| **Embeddings** | NVIDIA `minimaxai/minimax-m3` via OpenAI-compatible API |
| **Vector Index** | FAISS in-memory, dot-product similarity |
| **Generation** | NVIDIA `minimaxai/minimax-m3` with context injection |

---

## Running Locally

```bash
git clone https://github.com/zaid646/rag-pipeline-demo.git
cd rag-pipeline-demo
pip install -r requirements.txt
python app.py
```

### Required Environment Variables

| Variable | Description |
|---|---|
| `NVIDIA_API_KEY` | NVIDIA API key (embeddings + generation) |

---

## Usage

1. **Paste or upload** a document in the text area
2. Click **Ingest** — the system chunks, embeds, and indexes the document
3. **Ask questions** in the chat interface — replies draw from the ingested content

---

## License

MIT — see [LICENSE](LICENSE).
