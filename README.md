---
title: RAG Pipeline Demo
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.36.2
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# RAG Pipeline Demo

End-to-end RAG pipeline with hybrid search (dense + BM25), cross-encoder reranking, and multi-provider answer generation.

## How it works

1. **Ingest** — Paste text, it gets chunked, embedded (all-MiniLM-L6-v2), and stored in ChromaDB
2. **Retrieve** — Hybrid search: dense (vector) + sparse (BM25) with fusion
3. **Rerank** — Cross-encoder (ms-marco-MiniLM-L-6-v2) re-scores top results
4. **Generate** — Selected provider (Cerebras/NVIDIA) answers using retrieved context

## API Keys

| Secret | Value |
|---|---|
| `CEREBRAS_API_KEY` | Your Cerebras API key |
| `NVIDIA_API_KEY` | Your NVIDIA API key |
