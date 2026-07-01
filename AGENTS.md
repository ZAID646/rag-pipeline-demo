# rag-pipeline-demo

## Session Context (Jul 2026)

### Status
- Deployed at: https://zaid646-rag-pipeline-demo.hf.space
- Working: document upload, chunking, embedding, semantic search

### Key Decisions
- Uses NVIDIA minimax-m3 for embeddings
- Recursive split + merge for chunking
- FAISS for vector search in-memory

### Fixed Bugs
1. Off-by-one stripping in chunker — `strip().strip(".,!?;:")` removed trailing `. ` separator then stripped punctuation, leaving chunks empty. Fixed by removing separator before strip, and skipping empty chunks post-processing.

### Architecture
- Chunker: recursive split by paragraphs → sentences → words, merge until target size
- Embeddings: NVIDIA minimax-m3 via OpenAI-compatible API
- Search: dot-product similarity over FAISS index
- UI: Gradio with upload/chat interface
