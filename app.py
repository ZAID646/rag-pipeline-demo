import json
import os
import re
import numpy as np
from openai import OpenAI
import gradio as gr

PROVIDERS = {
    "Cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "models": ["gpt-oss-120b"],
    },
    "NVIDIA": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY",
        "models": ["minimaxai/minimax-m3"],
    },
}

_sentence_transformer = None
_cross_encoder = None
_chroma_collection = None
_bm25 = None
_bm25_docs_texts = []
_chunk_count = 0

RAG_PROMPT = """You are a helpful Q&A assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer concisely based on the context above."""


def get_encoder():
    global _sentence_transformer
    if _sentence_transformer is None:
        from sentence_transformers import SentenceTransformer
        _sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_transformer


def get_reranker():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path="./chroma_db")
        _chroma_collection = _chroma_client.get_or_create_collection(
            name="rag_docs",
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collection


def chunk_text(text, chunk_size=512, chunk_overlap=64):
    text = text.strip()
    if not text:
        return []
    separators = ["\n\n", "\n", " ", ""]

    def _split(text, depth=0):
        if not text or len(text) <= chunk_size:
            return [text] if text else []
        if depth >= len(separators):
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        sep = separators[depth]
        if not sep:
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        parts = text.split(sep)
        result = []
        for part in parts:
            if not part:
                continue
            if len(part) <= chunk_size:
                if part.strip():
                    result.append(part.strip())
            else:
                result.extend(_split(part, depth + 1))
        return result

    splits = _split(text)
    if not splits:
        return []

    chunks = []
    buffer = ""
    for split in splits:
        if not buffer:
            buffer = split
        elif len(buffer) + len(split) + 1 <= chunk_size:
            buffer += " " + split
        else:
            if not chunks or buffer != chunks[-1]:
                chunks.append(buffer)
            if chunk_overlap > 0 and len(buffer) > chunk_overlap:
                overlap_text = buffer[-chunk_overlap:]
                space_pos = overlap_text.find(" ")
                if space_pos > 0:
                    overlap_text = overlap_text[space_pos + 1:]
                buffer = overlap_text
            else:
                buffer = ""
            if buffer and len(buffer) + len(split) + 1 <= chunk_size:
                buffer += " " + split
            else:
                if buffer and (not chunks or buffer != chunks[-1]):
                    chunks.append(buffer)
                buffer = split
    if buffer and (not chunks or buffer != chunks[-1]):
        chunks.append(buffer)

    return [c.strip() for c in chunks if c.strip()]


def ingest_documents(text, progress=gr.Progress()):
    global _bm25, _bm25_docs_texts, _chunk_count

    progress(0, desc="Chunking text...")
    chunks = chunk_text(text)
    if not chunks:
        return "No text to ingest."

    progress(0.2, desc="Loading embedding model...")
    encoder = get_encoder()

    progress(0.4, desc=f"Embedding {len(chunks)} chunks...")
    embeddings = encoder.encode(chunks).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    progress(0.7, desc="Storing in vector database...")
    collection = get_collection()
    collection.add(ids=ids, embeddings=embeddings, documents=chunks)

    progress(0.85, desc="Building BM25 index...")
    from rank_bm25 import BM25Okapi
    tokenized = [c.split() for c in chunks]
    _bm25 = BM25Okapi(tokenized)
    _bm25_docs_texts = chunks

    _chunk_count += len(chunks)
    progress(1.0, desc="Done!")
    return f"✅ Ingested {len(chunks)} chunks. Total: {_chunk_count} chunks in database."


def hybrid_search(query, k=5):
    global _bm25, _bm25_docs_texts
    collection = get_collection()
    encoder = get_encoder()

    query_emb = encoder.encode([query]).tolist()[0]
    dense_results = collection.query(query_embeddings=[query_emb], n_results=k)
    dense_docs = []
    for i in range(len(dense_results["documents"][0])):
        dense_docs.append({
            "text": dense_results["documents"][0][i],
            "score": 1.0 - dense_results["distances"][0][i],
        })

    if _bm25 and _bm25_docs_texts:
        tokenized_query = query.split()
        bm25_scores = _bm25.get_scores(tokenized_query)
        seen = {d["text"] for d in dense_docs}
        for i, score in enumerate(bm25_scores):
            if score > 0 and _bm25_docs_texts[i] not in seen:
                dense_docs.append({
                    "text": _bm25_docs_texts[i],
                    "score": float(score) * 0.1,
                })

    dense_docs.sort(key=lambda x: x["score"], reverse=True)
    return dense_docs[:k]


def rerank(query, documents):
    reranker = get_reranker()
    if len(documents) <= 1:
        return documents
    pairs = [[query, d["text"]] for d in documents]
    scores = reranker.predict(pairs)
    if scores.ndim > 1:
        scores = scores[:, 0] if scores.shape[1] > 1 else scores.squeeze()
    scored = list(zip(documents, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [{**doc, "score": float(score)} for doc, score in scored]


def generate_answer(query, documents, provider_name, model):
    provider = PROVIDERS.get(provider_name)
    if not provider:
        return "Unknown provider."

    api_key = os.environ.get(provider["api_key_env"], "")
    if not api_key:
        return f"API key not set for {provider_name}."

    context = "\n\n".join(
        f"[Source {i+1}] {d['text']}" for i, d in enumerate(documents)
    )
    prompt = RAG_PROMPT.format(context=context, question=query)

    client = OpenAI(api_key=api_key, base_url=provider["base_url"])
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating answer: {e}"


def format_chat_response(message, history, provider, model):
    history = history or []

    if _chunk_count == 0:
        yield "Please ingest some documents first using the text area above, then ask questions."
        return

    yield "🔍 **Searching documents...**"

    retrieved = hybrid_search(message, k=5)
    if not retrieved:
        yield "No relevant documents found. Try ingesting more text or rephrasing your question."
        return

    yield "📊 **Reranking results...**"
    ranked = rerank(message, retrieved)

    yield "🤔 **Generating answer...**"
    answer = generate_answer(message, ranked, provider, model)

    sources = "\n\n".join(
        f"> **Source {i+1}** (relevance: {d['score']:.3f})\n> _{d['text'][:200]}..."
        for i, d in enumerate(ranked[:3])
    )

    result = f"### Answer\n\n{answer}\n\n---\n\n### Sources\n\n{sources}"
    yield result


def update_models(provider_name):
    provider = PROVIDERS.get(provider_name)
    if provider:
        models = provider["models"]
        return gr.Dropdown(choices=models, value=models[0])
    return gr.Dropdown(choices=[], value=None)


with gr.Blocks(
    title="RAG Pipeline Demo",
    theme=gr.themes.Soft(),
    fill_height=True,
) as demo:
    gr.Markdown(
        "# 🔍 RAG Pipeline Demo\n"
        "### Hybrid Search + Reranking + Multi-Provider Generation\n\n"
        "1. Paste text below and click **Ingest** to load it into the vector database\n"
        "2. Ask questions in the chat — the system retrieves relevant passages and answers"
    )

    provider_dd = gr.Dropdown(
        choices=list(PROVIDERS.keys()),
        value=list(PROVIDERS.keys())[0],
        label="AI Provider",
        scale=1,
    )
    model_dd = gr.Dropdown(
        choices=PROVIDERS[list(PROVIDERS.keys())[0]]["models"],
        value=PROVIDERS[list(PROVIDERS.keys())[0]]["models"][0],
        label="Model",
        scale=1,
    )

    provider_dd.change(fn=update_models, inputs=provider_dd, outputs=model_dd)

    with gr.Row():
        ingest_input = gr.Textbox(
            label="📄 Document Input",
            placeholder="Paste your document text here...",
            lines=5,
            scale=4,
        )
        ingest_btn = gr.Button("📥 Ingest", variant="primary", scale=1, min_width=100)

    ingest_status = gr.Markdown("No documents ingested yet.")

    ingest_btn.click(
        fn=ingest_documents,
        inputs=ingest_input,
        outputs=ingest_status,
    )

    gr.Markdown("---")
    gr.ChatInterface(
        fn=format_chat_response,
        additional_inputs=[provider_dd, model_dd],
        title="Ask Questions",
        description="Ask questions about your ingested documents.",
    )

if __name__ == "__main__":
    demo.launch()
