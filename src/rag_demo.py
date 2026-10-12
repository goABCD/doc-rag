#!/usr/bin/env python3
"""Minimal RAG demo: chunk -> embed -> FAISS -> chat completion.

Requires NVIDIA_API_KEY in the environment (never hardcode secrets).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import requests

EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"
CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
EMBED_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
CHAT_MODEL = "meta/llama-3.1-70b-instruct"


def require_api_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not key:
        print("Missing NVIDIA_API_KEY. Export it first, e.g.:")
        print('  export NVIDIA_API_KEY="nvapi-..."')
        print("On Windows PowerShell:")
        print('  $env:NVIDIA_API_KEY="nvapi-..."')
        sys.exit(1)
    return key


def chunk_text(text: str, chunk_size: int) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def embed_batch(headers: dict, texts: list[str], input_type: str) -> list[list[float]]:
    payload = {"input": texts, "model": EMBED_MODEL, "input_type": input_type}
    res = requests.post(EMBED_URL, headers=headers, json=payload, timeout=60)
    res.raise_for_status()
    data = res.json()["data"]
    return [item["embedding"] for item in data]


def build_index(headers: dict, chunks: list[str], batch_size: int) -> faiss.IndexFlatL2:
    vectors: list[list[float]] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors.extend(embed_batch(headers, batch, "passage"))
        print(f"  embedded {min(i + batch_size, len(chunks))}/{len(chunks)}")
        time.sleep(0.3)
    arr = np.array(vectors, dtype="float32")
    index = faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)
    return index


def answer(headers: dict, question: str, context: str) -> str:
    prompt = (
        "You are a careful analyst. Answer ONLY from the reference material. "
        "If the material does not contain the answer, say you do not know.\n"
        f"[Reference]\n{context}\n\n[Question] {question}\nAnswer:"
    )
    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }
    res = requests.post(CHAT_URL, headers=headers, json=payload, timeout=90)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Local document RAG demo (NVIDIA API + FAISS)")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "sample_corpus.txt",
        help="UTF-8 text file to index",
    )
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    api_key = require_api_key()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    if not args.file.is_file():
        print(f"File not found: {args.file}")
        sys.exit(1)

    text = args.file.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_text(text, args.chunk_size)
    if not chunks:
        print("Empty document.")
        sys.exit(1)

    print("=" * 60)
    print("Document RAG demo (chunk -> embed -> FAISS -> chat)")
    print("=" * 60)
    print(f"File: {args.file}  chunks: {len(chunks)}")
    print("Building FAISS index...")
    index = build_index(headers, chunks, args.batch_size)
    print("Ready. Type a question (q to quit).")

    while True:
        question = input("\n> ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break
        if not question:
            continue
        q_vec = np.array(embed_batch(headers, [question], "query"), dtype="float32")
        _dist, idxs = index.search(q_vec, min(args.top_k, len(chunks)))
        context = "\n\n---\n\n".join(chunks[i] for i in idxs[0])
        print("Retrieving and answering...")
        print(answer(headers, question, context))


if __name__ == "__main__":
    main()
