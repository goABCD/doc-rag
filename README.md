# Document RAG Demo

Minimal **Retrieval-Augmented Generation** pipeline (no LangChain):

1. Chunk a UTF-8 text file
2. Embed chunks via NVIDIA Integrate API (open model)
3. Store vectors in local **FAISS** (`IndexFlatL2`)
4. Retrieve top‑k passages and ask a chat model to answer **only** from that context

Built from a local experiment under `AI_CREATED/Test` and cleaned for public use: **API keys are read from the environment**, never committed.

## Why this exists

Portfolio demo for data / information retrieval + applied management Q&A: turn unstructured text into a searchable vector store and ground answers in retrieved passages.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Get an API key from [NVIDIA Build / Integrate](https://build.nvidia.com/), then:

```bash
# macOS / Linux
export NVIDIA_API_KEY="nvapi-..."

# Windows PowerShell
$env:NVIDIA_API_KEY="nvapi-..."
```

## Run

```bash
python src/rag_demo.py
# or point at your own corpus
python src/rag_demo.py --file path/to/your.txt --chunk-size 800 --top-k 3
```

Default corpus: `data/sample_corpus.txt` (short Macau diversification demo text).

## Limits (be honest in interviews)

- Infinite / cloud embeddings + chat; cost and rate limits depend on your NVIDIA quota
- `IndexFlatL2` is exact but not scaled for huge corpora
- Quality follows the source document; garbage in → garbage out
- Demo has no UI, auth, or evaluation harness—just the core loop

## Security

- Put secrets in `NVIDIA_API_KEY` only
- If a key was ever pasted into a local script, **revoke it** in the NVIDIA console and issue a new one before sharing screenshots or clones
- Do not commit `.env` or personal chat logs

## License

MIT for this demo code. Third-party models and APIs remain under their own terms.
