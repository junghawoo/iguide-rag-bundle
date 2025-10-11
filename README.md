
# I-GUIDE RAG Search Bundle (OpenSearch + API + UI + Ollama)

This bundle spins up:
- OpenSearch (w/ security disabled for local dev)
- OpenSearch Dashboards
- Ollama (local LLM for optional query rewrite)
- FastAPI search service (hybrid BM25 + KNN via RRF)
- Streamlit UI

## Quick start

```bash
# 1) Copy env template
cp .env.example .env

# 2) Bring everything up
docker compose up -d --build

# 3) (Optional) Pull a small model in Ollama for query rewrites
docker exec -it ollama ollama pull qwen2.5:3b-instruct

# 4) Verify:
curl http://localhost:8088/health
# UI:
# http://localhost:8501
# Dashboards:
# http://localhost:5601
```

## Index & Data

Set `OS_INDEX` (default `iguide_platform`) in `.env` to your existing index.
Your documents should include:
- `title` (text)
- `abstract` (text) or `contents` (text)
- `tags` (keyword) (optional)
- `authors` (keyword) (optional)
- `notebook_url` (keyword) → the link that launches on I-GUIDE
- `spatial-embedding` (knn_vector, dim must match your embeddings)

If you have *not* registered a server-side model in OpenSearch for `text_embedding`,
change the API to send a client-side `query_vector` instead (I can provide that patch).
Alternatively, register `sentence-transformers/all-MiniLM-L6-v2` in OpenSearch and keep the current API.

## Notes

- This compose disables OpenSearch security for simplicity. For staging/prod, enable security and TLS,
  set `OS_USER/OS_PASS`, and flip `use_ssl=True` in the API.
- The UI toggles LLM-based query rewrites. It's optional and off by default.
- To chunk notebooks later, create a `*_chunks` index and point the API to it; group results by notebook in the UI.
