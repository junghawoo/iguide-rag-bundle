import os, requests
from typing import List, Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from opensearchpy import OpenSearch, TransportError

# NEW: local embedding (same as the author)
import torch
from transformers import AutoTokenizer, AutoModel

OS_HOST = os.getenv("OS_HOST", "localhost")
OS_PORT = int(os.getenv("OS_PORT", "9200"))
OS_USER = os.getenv("OS_USER")
OS_PASS = os.getenv("OS_PASS")
INDEX = os.getenv("OS_INDEX") or os.getenv("INDEX") or "iguide_platform"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Embedding model (same one used by the dataset author)
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DEVICE = "cpu"  # keep CPU for simplicity; switch to "cuda" if you have a GPU

app = FastAPI(title="I-GUIDE Notebook Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# If security is enabled, use_ssl should be True.
auth = (OS_USER, OS_PASS) if OS_USER and OS_PASS else None
os_client = OpenSearch(
    hosts=[{"host": OS_HOST, "port": OS_PORT}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=False,       # self-signed in local compose
    ssl_show_warn=False,      # suppress urllib3 InsecureRequestWarning noise
    timeout=30
)

# ---- Embedding init (matches author's notebook) ----
_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(DEVICE)
_model.eval()

def embed_query(text: str) -> list:
    """Mean-pool the last_hidden_state (same as author's notebook)."""
    inputs = _tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = _model(**{k: v.to(DEVICE) for k, v in inputs.items()})
        vec = outputs.last_hidden_state.mean(dim=1)[0]   # (384,)
    return vec.cpu().tolist()

# ------------------ API Schemas ---------------------
class SearchHit(BaseModel):
    id: str
    title: str
    abstract: Optional[str] = None
    authors: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    notebook_url: Optional[str] = None
    score: float
    highlights: Optional[List[str]] = None

class SearchResponse(BaseModel):
    query: str
    rewritten_query: Optional[str] = None
    total: int
    hits: List[SearchHit]

# ----------------- LLM Rewrite (optional) -----------
def rewrite_query_with_llm(q: str) -> str:
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": "qwen2.5:3b-instruct",
                "prompt": f"Rewrite this as a concise search query for research notebooks: {q}"
            },
            timeout=10,
        )
        if resp.ok:
            text = resp.json().get("response", "").strip()
            return text or q
        return q
    except Exception:
        return q

# ----------------- Query Builders -------------------
def _lexical_query(q: str) -> dict:
    """
    BM25 across real fields + exact author boost on authors.keyword.
    """
    mm = {
        "multi_match": {
            "query": q,
            "fields": ["title^3", "contents^1.5", "tags^1.2", "authors^1.2"],
            "type": "best_fields"
        }
    }
    author_exact = {"term": {"authors.keyword": {"value": q, "case_insensitive": True}}}
    return {"bool": {"should": [mm, author_exact]}}

def build_bm25_body(q: str, k: int) -> dict:
    return {
        "size": k,
        "query": _lexical_query(q),
        "highlight": {
            "pre_tags": ["<mark>"], "post_tags": ["</mark>"],
            "fields": {"contents": {}, "title": {}, "tags": {}, "authors": {}}
        }
    }

def build_rrf_retriever_body_with_client_knn(q: str, k: int) -> dict:
    """
    Hybrid RRF using retriever API (OpenSearch 2.12+).
    Uses client-side query embedding (NO server-side model needed).
    """
    lexical = _lexical_query(q)
    qvec = embed_query(q)

    knn = {
        "field": "contents-embedding",   # your vector field (dim 384)
        "query_vector": qvec,
        "k": k,
        "num_candidates": max(k * 5, 100)
    }

    return {
        "size": k,
        "retriever": {
            "rrf": {
                "retrievers": [
                    {"standard": {"query": lexical}},
                    {"knn": {"query": knn}}
                ]
            }
        },
        "highlight": {
            "pre_tags": ["<mark>"], "post_tags": ["</mark>"],
            "fields": {"contents": {}, "title": {}, "tags": {}, "authors": {}}
        }
    }

def _fallback_worthy(exc: Exception) -> bool:
    s = str(exc) if exc else ""
    s_lower = s.lower()
    return ("parsing_exception" in s_lower) or ("retriever" in s_lower) or ("unknown key" in s_lower)

def search_with_fallback(index: str, primary_body: dict, bm25_body: dict) -> dict:
    """
    Try the new 'retriever' body first; if the cluster rejects it, retry with BM25.
    """
    try:
        return os_client.search(index=index, body=primary_body)
    except TransportError as e:
        if _fallback_worthy(e):
            return os_client.search(index=index, body=bm25_body)
        raise
    except Exception as e:
        if _fallback_worthy(e):
            return os_client.search(index=index, body=bm25_body)
        raise

# --------------------- Routes -----------------------
@app.get("/health")
def health():
    try:
        pong = os_client.cluster.health()
        return {"ok": True, "opensearch": pong.get("status"), "index": INDEX}
    except Exception as e:
        return {"ok": False, "error": str(e), "index": INDEX}

@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., description="User query"),
    use_llm: bool = Query(False, description="Rewrite query with LLM first?"),
    k: int = Query(10, ge=1, le=50),
):
    query_text = rewrite_query_with_llm(q) if use_llm else q

    # Build both bodies (primary = hybrid RRF w/ client-side KNN; fallback = BM25)
    body_primary = build_rrf_retriever_body_with_client_knn(query_text, k)
    body_bm25    = build_bm25_body(query_text, k)

    res = search_with_fallback(INDEX, body_primary, body_bm25)

    hits = []
    for h in res.get("hits", {}).get("hits", []):
        src = h.get("_source", {}) or {}
        highlights = []
        for v in h.get("highlight", {}).values():
            highlights.extend(v)
        hits.append(SearchHit(
            id = src.get("id") or h.get("_id"),
            title = src.get("title","(untitled)"),
            abstract = src.get("abstract"),  # your docs don’t have this; kept for schema compatibility
            authors = src.get("authors"),
            tags = src.get("tags"),
            notebook_url = src.get("notebook_url") or src.get("url"),
            score = float(h.get("_score", 0.0)),
            highlights = highlights[:3] if highlights else None,
        ))

    total = 0
    total_obj = res.get("hits", {}).get("total")
    if isinstance(total_obj, dict):
        total = int(total_obj.get("value", 0))
    elif isinstance(total_obj, int):
        total = total_obj

    return SearchResponse(
        query=q,
        rewritten_query=(query_text if use_llm else None),
        total=total,
        hits=hits
    )
