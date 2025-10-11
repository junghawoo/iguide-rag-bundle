import os, requests
from typing import List, Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from opensearchpy import OpenSearch, TransportError

OS_HOST = os.getenv("OS_HOST", "localhost")
OS_PORT = int(os.getenv("OS_PORT", "9200"))
OS_USER = os.getenv("OS_USER")
OS_PASS = os.getenv("OS_PASS")
# Accept either OS_INDEX or INDEX for flexibility
INDEX = os.getenv("OS_INDEX") or os.getenv("INDEX") or "iguide_platform"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

app = FastAPI(title="I-GUIDE Notebook Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security is disabled in this compose; if you enable it, pass http_auth.
auth = (OS_USER, OS_PASS) if OS_USER and OS_PASS else None
os_client = OpenSearch(
    hosts=[{"host": OS_HOST, "port": OS_PORT}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=False,       # self-signed in local compose
    ssl_show_warn=False,      # suppress urllib3 InsecureRequestWarning noise
    timeout=30
)

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

def build_bm25_body(q: str, k: int) -> dict:
    """
    Plain BM25 query (widely supported). No 'retriever' block.
    """
    return {
        "size": k,
        "query": {
            "multi_match": {
                "query": q,
                "fields": ["title^3", "abstract^2", "contents", "tags", "authors"],
                "type": "best_fields"
            }
        },
        "highlight": {
            "pre_tags": ["<mark>"], "post_tags": ["</mark>"],
            "fields": {"contents": {}, "abstract": {}, "title": {}}
        }
    }

def build_rrf_retriever_body(q: str, k: int) -> dict:
    """
    Your original (newer) hybrid RRF request using 'retriever'.
    This will 400 on older clusters/plugins, so we call it first
    and fall back to BM25 on parsing_exception.
    """
    lexical = {
        "multi_match": {
            "query": q,
            "fields": ["title^3", "abstract^2", "contents", "tags", "authors"],
            "type": "best_fields"
        }
    }

    # If your cluster doesn't support on-the-fly text_embedding or this model id,
    # this entire 'retriever' body will raise parsing_exception, and we'll fallback.
    knn = {
        "field": "spatial-embedding",
        "query_vector_builder": {
            "text_embedding": {
                "model_id": "sentence-transformers/all-MiniLM-L6-v2",
                "model_text": q
            }
        },
        "k": k,
        "num_candidates": max(k * 5, 100)
    }

    return {
        "size": k,
        "retriever": {
            "rrf": {
                "retrievers": [
                    {"standard": {"query": {"bool": {"should": [lexical]}}}},
                    {"knn": {"query": knn}}
                ]
            }
        },
        "highlight": {
            "pre_tags": ["<mark>"], "post_tags": ["</mark>"],
            "fields": {"contents": {}, "abstract": {}, "title": {}}
        }
    }

def _fallback_worthy(exc: Exception) -> bool:
    """
    Identify retriever-related parsing errors to trigger BM25 fallback.
    """
    s = str(exc) if exc else ""
    s_lower = s.lower()
    # Examples seen: 'parsing_exception', 'Unknown key for a START_OBJECT in [retriever]'
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

    # Build both bodies
    body_primary = build_rrf_retriever_body(query_text, k)   # may fail on older OpenSearch
    body_bm25    = build_bm25_body(query_text, k)            # safe fallback

    # Execute with automatic fallback
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
            abstract = src.get("abstract"),
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
