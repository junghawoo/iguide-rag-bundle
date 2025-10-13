import os, requests
from typing import List, Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from opensearchpy import OpenSearch, TransportError

# NEW: local embedding (same as the author)
import torch
import torch.nn.functional as F
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
# For future: automatic device selection
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

import os

# Element type will be pluralized in code; template has NO trailing "s"
LINK_TEMPLATE = os.getenv(
    "LINK_TEMPLATE",
    "https://platform.i-guide.io/{element_type}/{doc_id}"
)

def _pluralize_element_type(et: str) -> str:
    if not et:
        return ""
    et = et.strip().lower()
    # notebook -> notebooks, dataset -> datasets, code stays "code"
    return "code" if et == "code" else f"{et}s"

def _synthesize_link(src: dict, doc_id: str | None) -> str | None:
    # Prefer existing link fields if present
    url = src.get("notebook_url") or src.get("url")
    if url:
        return url
    element_type = src.get("resource-type") or src.get("element_type") or ""
    et_pl = _pluralize_element_type(element_type)
    if et_pl and doc_id:
        return LINK_TEMPLATE.format(element_type=et_pl, doc_id=doc_id)
    return None

###------------------ Embedding Function -------------------
def embed_query(text: str) -> list[float]:
    """Mean-pool over non-pad tokens and L2-normalize (ST-style).
        Masking pads: Prevents [PAD] vectors (often near zero but not guaranteed) from skewing the mean.
        Normalization: If you use cosine similarity (e.g., OpenSearch cosinesimil), normalized embeddings make scores consistent and improve nearest-neighbor behavior.
        No [CLS] pooling: ST defaults to mean pooling; outputs.pooler_output (dense over [CLS]) is not used and typically underperforms for semantic similarity.
        Truncation: ST uses max_length=512; longer inputs are truncated.
    """
    if not text or not text.strip():
        return [0.0] * 384  # or raise

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,   # single example; no need to pad
    )
    input_ids = inputs["input_ids"].to(DEVICE)           # [1, T]
    attn_mask = inputs["attention_mask"].to(DEVICE)      # [1, T]

    with torch.no_grad():
        out = _model(input_ids=input_ids, attention_mask=attn_mask)
        last_hidden = out.last_hidden_state              # [1, T, 384]

    # mean over *real* tokens only
    mask = attn_mask.unsqueeze(-1).type_as(last_hidden)  # [1, T, 1]
    summed = (last_hidden * mask).sum(dim=1)             # [1, 384]
    counts = mask.sum(dim=1).clamp(min=1.0)              # [1, 1]
    emb = summed / counts                                # [1, 384]

    # L2-normalize for cosine retrieval (recommended)
    emb = F.normalize(emb, p=2, dim=1)

    return emb[0].cpu().tolist()

# ------------------ API Schemas ---------------------
class SearchHit(BaseModel):
    id: str
    title: str
    abstract: Optional[str] = None
    authors: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    notebook_url: Optional[str] = None
    thumbnail_url: Optional[str] = None      # <-- NEW
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

    # Execute with fallback
    res = search_with_fallback(INDEX, body_primary, body_bm25)

    hits = []
    for h in res.get("hits", {}).get("hits", []):
        src = h.get("_source", {}) or {}
        doc_id = src.get("id") or h.get("_id")

        # collect up to 3 highlight snippets
        highlights = []
        for v in (h.get("highlight") or {}).values():
            highlights.extend(v)
        if highlights:
            highlights = highlights[:3]

        url = _synthesize_link(src, doc_id)  # <-- always compute

        # --- NEW: pick up thumbnail from common keys
        thumb = (src.get("thumbnail-image")
                or src.get("thumbnail_image")
                or src.get("thumbnail"))

        hits.append(SearchHit(
            id = doc_id,
            title = src.get("title","(untitled)"),
            abstract = src.get("abstract"),
            authors = src.get("authors"),
            tags = src.get("tags"),
            notebook_url = url,
            thumbnail_url = thumb,            # <-- NEW
            score = float(h.get("_score", 0.0)),
            highlights = highlights or None,
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
