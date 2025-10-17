import os, requests, json, re
from typing import List, Optional, Tuple
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from opensearchpy import OpenSearch, TransportError

# --- Local embedding (author's model) ---
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ------------------ Config ------------------
OS_HOST = os.getenv("OS_HOST", "opensearch-node1")  # match compose service name
OS_PORT = int(os.getenv("OS_PORT", "9200"))
OS_USER = os.getenv("OS_USER")
OS_PASS = os.getenv("OS_PASS")
INDEX = os.getenv("OS_INDEX") or os.getenv("INDEX") or "iguide_platform"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))  # keep in sync with mapping
DEVICE = "cpu"  # set to "cuda" if you have a GPU

LINK_TEMPLATE = os.getenv("LINK_TEMPLATE", "https://platform.i-guide.io/{element_type}/{doc_id}")

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()  # "anvilgpt" or "ollama"

ANVILGPT_BASE = os.getenv("ANVILGPT_BASE", "https://anvilgpt.rcac.purdue.edu/api").rstrip("/")
ANVILGPT_MODEL = os.getenv("ANVILGPT_MODEL", "llama3.2:latest")
ANVILGPT_API_KEY = os.getenv("ANVILGPT_API_KEY")
ANVILGPT_VERIFY = os.getenv("ANVILGPT_VERIFY", "true").lower() != "false"  # set to false only if you must



# ------------------ App init ------------------
app = FastAPI(title="I-GUIDE Notebook Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# OpenSearch client (security enabled => HTTPS)
auth = (OS_USER, OS_PASS) if OS_USER and OS_PASS else None
os_client = OpenSearch(
    hosts=[{"host": OS_HOST, "port": OS_PORT}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=False,       # self-signed in local compose
    ssl_show_warn=False,
    timeout=30,
)

# ---- Embedding init ----
_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(DEVICE)
_model.eval()

# ------------------ Helpers ------------------
def _pluralize_element_type(et: str) -> str:
    if not et:
        return ""
    et = et.strip().lower()
    # notebook -> notebooks, dataset -> datasets, code stays "code"
    return "code" if et == "code" else f"{et}s"

def _synthesize_link(src: dict, doc_id: Optional[str]) -> Optional[str]:
    url = src.get("notebook_url") or src.get("url")
    if url:
        return url
    element_type = src.get("resource-type") or src.get("element_type") or ""
    et_pl = _pluralize_element_type(element_type)
    if et_pl and doc_id:
        return LINK_TEMPLATE.format(element_type=et_pl, doc_id=doc_id)
    return None

def _pick_thumbnail(src: dict) -> Optional[str]:
    return src.get("thumbnail-image") or src.get("thumbnail_image") or src.get("thumbnail")

# ------------------ Embedding ------------------
def embed_query(text: str) -> List[float]:
    """Mean-pool over non-pad tokens and L2-normalize (ST-style)."""
    if not text or not text.strip():
        return [0.0] * EMBED_DIM

    enc = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=False)
    input_ids = enc["input_ids"].to(DEVICE)
    attn_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        last = _model(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state  # [1,T,EMBED_DIM]

    mask = attn_mask.unsqueeze(-1).type_as(last)  # [1,T,1]
    summed = (last * mask).sum(dim=1)             # [1,EMBED_DIM]
    counts = mask.sum(dim=1).clamp(min=1.0)       # [1,1]
    emb = summed / counts                         # [1,EMBED_DIM]
    emb = F.normalize(emb, p=2, dim=1)
    return emb[0].cpu().tolist()

# ------------------ Schemas ------------------
class SearchHit(BaseModel):
    id: str
    title: str
    abstract: Optional[str] = None
    authors: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    notebook_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    score: float
    highlights: Optional[List[str]] = None

class SearchResponse(BaseModel):
    query: str
    rewritten_query: Optional[str] = None
    total: int
    hits: List[SearchHit]
    mode: Optional[str] = None  # "retriever" or "bm25"
    summary: Optional[str] = None
    summary_error: Optional[str] = None
    raw_summary: Optional[str] = None

# ------------------ LLM rewrite ------------------
def rewrite_query_with_llm(q: str) -> str:
    messages = [
        {"role": "system", "content": "You rewrite user input as a concise search query for research notebooks."},
        {"role": "user", "content": q},
    ]
    resp = _call_llm_chat(messages, temperature=0.0, max_tokens=64, stream=False)
    if "text" in resp and resp["text"].strip():
        return resp["text"].strip()
    return q


def _call_llm_chat(messages, *, temperature=0.0, max_tokens=512, stream=False) -> dict:
    """
    Unified LLM caller. If LLM_BACKEND=anvilgpt, call its OpenAI-compatible
    /api/chat/completions endpoint. Otherwise fall back to Ollama /api/chat.
    Returns a dict with either {"text": "..."} or {"error": "...", "raw": "..."}.
    """
    try:
        if LLM_BACKEND == "anvilgpt":
            if not ANVILGPT_API_KEY:
                return {"error": "ANVILGPT_API_KEY missing"}
            url = f"{ANVILGPT_BASE}/chat/completions"
            headers = {
                "Authorization": f"Bearer {ANVILGPT_API_KEY}",
                "Content-Type": "application/json",
            }
            body = {
                "model": ANVILGPT_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            }
            r = requests.post(url, headers=headers, json=body, timeout=60, verify=ANVILGPT_VERIFY)
            if not r.ok:
                return {"error": f"HTTP {r.status_code}: {r.text}", "raw": r.text}
            data = r.json()
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return {"text": text}

        # ---- Ollama fallback ----
        url = f"{OLLAMA_HOST}/api/chat"
        body = {
            "model": "qwen2.5:3b-instruct",
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": stream,
        }
        r = requests.post(url, json=body, timeout=60)
        if not r.ok:
            return {"error": f"HTTP {r.status_code}: {r.text}", "raw": r.text}
        data = r.json()
        text = (data.get("message") or {}).get("content", "")
        return {"text": text}
    except Exception as e:
        return {"error": str(e)}


def generate_summary_with_llm(q: str, top_titles: Optional[List[str]] = None, debug_raw: bool = False) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Generate a concise multi-paragraph summary for the query and include short mentions of top matching notebooks.
    Returns (summary, error, raw_summary).
    """
    titles_text = ""
    if top_titles:
        def fmt(t):
            if isinstance(t, (list, tuple)) and len(t) >= 2 and t[1]:
                return f"- [{t[0]}]({t[1]})"
            return f"- {t}"
        titles_text = "\n\nRepresentative notebooks:\n\n" + "\n\n".join(fmt(t) for t in top_titles[:6])

    prompt = (
        f"Provide a concise Markdown summary (2-5 short paragraphs) describing how to find or work with '{q}' in the context of research notebooks."
        " Use normal English spacing and punctuation. Do NOT insert spaces inside words or inside multi-digit numbers (for example, write 'Imagery' not 'Imag ery' and write '1990' not '1 9 9 0')."
        " Do NOT add spaces before or inside domain names or URLs (for example, write 'i-guide.io' not 'i -guide .io')."
        " Mention common methods, data sources, and tools a researcher would try, in clear short paragraphs."
        " Then include a 'Representative notebooks' bullet list (3-5 items) using only the exact titles and URLs provided below — do NOT invent new titles, dataset names, or external URLs."
        " Ensure there is a blank line between each bullet in the 'Representative notebooks' list (i.e., separate bullets with one empty line)."
        f"{titles_text}\n\nOutput valid Markdown only and keep the summary under ~300 words."
    )

    # light, safe cleanup to avoid weird spacing/line-break artifacts
    def reflow_safe(t: str) -> str:
        t0 = t.strip()
        parts = re.split(r"\n{2,}", t0)
        word_counts = [len(p.split()) for p in parts if p.strip()]
        avg = (sum(word_counts) / len(word_counts)) if word_counts else 999

        if avg < 6:
            s = re.sub(r"\s+", " ", t0)
            sentences = re.split(r"(?<=[.!?])\s+", s)
            para_chunks = [" ".join(sentences[i:i+2]) for i in range(0, len(sentences), 2)]
            out = "\n\n".join(para_chunks).strip()
            out = re.sub(r"\s+([,\.:;\)\]])", r"\1", out)
            out = re.sub(r"([\(\[\{])\s+", r"\1", out)
            out = re.sub(r"(?<=\w)\s*\.\s*(?=\w)", ".", out)
            return out

        s = re.sub(r"\n{3,}", "\n\n", t0)
        s = re.sub(r"[ \t]{2,}", " ", s)
        s = re.sub(r"(\w)\n(\w)", r"\1 \2", s)
        s = re.sub(r"(?:(?<=\D)|^)(\d(?:[ \n]\d){3,})(?=\D|$)", lambda m: re.sub(r"[ \n]", "", m.group(1)), s)
        s = re.sub(r"\s*-\s*", "-", s)
        s = re.sub(r"\s+([,\.:;\)\]])", r"\1", s)
        s = re.sub(r"([\(\[\{])\s+", r"\1", s)
        s = re.sub(r"(?<=\w)\s*\.\s*(?=\w)", ".", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        s = re.sub(r"[ \t]{2,}", " ", s)
        return s.strip()

    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that writes concise, well-formatted Markdown for researchers."},
            {"role": "user", "content": prompt}
        ]
        resp_obj = _call_llm_chat(messages, temperature=0.0, max_tokens=512, stream=False)
    except Exception as e:
        return None, str(e), None

    if "error" in resp_obj:
        return None, resp_obj["error"], None

    raw_text = resp_obj.get("text", "") or ""
    out = reflow_safe(raw_text) if raw_text else ""
    return (out or None), None, (raw_text if debug_raw else None)


# ------------------ Query builders ------------------
def _lexical_query(q: str) -> dict:
    mm = {
        "multi_match": {
            "query": q,
            "fields": ["title^3", "contents^1.5", "tags^1.2", "authors^1.2"],
            "type": "best_fields",
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
            "fields": {"contents": {}, "title": {}, "tags": {}, "authors": {}},
        },
    }

def build_rrf_retriever_body_with_client_knn(q: str, k: int) -> dict:
    lexical = _lexical_query(q)
    qvec = embed_query(q)
    num_cand = min(10000, max(k * 5, 100))  # keep within OS limits

    knn = {
        "field": "contents-embedding",
        "query_vector": qvec,
        "k": k,
        "num_candidates": num_cand,
    }

    return {
        "size": k,
        "retriever": {
            "rrf": {
                "retrievers": [
                    {"standard": {"query": lexical}},
                    {"knn": {"query": knn}},
                ]
            }
        },
        "highlight": {
            "pre_tags": ["<mark>"], "post_tags": ["</mark>"],
            "fields": {"contents": {}, "title": {}, "tags": {}, "authors": {}},
        },
    }

def _fallback_worthy(exc: Exception) -> bool:
    s = (str(exc) or "").lower()
    return ("parsing_exception" in s) or ("retriever" in s) or ("unknown key" in s)

def search_with_fallback(index: str, primary_body: dict, bm25_body: dict) -> Tuple[dict, str]:
    try:
        res = os_client.search(index=index, body=primary_body)
        return res, "retriever"
    except TransportError as e:
        if _fallback_worthy(e):
            res = os_client.search(index=index, body=bm25_body)
            return res, "bm25"
        raise
    except Exception as e:
        if _fallback_worthy(e):
            res = os_client.search(index=index, body=bm25_body)
            return res, "bm25"
        raise

# ------------------ Routes ------------------
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
    debug_raw_summary: bool = Query(False, description="Return raw LLM generator output as raw_summary for debugging"),
):
    query_text = rewrite_query_with_llm(q) if use_llm else q

    # --- Build BM25 body ---
    body_bm25 = build_bm25_body(query_text, k)

    # --- Build kNN body (client-side embedding) ---
    try:
        qvec = embed_query(query_text)
    except Exception:
        qvec = None

    body_knn = {
        "size": k,
        "query": {"knn": {"contents-embedding": {"vector": qvec or [0.0]*EMBED_DIM, "k": k}}},
        "_source": True,
        "highlight": {
            "pre_tags": ["<mark>"], "post_tags": ["</mark>"],
            "fields": {"contents": {}, "title": {}, "tags": {}, "authors": {}},
        },
    }

    mode = "hybrid-rrf-client"
    try:
        bm25_res = os_client.search(index=INDEX, body=body_bm25)
        knn_res  = os_client.search(index=INDEX, body=body_knn) if qvec is not None else {"hits": {"hits": []}}

        # --- RRF fuse ---
        def rank_map(hits): return {h["_id"]: i for i, h in enumerate(hits)}
        b_hits = bm25_res.get("hits", {}).get("hits", []) or []
        k_hits = knn_res.get("hits", {}).get("hits", []) or []
        b_rank, k_rank = rank_map(b_hits), rank_map(k_hits)

        ids = set(b_rank) | set(k_rank)
        k_param = 60
        fused = []
        for _id in ids:
            score = 0.0
            if _id in b_rank: score += 1.0 / (k_param + b_rank[_id] + 1)
            if _id in k_rank: score += 1.0 / (k_param + k_rank[_id] + 1)

            # choose representative source & merge highlights
            src = None
            hl = {}

            if _id in b_rank:
                hb = b_hits[b_rank[_id]]
                src = hb.get("_source", src)
                for kf, arr in (hb.get("highlight") or {}).items():
                    hl.setdefault(kf, []).extend(arr)

            if _id in k_rank:
                hk = k_hits[k_rank[_id]]
                if src is None:
                    src = hk.get("_source")
                for kf, arr in (hk.get("highlight") or {}).items():
                    hl.setdefault(kf, []).extend(arr)

            fused.append({"_id": _id, "_source": src, "_score": score, "highlight": hl})

        fused.sort(key=lambda x: x["_score"], reverse=True)
        res_hits = fused[:k]

        # union-ish total (approx)
        total = max(
            bm25_res.get("hits", {}).get("total", {}).get("value", 0) if isinstance(bm25_res.get("hits", {}).get("total"), dict) else 0,
            knn_res.get("hits", {}).get("total", {}).get("value", 0) if isinstance(knn_res.get("hits", {}).get("total"), dict) else 0
        )

    except Exception:
        # Absolute fallback to BM25 only
        bm25_only = os_client.search(index=INDEX, body=body_bm25)
        res_hits = bm25_only.get("hits", {}).get("hits", [])
        total = bm25_only.get("hits", {}).get("total", {}).get("value", 0) if isinstance(bm25_only.get("hits", {}).get("total"), dict) else 0
        mode = "bm25"

    # --- Build response hits ---
    hits: List[SearchHit] = []
    for h in res_hits:
        src = h.get("_source", {}) or {}
        doc_id = src.get("id") or h.get("_id")

        # highlights (<=3)
        highlights = []
        for v in (h.get("highlight") or {}).values():
            highlights.extend(v)
        if highlights:
            highlights = highlights[:3]

        url = _synthesize_link(src, doc_id)
        thumb = _pick_thumbnail(src)

        hits.append(SearchHit(
            id=doc_id,
            title=src.get("title", "(untitled)"),
            abstract=src.get("abstract"),
            authors=src.get("authors"),
            tags=src.get("tags"),
            notebook_url=url,
            thumbnail_url=thumb,
            score=float(h.get("_score", 0.0)),
            highlights=highlights or None,
        ))

    # Optionally generate a short summary with the LLM when requested
    summary: Optional[str] = None
    if use_llm:
        try:
            # pass title + notebook_url when available so LLM uses only these exact items
            top_titles = []
            for h in hits[:6]:
                t = h.title
                u = h.notebook_url if getattr(h, "notebook_url", None) else None
                if t:
                    top_titles.append((t, u) if u else t)
            summary, summary_err, raw = generate_summary_with_llm(query_text, top_titles, debug_raw=debug_raw_summary)
        except Exception:
            summary = None
            summary_err = "exception"
    else:
        summary_err = None
    
    return SearchResponse(
        query=q,
        rewritten_query=(query_text if use_llm else None),
        total=total,
        hits=hits,
        mode=mode,
        summary=summary,
        summary_error=summary_err,
        raw_summary=(raw if debug_raw_summary else None),
    )




# --- add these builders ---
def build_knn_body(qvec: list[float], k: int) -> dict:
    return {
        "size": k,
        "query": {
            "knn": {
                "contents-embedding": {
                    "vector": qvec,
                    "k": k
                }
            }
        },
        "_source": True,
        "highlight": {
            "pre_tags": ["<mark>"], "post_tags": ["</mark>"],
            "fields": {"contents": {}, "title": {}, "tags": {}, "authors": {}},
        },
    }

def fetch_bm25(index: str, q: str, k: int) -> dict:
    body = build_bm25_body(q, k)
    return os_client.search(index=index, body=body)

def fetch_knn(index: str, q: str, k: int) -> dict:
    qvec = embed_query(q)
    body = build_knn_body(qvec, k)
    return os_client.search(index=index, body=body)

def rrf_fuse(bm25_res: dict, knn_res: dict, k: int, k_param: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion on two ranked lists.
    score = sum( 1 / (k_param + rank) )
    Returns top-k fused hits with merged highlights and a fused score.
    """
    def rank_map(hits):
        return {h["_id"]: i for i, h in enumerate(hits)}

    b_hits = bm25_res.get("hits", {}).get("hits", []) or []
    k_hits = knn_res.get("hits", {}).get("hits", []) or []

    b_rank = rank_map(b_hits)
    k_rank = rank_map(k_hits)

    ids = set(b_rank) | set(k_rank)
    fused = []
    for _id in ids:
        r = 1.0 / (k_param + b_rank[_id] + 1) if _id in b_rank else 0.0
        r += 1.0 / (k_param + k_rank[_id] + 1) if _id in k_rank else 0.0

        # merge representative hit pieces
        src = None; hl = {}
        # prefer whichever list has the better (lower) rank
        if _id in b_rank:
            h = b_hits[b_rank[_id]]
            src = h.get("_source", src)
            hl.update(h.get("highlight") or {})
        if _id in k_rank:
            h = k_hits[k_rank[_id]]
            if src is None: src = h.get("_source")
            for key, arr in (h.get("highlight") or {}).items():
                hl.setdefault(key, []).extend(arr)

        fused.append({"_id": _id, "_source": src, "_score": r, "highlight": hl})

    fused.sort(key=lambda x: x["_score"], reverse=True)
    return fused[:k]
