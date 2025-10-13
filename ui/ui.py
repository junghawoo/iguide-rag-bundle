import os
import requests
import streamlit as st

API = os.getenv("SEARCH_API", "http://localhost:8088")

st.set_page_config(page_title="I-GUIDE Notebook Search", layout="wide")
st.title("🔎 I-GUIDE Notebook Search")

# --- Query controls ---
q = st.text_input(
    "Describe what you’re looking for (topic, task, dataset, method)…",
    placeholder="e.g., flood risk mapping with SAR in the Midwest",
)
use_llm = st.toggle("Use LLM to rewrite my query", value=False)
k = st.slider("Results", 5, 30, 10)

def _truncate(text: str, n: int = 400) -> str:
    if not text:
        return ""
    return text if len(text) <= n else text[:n].rstrip() + "…"

def render_hit(h: dict):
    title = h.get("title") or "(untitled)"
    url = h.get("notebook_url")  # server now always tries to provide this
    abstract = h.get("abstract")
    highlights = h.get("highlights") or []
    authors = h.get("authors") or []
    tags = h.get("tags") or []
    score = h.get("score", 0.0)

    with st.container(border=True):
        col1, col2 = st.columns([4, 1])

        with col1:
            # Title (clickable if URL present)
            if url:
                st.markdown(f"### [{title}]({url})")
            else:
                st.markdown(f"### {title}")

            # Abstract / snippet
            if abstract:
                st.write(_truncate(abstract, 500))

            # Highlights (from BM25/combined search)
            if highlights:
                st.markdown("**Matches:** " + " … ".join(highlights))

            # Metadata
            if authors:
                st.caption("Authors: " + ", ".join(authors))
            if tags:
                st.caption("Tags: " + ", ".join(tags))

            # Launch button (Streamlit ≥1.25), fallback to markdown link
            if url:
                try:
                    st.link_button("🚀 Launch on I-GUIDE", url)
                except Exception:
                    st.markdown(f"[🚀 Launch on I-GUIDE]({url})")

        with col2:
            try:
                st.metric("Relevance", f"{float(score):.2f}")
            except Exception:
                st.write("")

if st.button("Search") and q.strip():
    with st.spinner("Searching…"):
        try:
            params = {"q": q, "use_llm": str(use_llm).lower(), "k": k}
            r = requests.get(f"{API}/search", params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

    # Show LLM rewrite when present
    rewritten = data.get("rewritten_query")
    if rewritten:
        st.caption(f"LLM-rewritten query: **{rewritten}**")

    total = data.get("total", 0)
    hits = data.get("hits", []) or []

    st.subheader(f"Top {len(hits)} of {total} results")
    for h in hits:
        render_hit(h)
