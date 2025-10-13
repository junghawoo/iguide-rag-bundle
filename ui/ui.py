import os
import requests
import streamlit as st
import io

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
    thumb = h.get("thumbnail_url")   # <-- NEW

    with st.container(border=True):
        col1, col2 = st.columns([4, 1])

        with col1:
            if url:
                st.markdown(f"### [{title}]({url})")
            else:
                st.markdown(f"### {title}")

            if abstract:
                st.write(abstract[:500] + ("…" if len(abstract) > 500 else ""))

            if highlights:
                st.markdown("**Matches:** " + " … ".join(highlights))

            if authors:
                st.caption("Authors: " + ", ".join(authors))
            if tags:
                st.caption("Tags: " + ", ".join(tags))

            if url:
                try:
                    st.link_button("🚀 Launch on I-GUIDE", url)
                except Exception:
                    st.markdown(f"[🚀 Launch on I-GUIDE]({url})")

        with col2:
            # --- NEW: thumbnail (if present)
            if thumb:
                try:
                    # Try to fetch the image bytes so we control errors/certs/timeouts
                    resp = requests.get(thumb, timeout=6)
                    resp.raise_for_status()
                    img_bytes = io.BytesIO(resp.content)
                    try:
                        # Newer Streamlit
                        st.image(img_bytes, caption="Preview", use_container_width=True)
                    except TypeError:
                        # Older Streamlit fallback
                        st.image(img_bytes, caption="Preview", use_column_width=True)
                except Exception as e:
                    # Last resort: try letting Streamlit fetch it directly
                    try:
                        st.image(thumb, caption="Preview")
                    except TypeError:
                        st.image(thumb, caption="Preview", use_column_width=True)
                    except Exception:
                        st.caption("Preview unavailable")

            # Relevance metric under the image
            try:
                st.metric("Relevance", f"{float(score):.2f}")
            except Exception:
                pass

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
