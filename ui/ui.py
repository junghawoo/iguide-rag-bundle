
import os, requests, streamlit as st

API = os.getenv("SEARCH_API", "http://localhost:8088")

st.set_page_config(page_title="I-GUIDE Notebook Search", layout="wide")
st.title("🔎 I-GUIDE Notebook Search")

q = st.text_input("Describe what you’re looking for (topic, task, dataset, method)…",
                  placeholder="e.g., flood risk mapping with SAR in the Midwest")
use_llm = st.toggle("Use LLM to rewrite my query", value=False)
k = st.slider("Results", 5, 30, 10)

if st.button("Search") and q.strip():
    with st.spinner("Searching…"):
        try:
            r = requests.get(f"{API}/search", params={"q": q, "use_llm": str(use_llm).lower(), "k": k}, timeout=60)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

        if data.get("rewritten_query"):
            st.caption(f"LLM rewritten query: **{data['rewritten_query']}**")
        total = data.get("total", 0)
        hits = data.get("hits", [])
        st.subheader(f"Top {len(hits)} of {total} results")
        for h in hits:
            with st.container(border=True):
                col1, col2 = st.columns([4,1])
                with col1:
                    st.markdown(f"### {h.get('title','(untitled)')}")
                    abstract = h.get("abstract")
                    if abstract:
                        st.write(abstract[:400] + ("…" if len(abstract)>400 else ""))
                    highlights = h.get("highlights") or []
                    if highlights:
                        st.markdown("**Matches:** " + " … ".join(highlights))
                    authors = h.get("authors") or []
                    tags = h.get("tags") or []
                    if authors:
                        st.caption("Authors: " + ", ".join(authors))
                    if tags:
                        st.caption("Tags: " + ", ".join(tags))
                    url = h.get("notebook_url")
                    if url:
                        st.link_button("Launch on I-GUIDE", url)
                with col2:
                    st.metric("Relevance", f"{h.get('score', 0):.2f}")
