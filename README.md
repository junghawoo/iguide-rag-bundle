
# I-GUIDE RAG Search Bundle (OpenSearch + API + UI + Ollama)

This bundle spins up:
- OpenSearch (w/ security disabled for local dev)
- OpenSearch Dashboards
- Ollama (local LLM for optional query rewrite)
- FastAPI search service (hybrid BM25 + KNN via RRF)
- Streamlit UI

## Preparing environment
Move heavy data to the attached disk (recommended)

Do two things:

- Bind-mount big-data services (Ollama, OpenSearch) to directories on your attached disk.
(Optional but best) move Docker’s data-root to the attached disk so all images/layers live there.

- Bind-mount Ollama and OpenSearch data
Pick a mount path on your attached disk. I’ll use /media/volume/smartsearch-vol as an example—replace with your real path.
```bash
sudo mkdir -p /media/volume/smartsearch-vol/ollama
sudo chown -R 1000:1000 /media/volume/smartsearch-vol/ollama

sudo mkdir -p /media/volume/smartsearch-vol/opensearch-data
sudo chown -R 1000:1000 /media/volume/smartsearch-vol/opensearch-data

sudo mkdir -p /media/volume/smartsearch-vol/dashboards-data
sudo chown -R 1000:1000 /media/volume/smartsearch-vol/dashboards-data

```
- In docker-compose.yaml, mount 
```yaml 
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      # use a bind mount on the attached disk instead of a named volume
      - /media/volume/smartsearch-vol/ollama:/root/.ollama
```

## Quick start
create .env similar to this:
```bash 

# Copy this file to .env and adjust as needed.
# OpenSearch
OS_HOST=opensearch
OS_PORT=9200
OS_INDEX=iguide_platform

OPENSEARCH_INITIAL_ADMIN_PASSWORD=Please replace this

# Ollama
OLLAMA_HOST=http://ollama:11434

# Streamlit UI talks to API through docker network
SEARCH_API=http://api:8088
```



```bash
# 1) Copy env template
# For security reasons, .env.example and .env are excluded from this repo.
#cp .env.example .env

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


## Loading embeddings and their persistence 

### Loading prebuilt embeddings 
Load the JSONL (uses your existing index name in .env, defaults to iguide_platform)
```bash 
docker compose run --rm prebuilt_ingest \
  python load_jsonl.py --jsonl /data/RagDB/i_guide_spatial_embedding_export.jsonl

```
The loaded embeddings in OpenSearch are not deleted by docker compose down.
They live inside OpenSearch’s persistent data volume, not in a transient container.

Our docker-compose.yml has this:
```yaml
volumes:
  - /media/volume/smartsearch-vol/opensearch-data/node1:/usr/share/opensearch/data
```
then the embeddings are actually stored on the attached disk (/media/volume/smartsearch-vol/...), not in the ephemeral container filesystem.

That means:

docker compose down → stops and removes containers ✅

it does not delete the mounted volume directory 🟢

only docker compose down -v would remove named volumes (and even then, bind mounts like yours are safe)


| Command                                                           | Embeddings Lost?                      | Explanation                                           |
| ----------------------------------------------------------------- | ------------------------------------- | ----------------------------------------------------- |
| `docker compose down`                                             | ❌ No                                  | Stops/removes containers but keeps your mounted data. |
| `docker compose restart`                                          | ❌ No                                  | Just restarts.                                        |
| `docker compose down -v`                                          | ⚠️ Only if you used **named** volumes | You’re using **bind mounts**, so still safe.          |
| `sudo rm -rf /media/volume/smartsearch-vol/opensearch-data/node1` | 🔴 Yes                                | That’s where your index files are physically stored.  |


## Build & Run 
- Rebuild/bring up:
```bash 
docker compose build
docker compose up -d
```

- Rebuild api 
```bash
docker compose down
docker compose build --no-cache api
docker compose up -d
```



## Notes

- This compose disables OpenSearch security for simplicity. For staging/prod, enable security and TLS,
  set `OS_USER/OS_PASS`, and flip `use_ssl=True` in the API.
- The UI toggles LLM-based query rewrites. It's optional and off by default.
- To chunk notebooks later, create a `*_chunks` index and point the API to it; group results by notebook in the UI.
