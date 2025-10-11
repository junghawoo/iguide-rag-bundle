
# Prebuilt Embeddings Loader (JSONL → OpenSearch)

This patch adds a small service to bulk-load an existing JSONL file that already contains embeddings
(e.g., `RagDB/i_guide_spatial_embedding_export.jsonl`) into your OpenSearch index.

## 1) Place your JSONL
Put the `RagDB` folder **next to** your `iguide-rag-bundle` folder, like:

```
/path/to/
  RagDB/
    i_guide_spatial_embedding_export.jsonl
  iguide-rag-bundle/
    docker-compose.yml
    ...
```

## 2) Add the service to docker-compose
Open `docker-compose.yml` in `iguide-rag-bundle/` and paste this under `services:`:

--- paste begin ---
  prebuilt_ingest:
    build: ./prebuilt_ingest
    container_name: iguide-prebuilt-ingest
    env_file: .env
    environment:
      - OS_HOST=opensearch
      - OS_PORT=9200
      - OS_INDEX=${OS_INDEX:-iguide_platform}
    # Mount the JSONL (assumes RagDB is a sibling folder)
    volumes:
      - ../RagDB:/data/RagDB:ro
    depends_on:
      - opensearch
--- paste end ---

## 3) Build the image
```bash
docker compose build prebuilt_ingest
```

## 4) Load the JSONL into OpenSearch
```bash
# Using the default path and index
docker compose run --rm prebuilt_ingest   python load_jsonl.py --jsonl /data/RagDB/i_guide_spatial_embedding_export.jsonl
```

### Advanced
- Different JSONL path:
```bash
docker compose run --rm prebuilt_ingest   python load_jsonl.py --jsonl /data/RagDB/your_file.jsonl --index your_index
```
- If your JSONL uses a different id field:
```bash
docker compose run --rm prebuilt_ingest   python load_jsonl.py --jsonl /data/RagDB/i_guide_spatial_embedding_export.jsonl --id_field notebook_id
```

> The loader **does not recompute embeddings**. It simply pushes the JSONL lines as documents.
> Ensure the vector field name and dimension in your JSONL match your index mapping.
