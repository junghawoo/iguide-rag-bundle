import os
import argparse
from opensearchpy import OpenSearch, helpers
import ujson as json

def ensure_index(client: OpenSearch, index: str):
    if client.indices.exists(index=index):
        return
    client.indices.create(index=index, body={"settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}}})

def main():
    p = argparse.ArgumentParser(description="Bulk load a JSONL (with precomputed embeddings) into OpenSearch.")
    p.add_argument("--jsonl", required=True, help="Path to JSONL")
    p.add_argument("--index", default=os.getenv("OS_INDEX", "iguide_platform"))
    p.add_argument("--host", default=os.getenv("OS_HOST", "opensearch-node1"))
    p.add_argument("--port", type=int, default=int(os.getenv("OS_PORT", "9200")))
    p.add_argument("--user", default=os.getenv("OS_USER", ""))
    p.add_argument("--password", default=os.getenv("OS_PASS", ""))
    p.add_argument("--id_field", default="id")
    p.add_argument("--batch", type=int, default=1000)
    args = p.parse_args()

    # ---- IMPORTANT: HTTPS + basic auth (self-signed cert) ----
    client = OpenSearch(
        hosts=[{"host": args.host, "port": args.port}],
        http_auth=(args.user, args.password) if args.user and args.password else None,
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
        timeout=60,
    )

    ensure_index(client, args.index)

    def gen_actions():
        with open(args.jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                _id = str(doc.get(args.id_field) or "")
                yield {"_index": args.index, "_id": _id or None, "_source": doc}

    helpers.bulk(client, gen_actions(), chunk_size=args.batch, request_timeout=120)
    print(client.count(index=args.index))
