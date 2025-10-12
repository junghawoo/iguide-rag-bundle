#!/usr/bin/env python3
import os
import argparse
import sys
from typing import Dict, Iterable, Optional, Tuple
from urllib.parse import urlparse

import ujson as json
from opensearchpy import OpenSearch, helpers, RequestsHttpConnection


def parse_os_url(os_url: str) -> Tuple[str, int, bool]:
    """
    Parse --os-url into (host, port, use_ssl)
    Accepts forms like:
      https://localhost:9200
      http://opensearch-node1:9200
      localhost:9200 (we'll assume http)
      opensearch-node1 (default http:9200)
    """
    if "://" not in os_url:
        # add default scheme
        if ":" in os_url:
            host, port_s = os_url.split(":", 1)
            return host, int(port_s), False
        return os_url, 9200, False

    u = urlparse(os_url)
    scheme = (u.scheme or "http").lower()
    host = u.hostname or "localhost"
    port = u.port or (443 if scheme == "https" else 9200)
    use_ssl = scheme == "https"
    return host, port, use_ssl


def build_client(
    os_url: str,
    user: str = "",
    password: str = "",
    insecure: bool = False,
    timeout: int = 60,
) -> OpenSearch:
    host, port, use_ssl = parse_os_url(os_url)
    auth = (user, password) if user or password else None
    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=auth,
        use_ssl=use_ssl,
        verify_certs=not insecure,
        ssl_show_warn=False,
        connection_class=RequestsHttpConnection,
        timeout=timeout,
    )
    return client


def ensure_index(
    client: OpenSearch,
    index: str,
    create_mapping: bool,
    vector_field: str,
    dim: int,
    replicas: int,
):
    if client.indices.exists(index=index):
        return
    settings = {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": replicas,
                "analysis": {
                    "analyzer": {
                        "en_std": {"type": "standard", "stopwords": "_english_"}
                    }
                },
            }
        }
    }
    if create_mapping:
        settings["mappings"] = {
            "properties": {
                "title": {"type": "text", "analyzer": "en_std"},
                "text": {"type": "text", "analyzer": "en_std"},
                "url": {"type": "keyword"},
                "tags": {"type": "keyword"},
                "source": {"type": "keyword"},
                "ts": {"type": "date"},
                vector_field: {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {"name": "hnsw", "space_type": "cosinesimil"},
                },
            }
        }
    client.indices.create(index=index, body=settings)


def recreate_index(
    client: OpenSearch,
    index: str,
    vector_field: str,
    dim: int,
    replicas: int,
):
    if client.indices.exists(index=index):
        client.indices.delete(index=index)
    ensure_index(
        client, index, create_mapping=True, vector_field=vector_field, dim=dim, replicas=replicas
    )


def l2_normalize(vec):
    # Normalize to unit length (avoid divide by zero)
    import math

    s = math.fsum(v * v for v in vec)
    if s <= 0.0:
        return vec
    inv = 1.0 / (s ** 0.5)
    return [v * inv for v in vec]


def coerce_vector(x, dim: int, normalize: bool) -> Optional[list]:
    if not isinstance(x, (list, tuple)):
        return None
    try:
        v = [float(t) for t in x]
    except Exception:
        return None
    if len(v) != dim:
        return None
    if normalize:
        v = l2_normalize(v)
    return v


def parse_field_renames(pairs: Iterable[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--rename expects OLD=NEW, got: {p}")
        old, new = p.split("=", 1)
        old = old.strip()
        new = new.strip()
        if not old or not new:
            raise ValueError(f"--rename invalid pair: {p}")
        mapping[old] = new
    return mapping


def apply_renames(doc: dict, renames: Dict[str, str]) -> dict:
    if not renames:
        return doc
    for old, new in renames.items():
        if old in doc and new not in doc:
            doc[new] = doc.pop(old)
    return doc


def gen_actions(
    jsonl_path: str,
    index: str,
    id_field: str,
    vector_field: str,
    dim: int,
    normalize: bool,
    renames: Dict[str, str],
) -> Iterable[dict]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except Exception as e:
                sys.stderr.write(f"[line {lineno}] JSON error: {e}\n")
                continue

            # rename fields if requested
            if renames:
                doc = apply_renames(doc, renames)

            # coerce vector
            raw_vec = doc.get(vector_field)
            vec = coerce_vector(raw_vec, dim=dim, normalize=normalize)
            if vec is None:
                sys.stderr.write(
                    f"[line {lineno}] skip: missing/invalid '{vector_field}' "
                    f"(dim expected {dim})\n"
                )
                continue
            doc[vector_field] = vec

            # id (optional)
            _id = str(doc.get(id_field)) if id_field and doc.get(id_field) is not None else None
            yield {"_index": index, "_id": _id, "_source": doc}


def main():
    p = argparse.ArgumentParser(
        description="Bulk load a JSONL with precomputed embeddings into OpenSearch (k-NN)."
    )
    # I/O and OS connection
    p.add_argument("--jsonl", required=True, help="Path to JSONL")
    p.add_argument(
        "--os-url",
        default=os.getenv("OS_URL", ""),  # prefer explicit URL
        help="OpenSearch URL, e.g., https://localhost:9200 (overrides host/port)",
    )
    p.add_argument("--host", default=os.getenv("OS_HOST", "opensearch-node1"))
    p.add_argument("--port", type=int, default=int(os.getenv("OS_PORT", "9200")))
    p.add_argument("--user", default=os.getenv("OS_USER", ""))
    p.add_argument("--password", default=os.getenv("OS_PASS", ""))
    p.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification (useful for self-signed certs).",
    )
    # Indexing behavior
    p.add_argument("--index", default=os.getenv("OS_INDEX", "iguide_platform"))
    p.add_argument("--recreate", action="store_true", help="Delete and re-create the index.")
    p.add_argument(
        "--create-mapping",
        action="store_true",
        help="If index does not exist, create it with a basic mapping (incl. knn_vector).",
    )
    p.add_argument("--replicas", type=int, default=0, help="Index number_of_replicas (default 0).")
    p.add_argument("--batch", type=int, default=1000, help="Bulk chunk size.")
    p.add_argument("--refresh", action="store_true", help="Force refresh after ingest.")
    # Data layout
    p.add_argument("--id-field", default="_id", help="Field to use as document _id (default _id).")
    p.add_argument(
        "--vector-field",
        default="contents-embedding",
        help="Vector field name in documents (default contents-embedding).",
    )
    p.add_argument("--dim", type=int, default=384, help="Vector dimensionality (default 384).")
    p.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize vectors before indexing (recommended for cosine).",
    )
    p.add_argument(
        "--rename",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Rename doc fields before indexing (can be specified multiple times).",
    )

    args = p.parse_args()

    os_url = args.os_url.strip()
    if not os_url:
        # fall back to host/port if --os-url not provided
        scheme = "https" if args.user or args.password else "http"
        os_url = f"{scheme}://{args.host}:{args.port}"

    client = build_client(
        os_url=os_url,
        user=args.user,
        password=args.password,
        insecure=args.insecure,
        timeout=120,
    )

    # (Re)create / ensure index
    if args.recreate:
        recreate_index(
            client,
            index=args.index,
            vector_field=args.vector_field,
            dim=args.dim,
            replicas=args.replicas,
        )
        print(f"Re-created index '{args.index}'")
    else:
        ensure_index(
            client,
            index=args.index,
            create_mapping=args.create_mapping,
            vector_field=args.vector_field,
            dim=args.dim,
            replicas=args.replicas,
        )

    # Field rename map
    renames = parse_field_renames(args.rename)

    # Bulk ingest
    actions = gen_actions(
        jsonl_path=args.jsonl,
        index=args.index,
        id_field=args.id_field,
        vector_field=args.vector_field,
        dim=args.dim,
        normalize=args.normalize,
        renames=renames,
    )

    success, failed = helpers.bulk(
        client,
        actions,
        chunk_size=args.batch,
        request_timeout=300,
        raise_on_error=False,
        stats_only=True,
    )
    # force refresh if asked
    if args.refresh:
        client.indices.refresh(index=args.index)

    # final count & summary
    try:
        count = client.count(index=args.index)["count"]
    except Exception:
        count = "?"
    print(
        f"RESULT: success={success} failed={failed} index={args.index} count={count}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
