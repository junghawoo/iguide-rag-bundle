docker compose run --rm prebuilt_ingest sh -lc '
python3 - << "PY"
import json, os, requests, sys

os_host = os.getenv("OS_HOST","opensearch-node1")
os_port = os.getenv("OS_PORT","9200")
os_user = os.getenv("OS_USER","admin")
os_pass = os.getenv("OS_PASS")
index   = os.getenv("OS_INDEX","iguide_platform")
src     = "/data/RagDB/i_guide_spatial_embedding_export.jsonl"

base = f"https://{os_host}:{os_port}"
sess = requests.Session()
sess.auth = (os_user, os_pass)
sess.verify = False  # self-signed

# sanity: ensure mapping exists
r = sess.get(f"{base}/{index}/_mapping"); r.raise_for_status()

good=bad=skip=0
with open(src) as fin:
    for i,line in enumerate(fin,1):
        line=line.strip()
        if not line: continue
        try:
            o=json.loads(line)
        except Exception as e:
            bad+=1; sys.stderr.write(f"[{i}] JSON error: {e}\\n"); continue

        vec = o.get("contents-embedding")
        if not isinstance(vec, list):
            skip+=1; continue
        try:
            vec = [float(x) for x in vec]
        except Exception as e:
            bad+=1; sys.stderr.write(f"[{i}] vector cast error: {e}\\n"); continue
        if len(vec)!=384:
            bad+=1; sys.stderr.write(f"[{i}] wrong dim {len(vec)}\\n"); continue

        o["contents-embedding"] = vec
        _id = o.pop("_id", None)
        if _id:
            r = sess.put(f"{base}/{index}/_doc/{_id}", json=o)
        else:
            r = sess.post(f"{base}/{index}/_doc", json=o)
        if r.status_code not in (200,201):
            bad+=1; sys.stderr.write(f"[{i}] index error {r.status_code}: {r.text[:200]}\\n")
        else:
            good+=1
        if good % 100 == 0:
            print(f"Indexed {good} docs...", flush=True)
print(f"RESULT: good={good} bad={bad} skipped(no vector)={skip}")
PY
'