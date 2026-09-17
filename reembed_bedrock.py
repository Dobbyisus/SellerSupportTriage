#!/usr/bin/env python3
"""Re-embed chunks.jsonl with a Bedrock embedding model.

WHY. The chunks were embedded locally with bge-small (384 dims) to build and
filter the corpus. At runtime the seller's ticket must be embedded with the SAME
model as the chunks, or the vectors are not comparable and retrieval returns
nonsense. Doing that in Lambda means shipping a ~130 MB ONNX model and paying
cold start for it. Embedding through Bedrock instead keeps Lambda small, puts
both sides of the comparison in one service, and lets OpenSearch call Bedrock
directly through a connector.

WHAT IT CHANGES. Three fields per chunk, all derived from the embedding:
  embedding, answers_questions, best_question, best_similarity
Everything else -- text, heading_path, section_id, rules, source_url -- is
untouched. The 45-document SELECTION is not re-run: it was a curation step that
a person validated (12 documents read start to finish), not a runtime component,
and re-running it could change the corpus and invalidate published numbers.

SETUP
  pip install boto3
  aws configure                 # credentials -- do NOT put keys in env vars
                                # or in the repo
  Bedrock console -> Model access -> request the embedding model, per region.

  Optional environment variables:
    AWS_REGION            default us-east-1
    BEDROCK_EMBED_MODEL   default amazon.titan-embed-text-v2:0
    BEDROCK_EMBED_DIM     default 1024   (Titan v2 accepts 256 / 512 / 1024)

USAGE
  python reembed_bedrock.py --dry-run     # no API calls; sizes and cost estimate
  python reembed_bedrock.py               # re-embed, backing up the old file
  python reembed_bedrock.py --verify      # compare retrieval against the local run
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

CHUNKS = Path("chunks.jsonl")
QUERIES = Path("filter_queries.tsv")
BACKUP = Path("chunks.local-bge.jsonl")

REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
MODEL = os.environ.get("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
DIM = int(os.environ.get("BEDROCK_EMBED_DIM", "1024"))

ANSWER_SIM = 0.72        # kept from the local run so the field means the same thing
MAX_RETRIES = 6
COHERE_BATCH = 96


def load_chunks():
    return [json.loads(l) for l in CHUNKS.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_queries():
    rows = []
    for line in QUERIES.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith(">"):
            continue
        p = line.split("\t")
        if len(p) >= 2 and p[1].strip():
            rows.append((p[0].strip(), p[1].strip()))
    return rows


def _call(client, body: dict):
    """Invoke with exponential backoff. Bedrock throttles hard on burst."""
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            r = client.invoke_model(modelId=MODEL, body=json.dumps(body),
                                    accept="application/json",
                                    contentType="application/json")
            return json.loads(r["body"].read())
        except Exception as e:                       # noqa: BLE001
            name = type(e).__name__
            last = e
            if "Throttling" in name or "TooManyRequests" in str(e) or "Throttl" in str(e):
                wait = (2 ** attempt) + random.random()
                print("    throttled, retrying in %.1fs" % wait)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("giving up after %d retries: %s" % (MAX_RETRIES, last))


def embed_titan(client, texts, label):
    out = []
    for i, t in enumerate(texts, 1):
        body = {"inputText": t, "dimensions": DIM, "normalize": True}
        out.append(_call(client, body)["embedding"])
        if i % 50 == 0 or i == len(texts):
            print("  %s %d/%d" % (label, i, len(texts)))
    return np.array(out, dtype=np.float32)


def embed_cohere(client, texts, label, input_type):
    out = []
    for s in range(0, len(texts), COHERE_BATCH):
        batch = texts[s:s + COHERE_BATCH]
        body = {"texts": batch, "input_type": input_type, "truncate": "END"}
        out.extend(_call(client, body)["embeddings"])
        print("  %s %d/%d" % (label, min(s + COHERE_BATCH, len(texts)), len(texts)))
    return np.array(out, dtype=np.float32)


def embed(client, texts, label, is_query):
    """Cohere distinguishes query from document; Titan does not."""
    if MODEL.startswith("cohere."):
        return embed_cohere(client, texts, label,
                            "search_query" if is_query else "search_document")
    return embed_titan(client, texts, label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="no API calls: report sizes, token estimate, and the "
                         "mapping dimension you will need")
    ap.add_argument("--verify", action="store_true",
                    help="after embedding, compare per-ticket retrieval with the "
                         "local bge run")
    a = ap.parse_args()

    chunks = load_chunks()
    queries = load_queries()
    qtext = [q for _, q in queries]
    texts = [c["text"] for c in chunks]
    words = sum(len(t.split()) for t in texts)

    print("chunks   : %d" % len(chunks))
    print("queries  : %d" % len(qtext))
    print("region   : %s" % REGION)
    print("model    : %s" % MODEL)
    print("dims     : %d  <-- use this in the OpenSearch knn_vector mapping" % DIM)
    print("words    : %s  (~%s tokens, rough)" % (format(words, ","),
                                                  format(int(words * 1.35), ",")))
    print("\nEmbedding is priced per token and this corpus is small; at any current\n"
          "Bedrock embedding price the whole run is a fraction of a cent. Check the\n"
          "pricing page for the exact figure rather than trusting this note.")

    if a.dry_run:
        print("\n--dry-run: no API calls made. Nothing written.")
        return

    try:
        import boto3
    except ImportError:
        sys.exit("boto3 not installed.  pip install boto3")

    client = boto3.client("bedrock-runtime", region_name=REGION)

    print("\nEmbedding queries ...")
    Q = embed(client, qtext, "queries", is_query=True)
    print("Embedding chunks ...")
    C = embed(client, texts, "chunks", is_query=False)

    if C.shape[1] != DIM:
        print("NOTE: model returned %d dims, not the %d requested. Use %d in the "
              "mapping." % (C.shape[1], DIM, C.shape[1]))

    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
    S = C @ Q.T

    if not BACKUP.exists():
        shutil.copy2(CHUNKS, BACKUP)
        print("\nBacked up the local-bge version to %s" % BACKUP)

    old_best = [c.get("best_similarity") for c in chunks]
    for i, c in enumerate(chunks):
        row = S[i]
        hits = np.where(row >= ANSWER_SIM)[0]
        order = hits[np.argsort(-row[hits])][:6]
        c["answers_questions"] = [qtext[j] for j in order]
        c["answer_categories"] = sorted({queries[j][0] for j in order})
        b = int(np.argmax(row))
        c["best_question"] = qtext[b]
        c["best_similarity"] = round(float(row[b]), 4)
        c["embedding"] = [round(float(x), 6) for x in C[i]]
        c["embedding_model"] = MODEL
        c["embedding_dims"] = int(C.shape[1])

    with CHUNKS.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    answered = sum(1 for c in chunks if c["answers_questions"])
    covered = len({q for c in chunks for q in c["answers_questions"]})
    print("\nWrote %s (%.1f MB)" % (CHUNKS, CHUNKS.stat().st_size / 1e6))
    print("  dims                        : %d" % C.shape[1])
    print("  chunks answering >=1 ticket : %d of %d" % (answered, len(chunks)))
    print("  tickets with an answer      : %d of %d" % (covered, len(qtext)))
    print("  by doc_type                 : %s"
          % dict(Counter(c["doc_type"] for c in chunks)))

    if a.verify:
        top = S.max(axis=0)
        print("\n--verify: top-1 similarity per ticket, Bedrock vs local bge")
        print("  mean %.3f | median %.3f | min %.3f" % (top.mean(), np.median(top), top.min()))
        print("  local bge for comparison: mean 0.764 | min 0.626 (see corpus notes 16.3)")
        weak = sorted(((float(top[j]), qtext[j]) for j in range(len(qtext))))[:8]
        print("  weakest tickets now:")
        for s, q in weak:
            print("    %.3f  %s" % (s, q[:66]))
        moved = sum(1 for i, c in enumerate(chunks)
                    if old_best[i] is not None
                    and abs(c["best_similarity"] - old_best[i]) > 0.15)
        print("  chunks whose best-match score moved by >0.15: %d" % moved)
        print("\n  Thresholds are NOT comparable across models -- 0.72 meant something\n"
              "  specific to bge. Re-check ANSWER_SIM against these numbers before\n"
              "  trusting answers_questions.")


if __name__ == "__main__":
    main()
