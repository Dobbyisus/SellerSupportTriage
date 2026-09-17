"""Hybrid retrieval: BM25 + kNN, combined by the normalization pipeline.

    python opensearch/search.py "my payment is on hold and I don't know why"
    python opensearch/search.py --category payments "when do I get paid"

Returns ranked passages with the metadata needed to render a citation, plus a
confidence reading the agent uses to decide whether to re-query.

TWO SCORES, ON PURPOSE
----------------------
The hybrid score decides the ORDER. It cannot decide CONFIDENCE: min-max
normalization rescales the best hit of every query to ~1.0, so a ticket that
matches nothing still comes back with a top hybrid score near 1.0. Reading
confidence off that number would make retrieval look perfect on exactly the
queries where it failed.

Confidence therefore comes from a separate exact-cosine query. It uses a
painless `cosineSimilarity` script rather than the kNN score, because the kNN
score formula differs between engines (lucene, faiss, nmslib) while the script
returns the raw cosine on any of them — the same number MIN_SIM was calibrated
against. Over 710 documents an exact scan costs nothing.
"""

import argparse
import json
import sys

import config

_embedder = None


def embed_query(text: str) -> list[float]:
    """Embed a ticket with the SAME model that produced the chunk vectors.

    Getting this wrong is silent: the vectors still have matching dimensions
    and cosine still returns a number, it is just meaningless.
    """
    global _embedder

    if config.EMBED_BACKEND == "local-bge":
        # fastembed's query_embed applies bge's query prefix. semantic_filter.py
        # used query_embed for queries and embed for passages; matching that
        # exactly is what keeps the numbers comparable.
        from fastembed import TextEmbedding

        if _embedder is None:
            _embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", threads=4)
        import numpy as np

        vec = np.array(list(_embedder.query_embed([text], parallel=1))[0], dtype="float32")
        vec /= np.linalg.norm(vec) + 1e-9
        return vec.tolist()

    import boto3

    if _embedder is None:
        _embedder = boto3.client("bedrock-runtime", region_name=config.REGION)
    body = json.dumps(
        {"inputText": text, "dimensions": config.EMBED_DIM, "normalize": True}
    )
    resp = _embedder.invoke_model(modelId="amazon.titan-embed-text-v2:0", body=body)
    return json.loads(resp["body"].read())["embedding"]


def lexical_query(text: str, category: str | None) -> dict:
    """BM25 half.

    The text match sits in `must` and the boosts in `should`, so a boost can
    only reorder documents that already matched the words. Putting the boosts
    in `must`/`should` with minimum_should_match would let an agreement chunk
    surface on doc_type alone.
    """
    should: list[dict] = []

    if category and category in config.AGREEMENT_BOOST_CATEGORIES:
        should.append(
            {"term": {"doc_type": {"value": "agreement", "boost": config.AGREEMENT_BOOST}}}
        )
    if category:
        should.append(
            {"term": {"category": {"value": category, "boost": config.CATEGORY_BOOST}}}
        )

    return {
        "bool": {
            "must": [
                {
                    "multi_match": {
                        "query": text,
                        "fields": config.LEXICAL_FIELDS,
                        "type": "best_fields",
                        "tie_breaker": 0.3,
                    }
                }
            ],
            "should": should,
        }
    }


def vector_query(vec: list[float]) -> dict:
    return {"knn": {"embedding": {"vector": vec, "k": config.CANDIDATE_K}}}


def cosine_query(vec: list[float]) -> dict:
    """Exact cosine over every chunk. Engine-independent, matches MIN_SIM.

    NOTE THE SIGNATURE. OpenSearch's k-NN painless extension is

        cosineSimilarity(params.query_value, doc['field'])

    which is NOT the Elasticsearch dense_vector form,
    `cosineSimilarity(params.q, 'field')`. The two look interchangeable and are
    not: the Elasticsearch spelling fails on OpenSearch with a compile error.
    Most snippets you find online are the Elasticsearch one.

    Returns 1.0 + cosine to stay non-negative, since script_score forbids
    negative scores. search() subtracts the 1.0 back off.
    """
    return {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "1.0 + cosineSimilarity(params.query_value, doc['embedding'])",
                "params": {"query_value": vec},
            },
        }
    }


SOURCE_FIELDS = [
    "chunk_id", "doc_id", "title", "heading_path", "section_id",
    "source_url", "category", "doc_type", "text", "rules", "best_question",
]


def search(os_client, text: str, category: str | None = None, k: int | None = None) -> dict:
    k = k or config.TOP_K
    vec = embed_query(text)

    # Two plain searches rather than one _msearch. The hybrid request has to
    # carry ?search_pipeline=..., and per-request pipelines are not reliably
    # honoured in an _msearch header across versions — a silently un-normalized
    # hybrid query still returns results, just badly ranked, which is the worst
    # kind of bug to inherit on demo day. Two round trips cost ~40ms.
    body_hybrid = {
        "size": k,
        "_source": SOURCE_FIELDS,
        "query": {"hybrid": {"queries": [lexical_query(text, category), vector_query(vec)]}},
    }
    hybrid_resp = os_client.transport.perform_request(
        "POST",
        "/%s/_search?search_pipeline=%s" % (config.INDEX, config.PIPELINE),
        body=body_hybrid,
    )

    body_cos = {"size": 50, "_source": ["chunk_id"], "query": cosine_query(vec)}
    cosine_resp = os_client.search(index=config.INDEX, body=body_cos)

    # script_score returns 1.0 + cosine to stay non-negative.
    cos_by_id = {
        h["_source"]["chunk_id"]: h["_score"] - 1.0 for h in cosine_resp["hits"]["hits"]
    }
    best_cosine = max(cos_by_id.values()) if cos_by_id else 0.0

    results = []
    for hit in hybrid_resp["hits"]["hits"]:
        src = hit["_source"]
        cid = src["chunk_id"]
        results.append(
            {
                "chunk_id": cid,
                "hybrid_score": round(hit["_score"], 4),
                # None means the chunk placed outside the top 50 by pure
                # meaning — it was ranked here by the lexical half.
                "cosine": round(cos_by_id[cid], 4) if cid in cos_by_id else None,
                "title": src.get("title"),
                "heading_path": src.get("heading_path"),
                "section_id": src.get("section_id"),
                "source_url": src.get("source_url"),
                "category": src.get("category"),
                "doc_type": src.get("doc_type"),
                "rules": src.get("rules"),
                "text": src.get("text", ""),
            }
        )

    return {
        "query": text,
        "category": category,
        "results": results,
        "confidence": {
            "best_cosine": round(best_cosine, 4),
            "min_sim": config.MIN_SIM,
            # The single fact the agent branches on: below this, re-query.
            "confident": best_cosine >= config.MIN_SIM,
            "quotable": best_cosine >= config.ANSWER_SIM,
            "calibrated_for": config.CALIBRATED_FOR,
        },
    }


def citation(hit: dict) -> str:
    """Render 'Business Solutions Agreement > General Terms, S-5 <url>'."""
    label = hit.get("heading_path") or hit.get("title") or hit["chunk_id"]
    if hit.get("section_id"):
        label = "%s, %s" % (label, hit["section_id"])
    url = hit.get("source_url")
    return "%s%s" % (label, "  <%s>" % url if url else "  [no source URL]")


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid policy retrieval")
    ap.add_argument("ticket", nargs="+", help="the seller's ticket text")
    ap.add_argument("--category", help="classifier output: account|returns|shipping|payments|listings")
    ap.add_argument("-k", type=int, default=config.TOP_K)
    ap.add_argument("--json", action="store_true", help="emit raw JSON")
    args = ap.parse_args()

    import client

    text = " ".join(args.ticket)
    os_client = client.connect()
    out = search(os_client, text, args.category, args.k)

    if args.json:
        print(json.dumps(out, indent=2))
        return

    conf = out["confidence"]
    print("\nticket   : %s" % text)
    print("category : %s" % (args.category or "(none)"))
    print(
        "confidence: best cosine %.3f vs MIN_SIM %.2f -> %s"
        % (conf["best_cosine"], conf["min_sim"], "CONFIDENT" if conf["confident"] else "RE-QUERY")
    )
    print("           thresholds calibrated for %s\n" % conf["calibrated_for"])

    for i, hit in enumerate(out["results"], 1):
        cos = "  cos %.3f" % hit["cosine"] if hit["cosine"] is not None else "  cos  <top50"
        print("%d. [%.4f%s] %s" % (i, hit["hybrid_score"], cos, hit["title"]))
        print("   %s" % citation(hit))
        body = " ".join(hit["text"].split())
        print("   %s\n" % (body[:220] + ("..." if len(body) > 220 else "")))


if __name__ == "__main__":
    main()
