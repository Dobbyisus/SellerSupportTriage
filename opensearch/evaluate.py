"""Re-runs the 67 filter tickets and compares live retrieval with local vectors.

    python opensearch/evaluate.py --local    # baseline only, no AWS needed
    python opensearch/evaluate.py            # live index vs baseline
    python opensearch/evaluate.py --save     # also write evaluate_results.json

THREE CHECKS, IN ORDER OF SEVERITY
----------------------------------
1. VECTOR FIDELITY. The exact cosine the live index reports for a ticket must
   equal the cosine computed locally from chunks.jsonl. If it does not, the
   wrong vectors are in the index, or the mapping is wrong, and nothing below
   means anything. This is the check that catches a broken mapping.

2. USABLE-PASSAGE RATE. How many of the 67 tickets surface a passage at or
   above the floor. Hybrid should MATCH OR BEAT the pure-vector baseline. Any
   ticket that gets worse means the lexical half is dragging results down —
   look at the field weights in config.LEXICAL_FIELDS.

3. RANK RESCUES. Tickets the vector half alone ranked badly that BM25 pulled
   up. This is the evidence that hybrid was worth building — the handoff
   predicts "how do I write a plan of action" (0.626 on meaning alone) as the
   clearest case, since it is a verbatim phrase.

NOTE ON THE §16.3 NUMBERS. corpus-collection.md §16.3 measured 555 chunks
produced by semantic_filter.py's own chunking. chunks.jsonl is the production
chunking and holds 710. The baseline computed here is therefore the right
comparison and will differ slightly from §16.3 — that is expected, not a
regression.
"""

import argparse
import json
import sys
import time

import numpy as np

import config

USABLE = 0.70   # the floor §16.3 reports against
STRONG = 0.75


def load_queries() -> list[tuple[str, str]]:
    rows = []
    for line in config.QUERIES.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith(">"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].strip():
            rows.append((parts[0].strip(), parts[1].strip()))
    return rows


def load_chunks() -> list[dict]:
    return [
        json.loads(line)
        for line in config.CHUNKS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def local_baseline(queries, chunks) -> dict:
    """Pure-vector top-1 for every ticket, straight from chunks.jsonl."""
    from search import embed_query

    matrix = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9

    print("embedding %d tickets with %s ..." % (len(queries), config.EMBED_BACKEND), flush=True)
    qvecs = np.array([embed_query(q) for _, q in queries], dtype=np.float32)
    qvecs /= np.linalg.norm(qvecs, axis=1, keepdims=True) + 1e-9

    sims = qvecs @ matrix.T
    out = {}
    for i, (cat, q) in enumerate(queries):
        order = np.argsort(-sims[i])
        top = order[0]
        out[q] = {
            "category": cat,
            "best_cosine": float(sims[i][top]),
            "best_chunk": chunks[top]["chunk_id"],
            "best_title": chunks[top].get("title"),
            "top5": [chunks[j]["chunk_id"] for j in order[:5]],
            "rank_of": {chunks[j]["chunk_id"]: r for r, j in enumerate(order[:200], start=1)},
        }
    return out


def summarize(values: list[float], label: str) -> dict:
    arr = np.array(values, dtype=np.float32)
    stats = {
        "n": len(arr),
        "usable": int((arr >= USABLE).sum()),
        "strong": int((arr >= STRONG).sum()),
        "mean": float(arr.mean()) if len(arr) else 0.0,
    }
    print(
        "%-22s usable(>=%.2f) %2d/%d (%.0f%%)   strong(>=%.2f) %2d/%d   mean %.3f"
        % (label, USABLE, stats["usable"], stats["n"], 100 * stats["usable"] / max(stats["n"], 1),
           STRONG, stats["strong"], stats["n"], stats["mean"])
    )
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help="baseline only, no OpenSearch")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--category", action="store_true", help="pass each ticket's category to search")
    args = ap.parse_args()

    print(config.summary())
    stale = config.warn_if_thresholds_stale()
    if stale:
        print("\nWARNING\n" + stale + "\n")

    queries = load_queries()
    chunks = load_chunks()
    print("%d tickets, %d chunks\n" % (len(queries), len(chunks)))

    base = local_baseline(queries, chunks)
    print()
    base_stats = summarize([b["best_cosine"] for b in base.values()], "local baseline")

    if args.local:
        weak = sorted(base.items(), key=lambda kv: kv[1]["best_cosine"])[:8]
        print("\nweakest tickets on meaning alone (BM25 should rescue these):")
        for q, b in weak:
            print("   %.3f  [%-8s] %s" % (b["best_cosine"], b["category"], q))
            print("          -> %s" % b["best_title"])
        if args.save:
            out = config.ROOT / "opensearch" / "evaluate_baseline.json"
            out.write_text(json.dumps(base, indent=2, default=str), encoding="utf-8")
            print("\nwrote %s" % out)
        return

    import client
    from search import search

    os_client = client.connect()
    client.require_hybrid_support(os_client)

    live_best, hybrid_top1, fidelity, rows = [], [], [], []
    rescues, regressions = [], []

    t0 = time.time()
    for i, (cat, q) in enumerate(queries, 1):
        res = search(os_client, q, cat if args.category else None)
        b = base[q]

        live_cos = res["confidence"]["best_cosine"]
        live_best.append(live_cos)
        fidelity.append(abs(live_cos - b["best_cosine"]))

        hits = res["results"]
        top1_cos = next((h["cosine"] for h in hits if h["cosine"] is not None), 0.0)
        hybrid_top1.append(top1_cos or 0.0)

        # Did BM25 promote something the vector half buried?
        if hits:
            top_id = hits[0]["chunk_id"]
            local_rank = b["rank_of"].get(top_id)
            if local_rank and local_rank > 5 and (hits[0]["cosine"] or 0) >= USABLE:
                rescues.append((q, top_id, local_rank))
            if b["best_cosine"] >= USABLE and (top1_cos or 0) < USABLE:
                regressions.append((q, b["best_cosine"], top1_cos))

        rows.append(
            {
                "ticket": q, "category": cat,
                "local_best_cosine": round(b["best_cosine"], 4),
                "live_best_cosine": live_cos,
                "hybrid_top1_cosine": top1_cos,
                "hybrid_top1_chunk": hits[0]["chunk_id"] if hits else None,
                "local_top1_chunk": b["best_chunk"],
                "confident": res["confidence"]["confident"],
            }
        )
        if i % 10 == 0:
            print("  %d/%d ..." % (i, len(queries)), flush=True)

    elapsed = time.time() - t0

    print("\n--- 1. vector fidelity ---")
    max_err = max(fidelity)
    print("max |live cosine - local cosine| = %.6f" % max_err)
    if max_err > 0.01:
        print("FAIL — the index does not hold the vectors in chunks.jsonl.")
        print("       Re-run create_index.py --recreate then bulk_upload.py.")
    else:
        print("OK — the live index agrees with chunks.jsonl.")

    print("\n--- 2. usable-passage rate ---")
    summarize([r["local_best_cosine"] for r in rows], "local baseline")
    live_stats = summarize([r["hybrid_top1_cosine"] for r in rows], "hybrid top-1")
    delta = live_stats["usable"] - base_stats["usable"]
    print("delta: %+d tickets at rank 1 (%.1fs total, %.2fs/ticket)"
          % (delta, elapsed, elapsed / len(queries)))

    print("\n--- 3. rank rescues (BM25 promoted a passage meaning alone buried) ---")
    if rescues:
        for q, cid, rank in rescues[:10]:
            print("   local rank %3d -> 1   %s" % (rank, q))
            print("                         %s" % cid)
    else:
        print("   none")

    if regressions:
        print("\nREGRESSIONS — usable locally, not at rank 1 in hybrid:")
        for q, lo, hi in regressions:
            print("   local %.3f -> hybrid top-1 %.3f   %s" % (lo, hi or 0.0, q))
        print("\nIf this list is long the lexical half is too strong. Lower")
        print("WEIGHT_LEXICAL or the answers_questions boost in config.py.")

    if args.save:
        out = config.ROOT / "opensearch" / "evaluate_results.json"
        out.write_text(
            json.dumps(
                {"config": config.summary(), "rows": rows,
                 "fidelity_max_error": max_err,
                 "baseline": base_stats, "hybrid": live_stats},
                indent=2, default=str),
            encoding="utf-8",
        )
        print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
