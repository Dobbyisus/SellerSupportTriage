"""Uploads chunks.jsonl into the index.

    python opensearch/bulk_upload.py            # upload
    python opensearch/bulk_upload.py --dry-run  # validate + report, no network
    python opensearch/bulk_upload.py --verify   # count and spot-check what is live

chunks.jsonl is one document per line, but _bulk needs an action line before
every document line, so the two get interleaved here.

_id is set to chunk_id, which is unique and stable. That makes re-uploading
idempotent: a second run overwrites in place instead of duplicating, so a
failed run can simply be repeated.
"""

import json
import sys

import config

# The BSA was collected as a PDF and its chunks carry no source_url, which
# leaves 188 of 710 chunks (26%) with a dead citation — and they are the
# highest-stakes 26%, since the payment-hold and deactivation clauses that the
# gate rules cite live in the agreement rather than the help hub.
#
# The trailing dot in "help./" is real. Correcting it 404s.
BSA_DOC_ID = "1.1_BSA_PDF_English"
BSA_URL = "https://m.media-amazon.com/images/G/01/rainier/help./1.1_BSA_PDF_English.pdf"

# Fields the index declares. `dynamic: strict` rejects anything else, so
# unknown keys are dropped here rather than failing the whole batch.
INDEXED_FIELDS = {
    "embedding", "text", "answers_questions", "best_question", "title",
    "heading_path", "chunk_id", "doc_id", "category", "answer_categories",
    "doc_type", "section_id", "source_url", "local_copy", "word_count",
    "best_similarity", "norm_char_start", "norm_char_end", "rules",
}

BATCH_SIZE = 100


def load_chunks() -> list[dict]:
    return [
        json.loads(line)
        for line in config.CHUNKS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def prepare(chunk: dict, stats: dict) -> dict:
    """Trim to the declared fields and backfill the missing BSA citation."""
    doc = {k: v for k, v in chunk.items() if k in INDEXED_FIELDS}

    if not doc.get("source_url"):
        if doc.get("doc_id") == BSA_DOC_ID:
            section = doc.get("section_id")
            doc["source_url"] = BSA_URL
            # Section markers are internal to the PDF, so there is no anchor to
            # link to; carrying the marker lets the console render
            # "Business Solutions Agreement, S-5" beside the link.
            if section:
                stats["bsa_with_section"] += 1
            stats["backfilled"] += 1
        else:
            stats["still_missing"] += 1

    # An absent value and an empty list mean the same thing downstream; keep
    # the payload small.
    for key in ("answers_questions", "answer_categories", "rules"):
        if key in doc and not doc[key]:
            doc.pop(key)

    return doc


def batches(docs: list[dict], size: int):
    for i in range(0, len(docs), size):
        yield docs[i : i + size]


def to_bulk_body(docs: list[dict]) -> list[dict]:
    """Interleave an action line before each document line."""
    body = []
    for doc in docs:
        body.append({"index": {"_index": config.INDEX, "_id": doc["chunk_id"]}})
        body.append(doc)
    return body


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    verify = "--verify" in sys.argv

    print(config.summary())
    stale = config.warn_if_thresholds_stale()
    if stale:
        print("\nWARNING\n" + stale + "\n")

    chunks = load_chunks()
    stats = {"backfilled": 0, "still_missing": 0, "bsa_with_section": 0}
    docs = [prepare(c, stats) for c in chunks]

    dims = {len(d["embedding"]) for d in docs if "embedding" in d}
    if len(dims) != 1:
        raise SystemExit("Mixed embedding dimensions in chunks.jsonl: %s" % sorted(dims))
    dim = dims.pop()
    if dim != config.EMBED_DIM:
        raise SystemExit("chunks.jsonl has %d-dim vectors, config expects %d." % (dim, config.EMBED_DIM))

    missing_id = [d for d in docs if not d.get("chunk_id")]
    if missing_id:
        raise SystemExit("%d chunks have no chunk_id — cannot set a stable _id." % len(missing_id))
    if len({d["chunk_id"] for d in docs}) != len(docs):
        raise SystemExit("Duplicate chunk_id values — _id would collide.")

    print("\n%d chunks, %d dims, all chunk_ids unique" % (len(docs), dim))
    print("citations: %d BSA chunks backfilled (%d carry a section marker), %d still without a URL"
          % (stats["backfilled"], stats["bsa_with_section"], stats["still_missing"]))
    with_rules = sum(1 for d in docs if d.get("rules"))
    with_questions = sum(1 for d in docs if d.get("answers_questions"))
    print("fields   : %d chunks carry structured rules, %d carry answers_questions"
          % (with_rules, with_questions))

    if dry_run:
        sample = to_bulk_body(docs[:1])
        print("\nfirst two bulk lines:")
        print("  " + json.dumps(sample[0]))
        preview = dict(sample[1])
        preview["embedding"] = "[... %d floats ...]" % dim
        print("  " + json.dumps(preview)[:400] + " ...")
        print("\ndry run: nothing was uploaded.")
        return

    import client

    os_client = client.connect()

    if verify:
        os_client.indices.refresh(index=config.INDEX)
        count = os_client.count(index=config.INDEX)["count"]
        print("\nlive documents: %d (expected %d)" % (count, len(docs)))
        if count != len(docs):
            print("MISMATCH — re-run the upload.")
        probe = os_client.get(index=config.INDEX, id=docs[0]["chunk_id"], _source_excludes=["embedding"])
        print("spot check %s:" % probe["_id"])
        for key in ("title", "heading_path", "category", "doc_type", "source_url"):
            print("   %-14s %s" % (key, probe["_source"].get(key)))
        return

    from opensearchpy.helpers import bulk

    total, failures = 0, []
    for i, batch in enumerate(batches(docs, BATCH_SIZE), start=1):
        actions = [
            {"_op_type": "index", "_index": config.INDEX, "_id": d["chunk_id"], "_source": d}
            for d in batch
        ]
        ok, errors = bulk(os_client, actions, raise_on_error=False, request_timeout=120)
        total += ok
        if errors:
            failures.extend(errors)
        print("  batch %2d: %3d indexed (%d total)" % (i, ok, total), flush=True)

    os_client.indices.refresh(index=config.INDEX)
    print("\nindexed %d/%d documents" % (total, len(docs)))
    if failures:
        print("%d failures, first one:" % len(failures))
        print(json.dumps(failures[0], indent=2)[:800])
        raise SystemExit(1)

    count = os_client.count(index=config.INDEX)["count"]
    print("index '%s' now holds %d documents" % (config.INDEX, count))
    print("\nNext: python opensearch/evaluate.py")


if __name__ == "__main__":
    main()
