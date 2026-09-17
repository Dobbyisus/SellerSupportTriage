"""Creates the past-tickets index that the precedent check searches.

    python opensearch/tickets_index.py              # create if absent
    python opensearch/tickets_index.py --recreate   # DELETE then create
    python opensearch/tickets_index.py --dry-run    # print the mapping, touch nothing

Then fill it from the decision trail:

    python pipeline/precedent.py --backfill

WHY A SECOND INDEX
    `seller-policy` is the frozen corpus: 710 passages, replaced only when the
    knowledge base is rebuilt. Past tickets are the opposite — one document per
    ticket, appended on every run, never authoritative about policy. Mixing
    them would let a past ticket come back as a "policy passage", and the
    grounding checker would happily verify a quote against it.

WHY CREATION HAPPENS HERE AND NOT IN THE LAMBDA
    The function's role holds es:ESHttpGet and es:ESHttpPost only. Creating an
    index is a PUT. So the index is created once from a developer machine (the
    seller-triage-dev user has ESHttp*), and the function only ever POSTs
    documents into it and POSTs searches against it — both allowed.

    Note that opensearch-py's `index()` helper issues a PUT when an id is
    given. pipeline/precedent.py therefore posts to `/<index>/_doc/<id>`
    through perform_request, which is a legal POST. Do not "simplify" it back
    to `client.index()` — it will 403 on the function and work on your laptop.
"""

import sys

import client
import config


def build_mapping(dim: int | None = None) -> dict:
    dim = dim or config.EMBED_DIM
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
                # A ticket must be findable by the very next ticket. The writer
                # asks for refresh=true on every put as well; this is the
                # backstop.
                "refresh_interval": "1s",
            }
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                # Same model, same dimension and same space as the policy index,
                # so a ticket vector compares to a ticket vector on the same
                # scale MIN_SIM and PRECEDENT_SIM were calibrated against.
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": config.KNN_SPACE,
                        "engine": config.KNN_ENGINE,
                        "parameters": {
                            "m": config.KNN_M,
                            "ef_construction": config.KNN_EF_CONSTRUCTION,
                        },
                    },
                },
                "ticket_id": {"type": "keyword"},
                "text": {"type": "text"},
                "created_ms": {"type": "long"},
                "category": {"type": "keyword"},
                "decision": {"type": "keyword"},
                "blocked_by": {"type": "keyword"},
                "sent": {"type": "boolean"},
                # The policy the reply was built on. doc_id is the identity the
                # conflict check compares; the rest is for display.
                "doc_id": {"type": "keyword"},
                "policy_title": {"type": "keyword", "index": False},
                "policy_path": {"type": "keyword", "index": False},
                "policy_url": {"type": "keyword", "index": False},
                # What was actually sent (or would have been). Stored, never
                # searched — a past draft must not be retrievable as policy.
                "draft": {"type": "text", "index": False},
            },
        },
    }


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    recreate = "--recreate" in sys.argv

    if dry_run:
        import json

        print(json.dumps(build_mapping(), indent=2))
        print("\ndry run: nothing was created.")
        return

    os_client = client.connect()
    info = client.ping(os_client)
    print("connected: %s %s (cluster %s)" % (info["distribution"], info["number"], info["cluster"]))

    name = config.TICKETS_INDEX
    exists = os_client.indices.exists(index=name)
    if exists and recreate:
        os_client.indices.delete(index=name)
        print("deleted existing index '%s'" % name)
        exists = False
    elif exists:
        live = os_client.indices.get_mapping(index=name)[name]["mappings"]["properties"]
        live_dim = live.get("embedding", {}).get("dimension")
        if live_dim != config.EMBED_DIM:
            raise SystemExit(
                "Index '%s' exists at %s dims but chunks.jsonl holds %d-dim vectors.\n"
                "Rerun with --recreate, then `python pipeline/precedent.py --backfill`."
                % (name, live_dim, config.EMBED_DIM)
            )
        count = os_client.count(index=name)["count"]
        print("index '%s' already exists at %d dims with %d tickets — leaving it alone"
              % (name, live_dim, count))
        return

    os_client.indices.create(index=name, body=build_mapping())
    print("created index '%s' with %d-dim knn_vector (%s/%s)"
          % (name, config.EMBED_DIM, config.KNN_ENGINE, config.KNN_SPACE))
    print("\nNext: python pipeline/precedent.py --backfill")


if __name__ == "__main__":
    main()
