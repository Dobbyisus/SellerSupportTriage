"""Creates the index and the hybrid search pipeline.

    python opensearch/create_index.py             # create if absent
    python opensearch/create_index.py --recreate  # DELETE then create
    python opensearch/create_index.py --dry-run   # print, touch nothing

The search pipeline is a separate object from the index and is easy to forget.
Without it a `hybrid` query still runs but the sub-query scores are never
normalized, so BM25 swamps the vector half and results look subtly wrong
rather than failing outright.
"""

import sys

import client
import config
from index_mapping import build_mapping, build_pipeline


def put_pipeline(os_client) -> None:
    body = build_pipeline()
    os_client.transport.perform_request(
        "PUT", "/_search/pipeline/%s" % config.PIPELINE, body=body
    )
    print("search pipeline '%s' created/updated" % config.PIPELINE)
    print(
        "   normalization=min_max  combination=arithmetic_mean  weights=[lex %.1f, vec %.1f]"
        % (config.WEIGHT_LEXICAL, config.WEIGHT_VECTOR)
    )


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    recreate = "--recreate" in sys.argv

    print(config.summary())
    stale = config.warn_if_thresholds_stale()
    if stale:
        print("\nWARNING\n" + stale + "\n")

    if dry_run:
        import json

        print("\n--- index mapping ---")
        print(json.dumps(build_mapping(), indent=2)[:1200] + "\n...")
        print("\n--- search pipeline ---")
        print(json.dumps(build_pipeline(), indent=2))
        print("\ndry run: nothing was created.")
        return

    os_client = client.connect()
    info = client.ping(os_client)
    print("connected: %s %s (cluster %s)" % (info["distribution"], info["number"], info["cluster"]))
    client.require_hybrid_support(os_client)

    exists = os_client.indices.exists(index=config.INDEX)

    if exists and recreate:
        os_client.indices.delete(index=config.INDEX)
        print("deleted existing index '%s'" % config.INDEX)
        exists = False
    elif exists:
        current = os_client.indices.get_mapping(index=config.INDEX)
        props = current[config.INDEX]["mappings"]["properties"]
        live_dim = props.get("embedding", {}).get("dimension")
        if live_dim != config.EMBED_DIM:
            raise SystemExit(
                "Index '%s' already exists with embedding dimension %s, but chunks.jsonl\n"
                "now holds %d-dim vectors. knn_vector dimension is immutable — rerun with\n"
                "--recreate, then re-run bulk_upload.py." % (config.INDEX, live_dim, config.EMBED_DIM)
            )
        print("index '%s' already exists at %d dims — leaving it alone" % (config.INDEX, live_dim))
        put_pipeline(os_client)
        return

    os_client.indices.create(index=config.INDEX, body=build_mapping())
    print("created index '%s' with %d-dim knn_vector (%s/%s)"
          % (config.INDEX, config.EMBED_DIM, config.KNN_ENGINE, config.KNN_SPACE))

    put_pipeline(os_client)
    print("\nNext: python opensearch/bulk_upload.py")


if __name__ == "__main__":
    main()
