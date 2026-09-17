"""Offline consistency checks. No AWS, no network.

    python opensearch/preflight.py

There is no local OpenSearch on this machine to rehearse against, so these
checks stand in for the parts a live cluster would catch. They target the two
failure modes that are silent rather than loud:

  * `dynamic: strict` means any field present in a document but absent from the
    mapping fails the whole bulk batch. Drift between bulk_upload.INDEXED_FIELDS
    and the mapping is the likeliest cause.
  * A typo in config.LEXICAL_FIELDS does NOT error. multi_match happily scores
    against a field that does not exist and contributes nothing, so the lexical
    half quietly degrades and only shows up as mediocre ranking.

Run this before create_index.py, and again after editing config.py.
"""

import json
import sys

import config
from index_mapping import build_mapping, build_pipeline, load_synonyms

FAILURES: list[str] = []
WARNINGS: list[str] = []


def check(condition: bool, message: str, warn_only: bool = False) -> None:
    if condition:
        print("  ok    %s" % message)
    elif warn_only:
        WARNINGS.append(message)
        print("  WARN  %s" % message)
    else:
        FAILURES.append(message)
        print("  FAIL  %s" % message)


def main() -> None:
    print(config.summary())
    print()

    mapping = build_mapping()
    props = mapping["mappings"]["properties"]
    analysis = mapping["settings"]["analysis"]

    print("1. mapping / document agreement")
    import bulk_upload

    declared = set(props)
    indexed = set(bulk_upload.INDEXED_FIELDS)
    check(
        indexed <= declared,
        "every field bulk_upload sends is declared (%s)" % (sorted(indexed - declared) or "none extra"),
    )

    chunks_fields: set[str] = set()
    with config.CHUNKS.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if line.strip():
                chunks_fields |= set(json.loads(line))
            if i > 200:
                break
    dropped = chunks_fields - indexed
    check(True, "chunks.jsonl fields intentionally dropped: %s" % (sorted(dropped) or "none"))
    undeclared = (chunks_fields & indexed) - declared
    check(not undeclared, "no chunk field would be rejected by dynamic:strict (%s)" % (sorted(undeclared) or "none"))

    print("\n2. lexical field names resolve")
    for spec in config.LEXICAL_FIELDS:
        name = spec.split("^")[0]
        field = props.get(name)
        ok = field is not None and field.get("type") == "text"
        check(ok, "%-24s -> %s" % (spec, "text field" if ok else "MISSING or not text"))

    print("\n3. source fields resolve")
    import search as search_mod

    missing = [f for f in search_mod.SOURCE_FIELDS if f not in props]
    check(not missing, "_source fields all declared (%s)" % (missing or "none missing"))

    print("\n4. vector configuration")
    emb = props["embedding"]
    check(emb["type"] == "knn_vector", "embedding is knn_vector")
    check(
        emb["dimension"] == config.EMBED_DIM,
        "dimension %d matches chunks.jsonl" % emb["dimension"],
    )
    check(mapping["settings"]["index"]["knn"] is True, "index.knn enabled")
    check(
        emb["method"]["space_type"] == "cosinesimil",
        "space_type cosinesimil (vectors are unit-length)",
    )

    dims = set()
    with config.CHUNKS.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                dims.add(len(json.loads(line)["embedding"]))
    check(len(dims) == 1, "all chunk vectors share one dimension %s" % sorted(dims))

    print("\n5. hybrid pipeline")
    pipe = build_pipeline()
    weights = pipe["phase_results_processors"][0]["normalization-processor"]["combination"][
        "parameters"
    ]["weights"]
    check(abs(sum(weights) - 1.0) < 1e-6, "weights sum to 1.0 %s" % weights)
    check(len(weights) == 2, "exactly two weights, matching two sub-queries")

    body = {
        "query": {
            "hybrid": {
                "queries": [
                    search_mod.lexical_query("test ticket", "payments"),
                    search_mod.vector_query([0.0] * config.EMBED_DIM),
                ]
            }
        }
    }
    try:
        json.dumps(body)
        order_ok = "bool" in body["query"]["hybrid"]["queries"][0]
        check(order_ok, "sub-query order is [lexical, vector], matching the weights")
    except (TypeError, ValueError) as exc:
        check(False, "hybrid body serializes: %s" % exc)

    boosted = search_mod.lexical_query("q", "payments")
    check(
        len(boosted["bool"]["should"]) == 2 and boosted["bool"]["must"],
        "payments ticket gets agreement + category boosts, text match stays in must",
    )
    unboosted = search_mod.lexical_query("q", None)
    check(
        unboosted["bool"]["should"] == [],
        "uncategorised ticket gets no boosts",
    )

    print("\n6. analyzers")
    check(
        "seller_synonyms" in analysis["analyzer"]["policy_search"]["filter"],
        "synonyms applied at search time",
    )
    check(
        "seller_synonyms" not in analysis["analyzer"]["policy_index"]["filter"],
        "synonyms NOT applied at index time (list stays editable)",
    )
    syns = load_synonyms()
    check(len(syns) > 0, "%d synonym rules loaded" % len(syns))
    bad = [s for s in syns if "," not in s and "=>" not in s]
    check(not bad, "every synonym line is valid Solr format (%s)" % (bad[:2] or "all valid"))

    print("\n7. thresholds")
    stale = config.warn_if_thresholds_stale()
    check(stale is None, stale or "thresholds match the embedding model in chunks.jsonl", warn_only=True)

    print()
    if FAILURES:
        print("%d FAILURES:" % len(FAILURES))
        for f in FAILURES:
            print("   - %s" % f)
        sys.exit(1)
    print("preflight passed%s." % (" with %d warning(s)" % len(WARNINGS) if WARNINGS else ""))
    print("\nWhat this cannot check without a live cluster: that the hybrid query")
    print("and normalization pipeline behave as expected. evaluate.py is the test")
    print("for that — run it as soon as the domain exists.")


if __name__ == "__main__":
    main()
