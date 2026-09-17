"""Builds the index mapping at the configured embedding dimension.

Run directly to print the mapping without touching AWS:

    python opensearch/index_mapping.py            # pretty-print
    python opensearch/index_mapping.py --save     # write index_mapping.json
"""

import json
import sys

import config


def load_synonyms() -> list[str]:
    """Read synonyms.txt, dropping comments and blanks."""
    if not config.SYNONYMS_FILE.exists():
        return []
    out = []
    for line in config.SYNONYMS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def build_analysis() -> dict:
    """Index-time and search-time analyzers.

    Synonyms are applied at SEARCH time only. Index-time expansion would bake
    the current list into the postings, so every edit would mean re-indexing
    all 710 chunks. At search time the list can change on demo morning.

    On Amazon OpenSearch Service the production route is a custom package:
    upload synonyms.txt, associate it with the domain, then set
    SYNONYMS_PACKAGE_ID to the returned id (e.g. F123456789). Without that we
    inline the list, which works everywhere but needs a close/reopen to change.
    """
    import os

    package_id = os.environ.get("SYNONYMS_PACKAGE_ID", "").strip()
    if package_id:
        synonym_filter = {
            "type": "synonym_graph",
            "synonyms_path": "analyzers/%s" % package_id,
            "updateable": True,
        }
    else:
        synonyms = load_synonyms()
        synonym_filter = {"type": "synonym_graph", "synonyms": synonyms}

    return {
        "filter": {
            "seller_synonyms": synonym_filter,
            "english_stop": {"type": "stop", "stopwords": "_english_"},
            "english_stemmer": {"type": "stemmer", "language": "light_english"},
        },
        "analyzer": {
            # No synonyms here — see the docstring.
            "policy_index": {
                "type": "custom",
                "tokenizer": "standard",
                "filter": ["lowercase", "asciifolding", "english_stop", "english_stemmer"],
            },
            # Synonyms expand the seller's wording into policy vocabulary.
            "policy_search": {
                "type": "custom",
                "tokenizer": "standard",
                "filter": [
                    "lowercase",
                    "asciifolding",
                    "seller_synonyms",
                    "english_stop",
                    "english_stemmer",
                ],
            },
        },
    }


def _text(analyzed: bool = True, with_keyword: bool = False) -> dict:
    field: dict = {"type": "text"}
    if analyzed:
        field["analyzer"] = "policy_index"
        field["search_analyzer"] = "policy_search"
    if with_keyword:
        field["fields"] = {"raw": {"type": "keyword", "ignore_above": 512}}
    return field


def build_mapping(dim: int | None = None) -> dict:
    dim = dim or config.EMBED_DIM

    return {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": config.KNN_EF_SEARCH,
                # 710 documents is a single shard by any measure. Replicas are
                # 0 because a demo domain is single-node; raise to 1 if the
                # domain is ever created with more than one data node.
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "1s",
            },
            "analysis": build_analysis(),
        },
        "mappings": {
            # Everything indexed is declared below. Reject anything else rather
            # than letting OpenSearch guess a type for a field we forgot.
            "dynamic": "strict",
            "properties": {
                # ---- the semantic half ----
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
                # ---- the lexical half ----
                # The passage itself: what BM25 scores and what gets quoted.
                "text": _text(),
                # Plain-language seller questions this chunk answers, produced
                # during semantic filtering. 431 of 710 chunks carry at least
                # one. Matching a ticket against these is seller-language to
                # seller-language, which is how the jargon gap gets closed.
                "answers_questions": _text(),
                "best_question": _text(),
                "title": _text(with_keyword=True),
                # Both: analysed for matching, .raw for rendering the citation
                # "Seller Code of Conduct > Acting Fairly".
                "heading_path": _text(with_keyword=True),
                # ---- filters, boosts and identity ----
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "category": {"type": "keyword"},
                "answer_categories": {"type": "keyword"},
                "doc_type": {"type": "keyword"},
                "section_id": {"type": "keyword"},
                # ---- stored for display, never searched ----
                "source_url": {"type": "keyword", "index": False},
                "local_copy": {"type": "keyword", "index": False},
                "word_count": {"type": "integer"},
                "best_similarity": {"type": "float"},
                "norm_char_start": {"type": "integer", "index": False},
                "norm_char_end": {"type": "integer", "index": False},
                # ---- structured thresholds ----
                # 16 chunks carry machine-readable metric rules such as
                # ODR < 1% and CR < 2.5%. Keeping them structured lets a draft
                # quote the exact figure instead of paraphrasing it, and gives
                # the Cedar gate something to reason over later.
                "rules": {
                    "type": "nested",
                    "properties": {
                        "metric": {"type": "keyword"},
                        "operator": {"type": "keyword"},
                        "value": {"type": "float"},
                        "unit": {"type": "keyword"},
                        "evidence": {"type": "text", "index": False},
                    },
                },
            },
        },
    }


def build_pipeline() -> dict:
    """The hybrid search pipeline.

    BM25 scores are unbounded and corpus-relative; cosine similarity sits in
    [0, 1]. Adding them directly lets BM25 dominate. The normalization
    processor rescales each sub-query's scores per request, then combines them
    by weight.
    """
    return {
        "description": "Normalizes and combines BM25 + kNN for seller policy retrieval",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {
                            "weights": [config.WEIGHT_LEXICAL, config.WEIGHT_VECTOR]
                        },
                    },
                }
            }
        ],
    }


def main() -> None:
    mapping = build_mapping()
    if "--save" in sys.argv:
        out = config.ROOT / "opensearch" / "index_mapping.json"
        out.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        print("wrote %s" % out)
        out = config.ROOT / "opensearch" / "search_pipeline.json"
        out.write_text(json.dumps(build_pipeline(), indent=2), encoding="utf-8")
        print("wrote %s" % out)
    else:
        print(json.dumps(mapping, indent=2))

    syn = load_synonyms()
    print(
        "\n%s\nsynonym rules: %d  vector field: %d dims"
        % (config.summary(), len(syn), config.EMBED_DIM),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
