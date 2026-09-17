"""Single source of truth for the OpenSearch side.

The one number that matters here is the embedding dimension. `knn_vector`
dimension is fixed when the index is created and cannot be altered afterwards,
so every script reads it from one place rather than hardcoding it.

It is auto-detected from chunks.jsonl, which means the local-bge (384) to
Bedrock Titan (1024) swap needs no code change at all: re-run
`reembed_bedrock.py`, then recreate the index and re-upload.

Environment overrides:
    OPENSEARCH_ENDPOINT   domain endpoint, no scheme (required to talk to AWS)
    OPENSEARCH_INDEX      index name              (default seller-policy)
    OPENSEARCH_LOCAL      "1" to target http://localhost:9200 with no auth
    AWS_REGION            defaults to us-east-1
    EMBED_DIM             override the auto-detected dimension
    SYNONYMS_PACKAGE_ID   Amazon OpenSearch custom package id, e.g. F123456789
"""

import json
import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "chunks.jsonl"
QUERIES = ROOT / "filter_queries.tsv"
SYNONYMS_FILE = ROOT / "opensearch" / "synonyms.txt"

# --------------------------------------------------------------------------
# Connection
# --------------------------------------------------------------------------

REGION = os.environ.get("AWS_REGION", "us-east-1")
ENDPOINT = os.environ.get("OPENSEARCH_ENDPOINT", "").replace("https://", "").rstrip("/")
USE_LOCAL = os.environ.get("OPENSEARCH_LOCAL") == "1"
LOCAL_URL = "http://localhost:9200"

INDEX = os.environ.get("OPENSEARCH_INDEX", "seller-policy")
PIPELINE = "seller-triage-hybrid"

# Past tickets, one document each, so a new ticket can be checked against what
# was already answered. Separate index: it grows with every ticket, carries a
# different mapping, and must never be confused with the frozen policy corpus.
TICKETS_INDEX = os.environ.get("OPENSEARCH_TICKETS_INDEX", "seller-tickets")

# --------------------------------------------------------------------------
# Embedding dimension — auto-detected, never hardcoded
# --------------------------------------------------------------------------


def detect_dim(path: Path = CHUNKS) -> int:
    """Read the vector length off the first chunk in chunks.jsonl."""
    override = os.environ.get("EMBED_DIM")
    if override:
        return int(override)
    if not path.exists():
        raise SystemExit(
            "%s not found — cannot detect the embedding dimension.\n"
            "Set EMBED_DIM explicitly if you are creating the index first." % path
        )
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                vec = json.loads(line).get("embedding")
                if not vec:
                    raise SystemExit("First chunk in %s has no 'embedding' field." % path)
                return len(vec)
    raise SystemExit("%s is empty." % path)


EMBED_DIM = detect_dim()

# Which model produced those vectors. Only used for human-readable warnings —
# 384 is bge-small-en-v1.5 (local), 1024/512/256 is Titan Text Embeddings V2.
EMBED_BACKEND = "local-bge" if EMBED_DIM == 384 else "bedrock-titan"

# --------------------------------------------------------------------------
# kNN index parameters
# --------------------------------------------------------------------------

# Both bge and Titan (with normalize=True) emit unit-length vectors, so cosine
# is the right space. lucene is chosen over faiss deliberately: at 710 vectors
# the ANN engine is irrelevant to recall, and lucene needs no extra native
# memory on a t3.small.search box, which has 2 GB total.
KNN_ENGINE = "lucene"
KNN_SPACE = "cosinesimil"
KNN_M = 16
KNN_EF_CONSTRUCTION = 128
KNN_EF_SEARCH = 100

# --------------------------------------------------------------------------
# Hybrid search
# --------------------------------------------------------------------------

# Weights for the normalization processor, [lexical, vector]. They must sum
# to 1.0. Vector is weighted higher because sellers write plain language, but
# the lexical half is what rescues exact-phrase tickets such as
# "how do I write a plan of action", which scores only 0.626 on meaning alone.
WEIGHT_LEXICAL = 0.4
WEIGHT_VECTOR = 0.6

TOP_K = 5           # results returned to the agent
CANDIDATE_K = 20    # candidates each sub-query contributes before combination

# Field weights inside the lexical sub-query.
#
# `answers_questions` holds the plain-language seller questions each chunk was
# found to answer during semantic filtering. Matching a ticket against those is
# seller-language against seller-language, which is why it outranks `text`.
LEXICAL_FIELDS = [
    "answers_questions^3.0",
    "title^2.0",
    "heading_path^1.5",
    "text^1.0",
]

# --------------------------------------------------------------------------
# doc_type boost
# --------------------------------------------------------------------------

# On payments and account tickets the Business Solutions Agreement is the
# authoritative source and should outrank a help page: the withheld-payments
# and deactivation-grounds clauses live in the agreement, not the help hub.
# Applied as a soft boost inside the lexical sub-query, never as a filter.
AGREEMENT_BOOST = 1.6
AGREEMENT_BOOST_CATEGORIES = {"payments", "account"}

# A ticket classified into a category gets chunks from that category boosted.
# Soft, not a filter — a misclassification must not be able to hide the only
# passage that would have answered the ticket.
CATEGORY_BOOST = 1.4

# --------------------------------------------------------------------------
# Confidence thresholds
# --------------------------------------------------------------------------

# IMPORTANT: these are raw cosine similarities, and they belong to whichever
# model produced the vectors currently in chunks.jsonl.
#
# 0.72 was calibrated against bge-small. Thresholds do NOT transfer between
# embedding models — after switching to Titan, re-read them off
# `reembed_bedrock.py --verify` before trusting them.
#
# They must NOT be compared against a hybrid score. Min-max normalization
# rescales the best hit of every query to ~1.0, so a query that matches nothing
# still comes back with a top hybrid score near 1.0. Confidence has to be read
# from the pure-kNN score instead, which is what search.py does.
MIN_SIM = 0.72      # below this, retrieval is not confident -> agent re-queries
ANSWER_SIM = 0.72   # a passage at or above this is quotable in a draft

CALIBRATED_FOR = "bge-small-en-v1.5 (384d)"

# --------------------------------------------------------------------------
# Precedent thresholds — ticket-to-ticket cosine, same model as above
# --------------------------------------------------------------------------

# Measured on the 67 filter tickets, which are all DIFFERENT questions: the
# nearest neighbour of each one sits at p50 0.791, p90 0.828, max 0.860. Two
# phrasings of the SAME question ("my listing is suppressed and not showing up
# in search" / "my listing is not showing up in search results") sit at
# 0.85-0.89. So:
#
#   PRECEDENT_SIM   0.80  "similar enough to show the agent" — worth a look,
#                         may be a neighbouring question rather than this one
#   PRECEDENT_SAME  0.86  "the same question" — above the highest similarity
#                         any two distinct filter tickets reach. Only at this
#                         level does a different policy page count as a
#                         conflicting answer, because below it the two tickets
#                         may legitimately be about different things.
#
# Both are raw cosines and belong to bge-small like MIN_SIM does.
PRECEDENT_SIM = 0.80
PRECEDENT_SAME = 0.86
PRECEDENT_K = 5


def warn_if_thresholds_stale() -> str | None:
    """Return a warning if the thresholds were calibrated for another model."""
    if EMBED_BACKEND != "local-bge":
        return (
            "MIN_SIM/ANSWER_SIM = %.2f was calibrated for %s, but chunks.jsonl now\n"
            "holds %d-dim %s vectors. Re-check against `reembed_bedrock.py --verify`."
            % (MIN_SIM, CALIBRATED_FOR, EMBED_DIM, EMBED_BACKEND)
        )
    return None


def summary() -> str:
    return (
        "index=%s  dim=%d (%s)  engine=%s/%s  weights=lex %.1f/vec %.1f  target=%s"
        % (
            INDEX,
            EMBED_DIM,
            EMBED_BACKEND,
            KNN_ENGINE,
            KNN_SPACE,
            WEIGHT_LEXICAL,
            WEIGHT_VECTOR,
            LOCAL_URL if USE_LOCAL else (ENDPOINT or "<OPENSEARCH_ENDPOINT unset>"),
        )
    )
