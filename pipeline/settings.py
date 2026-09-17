"""Backend selection for the pipeline, and the import bootstrap.

Everything here is a switch between a LOCAL implementation that runs today and
an AWS implementation that runs once the account is unblocked. Nothing in the
orchestrator knows which is in use.

    PIPELINE_RETRIEVER  local | opensearch     (default: local)
    PIPELINE_CLASSIFIER local | bedrock        (default: local)
    PIPELINE_DRAFTER    template | mantle | bedrock  (default: mantle)
    MANTLE_MODEL        a bedrock-mantle model id   (default: Mistral Large 3)
    PIPELINE_TRAIL      jsonl | dynamodb       (default: jsonl)
    PIPELINE_PRECEDENT  jsonl | opensearch | off
                        (default: opensearch when the retriever is, else jsonl)

IMPORT BOOTSTRAP
    opensearch/ and gate/ were written as flat scripts — they do `import
    config`, `import topics`, `import gate`. Rather than rewrite them into
    packages (and break every documented command), their directories go on
    sys.path here.

    This is why no module in pipeline/ may be named config, client, search,
    topics, gate, drafting, session, evaluate or preflight — those names belong
    to the two directories being imported. Hence settings.py, not config.py.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OPENSEARCH_DIR = ROOT / "opensearch"
GATE_DIR = ROOT / "gate"


def bootstrap() -> None:
    """Put gate/ and opensearch/ on sys.path, once, in a stable order."""
    for d in (GATE_DIR, OPENSEARCH_DIR):
        p = str(d)
        if p not in sys.path:
            sys.path.insert(0, p)


bootstrap()


def _pick(var: str, default: str, allowed: set[str]) -> str:
    value = os.environ.get(var, default).lower()
    if value not in allowed:
        raise SystemExit(
            "%s=%r is not one of %s" % (var, value, sorted(allowed))
        )
    return value


RETRIEVER = _pick("PIPELINE_RETRIEVER", "local", {"local", "opensearch"})
CLASSIFIER = _pick("PIPELINE_CLASSIFIER", "local", {"local", "bedrock"})
DRAFTER = _pick("PIPELINE_DRAFTER", "mantle", {"template", "mantle", "bedrock"})
TRAIL = _pick("PIPELINE_TRAIL", "jsonl", {"jsonl", "dynamodb"})

# Where past tickets live for the precedent check. They follow the retriever:
# if policy search is on OpenSearch the past tickets sit in a second index on
# the same domain; in-process retrieval pairs with a local JSONL file. The
# deployed function cannot scan DynamoDB (its role holds PutItem and Query
# only), which is why the trail is not the store here.
PRECEDENT = _pick(
    "PIPELINE_PRECEDENT",
    "opensearch" if RETRIEVER == "opensearch" else "jsonl",
    {"jsonl", "opensearch", "off"},
)

# bedrock-mantle is Bedrock's other inference endpoint. It works on this
# account, where bedrock-runtime is blocked by the new-account eligibility
# restriction. Mistral Large 3 is the default because it was the most
# disciplined of the models tested — it declined to invent next steps the
# passage did not contain, which is rule 5 of DRAFTING_INSTRUCTIONS.md.
MANTLE_MODEL = os.environ.get("MANTLE_MODEL", "mistral.mistral-large-3-675b-instruct")
MANTLE_REGION = os.environ.get("AWS_REGION", "us-east-1")
DRAFT_TIMEOUT = int(os.environ.get("PIPELINE_DRAFT_TIMEOUT", "60"))

TRAIL_PATH = ROOT / "pipeline" / "decision_trail.jsonl"
PRECEDENTS_PATH = ROOT / "pipeline" / "precedents.jsonl"
DYNAMO_TABLE = os.environ.get("PIPELINE_TABLE", "seller-triage-tickets")

# How many chunks the local classifier votes over.
CLASSIFY_VOTE_K = 5

# Retrieval results carried into drafting and the trail.
TOP_K = 5


def summary() -> str:
    return "retriever=%s classifier=%s drafter=%s trail=%s precedent=%s" % (
        RETRIEVER,
        CLASSIFIER,
        DRAFTER,
        TRAIL,
        PRECEDENT,
    )


def banner() -> str:
    """Say plainly which parts are real and which are standing in."""
    notes = []
    if DRAFTER == "template":
        notes.append(
            "drafter=template — the passage, quote and citation are real; "
            "the prose around them is assembled, not generated"
        )
    if DRAFTER == "mantle":
        notes.append(
            "drafter=bedrock-mantle/%s — real generation on Amazon Bedrock, "
            "falling back to the template if the call fails" % MANTLE_MODEL
        )
    if CLASSIFIER == "local":
        notes.append(
            "classifier=local — majority vote over the categories of the "
            "top %d retrieved chunks, no model" % CLASSIFY_VOTE_K
        )
    if RETRIEVER == "local":
        notes.append(
            "retriever=local — exact same vectors and model as the index will "
            "hold; cosines are production-accurate, BM25 is not in play"
        )
    return "\n".join("  note: " + n for n in notes)
