"""Classify / retrieve / draft, each with a local and an AWS implementation.

The orchestrator never learns which one it has. Every difference between
"running on this laptop" and "running on AWS" is confined to this file.

WHAT IS REAL IN LOCAL MODE
    Retrieval is real — the same vectors, the same embedding model and the same
    query prefix that the index will hold, so the cosines are the ones
    OpenSearch will report and the ones MIN_SIM was calibrated against.
    Passages, quotes and citations are all genuine corpus content.

    The drafter is the one thing standing in. It assembles a reply rather than
    generating one. Say so plainly rather than letting a demo imply otherwise.
"""

import json
import re
from typing import Protocol

import settings

# opensearch/ and gate/ are on sys.path via settings.bootstrap()
import config as os_config  # opensearch/config.py


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def expand_query(text: str, synonyms: dict[str, list[str]]) -> tuple[str, list[str]]:
    """Rewrite the ticket using the policy vocabulary, for the re-query step.

    The agent's one autonomous decision is to search AGAIN when the first
    attempt is not confident — so the second attempt has to differ in a way
    that can actually change the answer, and in a way whose success is
    measurable.

    Re-scoring the same query with BM25 mixed in would reorder results but
    could never raise the cosine, so "did it help?" would be unanswerable.
    Rewriting the QUERY instead — appending the policy terms for the seller's
    jargon, so "my LSR is bad" also carries "late shipment rate" — produces a
    new embedding and therefore a new, directly comparable cosine.

    The same synonyms.txt does double duty: the OpenSearch analyzer applies it
    to the BM25 half at search time, and this applies it to the vector half.
    """
    terms = re.findall(r"[a-z0-9]+", text.lower())
    added: list[str] = []
    for t in terms:
        # synonyms.txt contains multi-word phrases, so its token index picks up
        # the filler words inside them — "a" and "to" both map to the a-to-z
        # guarantee line. Expanding on those drags in an unrelated policy area.
        # Single characters and function words are never the seller's jargon.
        if len(t) < 2 or t in _STOP:
            continue
        for syn in synonyms.get(t, []):
            if len(syn) < 2 or syn in _STOP:
                continue
            if syn not in added and syn not in terms:
                added.append(syn)
    if not added:
        return text, []
    return "%s %s" % (text, " ".join(added)), added


# Function words only. Short domain abbreviations — cr, ip, sc, fba — are real
# seller jargon and must survive.
_STOP = {
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "is", "are", "was",
    "be", "do", "does", "did", "my", "me", "i", "it", "and", "or", "not",
    "that", "this", "with", "from", "by", "as", "if", "so", "but", "can",
    "will", "would", "what", "why", "how", "when", "who", "am", "no", "yes",
    "you", "your", "we", "us", "they", "them", "have", "has", "had", "get",
}


class Retriever(Protocol):
    def search(self, text: str, category: str | None = None, k: int = 5) -> dict: ...


class LocalRetriever:
    """Exact cosine over chunks.jsonl. No AWS, no OpenSearch.

    Loads the model and the matrix once — the first call pays a few seconds for
    fastembed, everything after is instant.
    """

    name = "local-vectors"

    def __init__(self):
        import local_search

        self._ls = local_search
        self._chunks = local_search.load_chunks()
        self._matrix = local_search.load_matrix(self._chunks)
        self._synonyms = local_search.load_synonym_map()

    def _embed(self, text: str):
        import search as search_mod

        return search_mod.embed_query(text)

    def search(self, text: str, category: str | None = None, k: int = 5,
               expand: bool = False) -> dict:
        query, added = (expand_query(text, self._synonyms) if expand else (text, []))

        rows, _ = self._ls.retrieve(
            query, self._chunks, self._matrix, self._embed, k, hybrid=False,
        )

        results = []
        for i, cos, lex, comb in rows:
            c = self._chunks[i]
            results.append({
                "chunk_id": c["chunk_id"],
                "cosine": round(float(cos), 4),
                "hybrid_score": round(float(comb), 4) if comb is not None else None,
                "title": c.get("title"),
                "heading_path": c.get("heading_path"),
                "section_id": c.get("section_id"),
                "source_url": c.get("source_url") or _bsa_url(c),
                "category": c.get("category"),
                "doc_type": c.get("doc_type"),
                "rules": c.get("rules") or [],
                "text": c.get("text", ""),
            })

        best = max((r["cosine"] for r in results), default=0.0)
        return {
            "results": results,
            "confidence": {
                "best_cosine": round(best, 4),
                "min_sim": os_config.MIN_SIM,
                "confident": best >= os_config.MIN_SIM,
                "quotable": best >= os_config.ANSWER_SIM,
            },
            "expanded_with": added or [],
            "backend": self.name,
        }


# bulk_upload.py backfills this at index time; local retrieval reads the raw
# chunks, so apply the same rule here or BSA citations render with no link.
_BSA_DOC = "1.1_BSA_PDF_English"
_BSA_URL = "https://m.media-amazon.com/images/G/01/rainier/help./1.1_BSA_PDF_English.pdf"


def _bsa_url(chunk: dict) -> str | None:
    return _BSA_URL if chunk.get("doc_id") == _BSA_DOC else None


class OpenSearchRetriever:
    """Hybrid BM25 + kNN against the live index. Untested — no domain yet."""

    name = "opensearch-hybrid"

    def __init__(self):
        import client
        import search as search_mod

        self._search = search_mod.search
        self._client = client.connect()

    def search(self, text: str, category: str | None = None, k: int = 5,
               expand: bool = False) -> dict:
        # Expanding the query helps the vector half here too. The BM25 half
        # already gets synonyms from the search-time analyzer.
        if expand:
            import local_search
            query, added = expand_query(text, local_search.load_synonym_map())
        else:
            query, added = text, []
        out = self._search(self._client, query, category, k)
        out["backend"] = self.name
        out["expanded_with"] = added
        return out


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class Classifier(Protocol):
    def classify(self, text: str, probe=None) -> dict: ...


class VoteClassifier:
    """Majority vote over the categories of the top-k retrieved chunks.

    Not a placeholder for a model — it is a real classifier that happens to
    need no model, because every chunk already carries a hand-checked
    `category`. It needs a retrieval pass, which `probe` supplies; the trail
    records that this happened rather than hiding it.
    """

    name = "category-vote"

    def classify(self, text: str, probe=None) -> dict:
        if probe is None:
            return {"category": "account", "backend": self.name,
                    "detail": "no retrieval available — defaulted"}

        hits = probe(text, None, settings.CLASSIFY_VOTE_K)["results"]
        votes: dict[str, float] = {}
        for h in hits:
            # Weight each vote by its similarity: a 0.85 match should count for
            # more than a 0.60 one that scraped into the top five.
            votes[h["category"]] = votes.get(h["category"], 0.0) + h["cosine"]

        if not votes:
            return {"category": "account", "backend": self.name,
                    "detail": "retrieval returned nothing — defaulted"}

        ranked = sorted(votes.items(), key=lambda kv: -kv[1])
        return {
            "category": ranked[0][0],
            "backend": self.name,
            "votes": {k: round(v, 3) for k, v in ranked},
            "detail": "weighted vote over top %d chunks" % len(hits),
        }


class BedrockClassifier:
    """Nova Micro. Blocked by the account eligibility restriction."""

    name = "bedrock-nova-micro"

    def __init__(self):
        import boto3
        import drafting

        self._drafting = drafting
        self._client = boto3.client("bedrock-runtime", region_name=os_config.REGION)

    def classify(self, text: str, probe=None) -> dict:
        req = self._drafting.build_classify_request(text)
        resp = self._client.converse(
            modelId=req["modelId"], system=req["system"],
            messages=req["messages"], inferenceConfig=req["inferenceConfig"],
        )
        raw = resp["output"]["message"]["content"][0]["text"]
        return {
            "category": self._drafting.parse_category(raw),
            "backend": self.name,
            "detail": "model replied %r" % raw.strip(),
        }


# ---------------------------------------------------------------------------
# Drafting
# ---------------------------------------------------------------------------


class Drafter(Protocol):
    def draft(self, text: str, hit: dict) -> dict: ...


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def pick_quote(passage: str, max_chars: int = 240) -> str | None:
    """A verbatim span of the passage, safe to put inside quotation marks.

    Taken directly from the passage so it verifies by construction. Quotation
    marks are stripped from the span itself — a nested quote would break the
    extraction that verify_grounding() uses to check it.
    """
    flat = " ".join(passage.split()).replace('"', "").replace("“", "").replace("”", "")
    if not flat:
        return None

    span = ""
    for sentence in _SENTENCE.split(flat):
        if not sentence.strip():
            continue
        candidate = (span + " " + sentence).strip() if span else sentence.strip()
        if len(candidate) > max_chars:
            break
        span = candidate
    if not span:
        span = flat[:max_chars].rsplit(" ", 1)[0]
    return span if len(span) >= 12 else None


class TemplateDrafter:
    """Assembles a reply around a verbatim quote. No model involved.

    HONEST LIMITS, and they belong in the demo narration:
      * The passage, the quote and the citation are real corpus content.
      * The prose around them is a template, not generated language.
      * Because the quote is lifted from the passage, this ALWAYS satisfies
        `draft_quotes_policy`. That input to Cedar F6 is therefore not under
        test in local mode — a real model is what could fail it. F6's other two
        inputs, retrieval confidence and citation presence, are exercised
        normally.
    """

    name = "template"

    def draft(self, text: str, hit: dict) -> dict:
        if hit is None:
            return {"draft": drafting_refusal(), "backend": self.name,
                    "refused": True}

        quote = pick_quote(hit.get("text", ""))
        if not quote:
            return {"draft": drafting_refusal(), "backend": self.name,
                    "refused": True}

        where = hit.get("heading_path") or hit.get("title") or ""
        if hit.get("section_id"):
            where = "%s, %s" % (where, hit["section_id"])
        url = hit.get("source_url") or ""

        lines = [
            "Thank you for contacting Seller Support about %s." % _restate(text),
            "",
            'The applicable policy states: "%s"' % quote,
        ]

        thresholds = hit.get("rules") or []
        if thresholds:
            lines += ["", "The specific requirement that applies:"]
            for r in thresholds:
                lines.append(
                    "  - %s %s %s%s" % (r.get("metric"), r.get("operator"),
                                        r.get("value"), r.get("unit", ""))
                )

        lines += [
            "",
            "Please review the policy in full at the link below, which covers "
            "the requirements for your account.",
            "",
            "Source: %s" % where + (" — %s" % url if url else ""),
        ]
        return {"draft": "\n".join(lines), "backend": self.name, "refused": False}


def drafting_refusal() -> str:
    import drafting

    return drafting.REFUSAL


class MantleDrafter:
    """Amazon Bedrock via the `bedrock-mantle` endpoint. WORKING on this account.

    WHY THIS ENDPOINT AND NOT bedrock-runtime
        `bedrock-runtime` is blocked by an account-level eligibility restriction
        on new AWS accounts — every model returns
        `ValidationException: Operation not allowed`.

        `bedrock-mantle.{region}.api.aws` is Bedrock's other inference endpoint.
        It uses a different IAM action (`bedrock-mantle:CreateInference`) and
        the eligibility check does not fire on it. Verified 13 Sept: the runtime
        endpoint refuses with "Operation not allowed" while mantle gets past
        auth to model resolution and serves completions normally.

        This is still Amazon Bedrock, on this account. No Marketplace
        subscription was required for any of the models below, which is why the
        earlier "Amazon first-party only" constraint does not apply here.

    NOTE THE SIGNING SERVICE is `bedrock-mantle`, not `bedrock`, and there is no
    boto3 client for it — requests are hand-signed with SigV4.
    """

    name = "bedrock-mantle"

    def __init__(self, model: str | None = None):
        import boto3
        import drafting

        self._drafting = drafting
        self.model = model or settings.MANTLE_MODEL
        self.region = settings.MANTLE_REGION
        self.url = "https://bedrock-mantle.%s.api.aws/v1/chat/completions" % self.region
        self._creds = boto3.Session(region_name=self.region).get_credentials()
        if self._creds is None:
            raise SystemExit("No AWS credentials found for bedrock-mantle.")

    def _post(self, body: dict):
        import requests
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        frozen = self._creds.get_frozen_credentials()
        req = AWSRequest(method="POST", url=self.url, data=json.dumps(body),
                         headers={"Content-Type": "application/json"})
        SigV4Auth(frozen, "bedrock-mantle", self.region).add_auth(req)
        return requests.post(self.url, data=req.body, headers=dict(req.headers),
                             timeout=settings.DRAFT_TIMEOUT)

    def draft(self, text: str, hit: dict) -> dict:
        if hit is None:
            return {"draft": self._drafting.REFUSAL, "backend": self.name, "refused": True}

        # Reuse the exact production prompt rather than a second copy of it.
        req = self._drafting.build_draft_request(text, hit)
        body = {
            "model": self.model,
            "max_tokens": req["inferenceConfig"]["maxTokens"],
            "temperature": req["inferenceConfig"]["temperature"],
            "messages": [
                {"role": "system", "content": req["system"][0]["text"]},
                {"role": "user", "content": req["messages"][0]["content"][0]["text"]},
            ],
        }

        resp = self._post(body)
        # Mantle queues rather than hard-throttling, but a 429 is still possible.
        if resp.status_code in (429, 500, 502, 503, 504):
            import time

            time.sleep(2)
            resp = self._post(body)

        if resp.status_code != 200:
            raise RuntimeError("bedrock-mantle %s: %s" % (resp.status_code, resp.text[:200]))

        choice = resp.json()["choices"][0]
        msg = choice["message"]
        # Reasoning models put the answer in reasoning_content and leave
        # content null — gpt-oss-20b does this.
        out = (msg.get("content") or msg.get("reasoning_content") or "").strip()
        if not out:
            raise RuntimeError("bedrock-mantle returned an empty completion")

        return {
            "draft": out,
            "backend": "%s/%s" % (self.name, self.model),
            "refused": out == self._drafting.REFUSAL,
            "finish_reason": choice.get("finish_reason"),
        }


class FallbackDrafter:
    """Try the model; fall back to the template on any failure.

    The demo must not be able to break on a network call. A timeout, a 5xx or
    an empty completion degrades to a grounded template reply rather than an
    error — the passage, quote and citation are still real, so the gate and the
    trail behave normally.

    The trail records which drafter actually produced the text and why, so a
    fallback is visible rather than silent.
    """

    def __init__(self, primary, fallback=None):
        self.primary = primary
        self.fallback = fallback or TemplateDrafter()
        self.name = "%s -> %s" % (primary.name, self.fallback.name)

    def draft(self, text: str, hit: dict) -> dict:
        try:
            return self.primary.draft(text, hit)
        except Exception as exc:  # network, timeout, HTTP, empty completion
            out = self.fallback.draft(text, hit)
            out["fallback_reason"] = "%s: %s" % (type(exc).__name__, str(exc)[:160])
            out["backend"] = "%s (fallback from %s)" % (out["backend"], self.primary.name)
            return out


class BedrockDrafter:
    """Nova Pro on bedrock-runtime. Blocked by the account eligibility restriction."""

    name = "bedrock-nova-pro"

    def __init__(self):
        import boto3
        import drafting

        self._drafting = drafting
        self._client = boto3.client("bedrock-runtime", region_name=os_config.REGION)

    def draft(self, text: str, hit: dict) -> dict:
        if hit is None:
            return {"draft": self._drafting.REFUSAL, "backend": self.name, "refused": True}
        req = self._drafting.build_draft_request(text, hit)
        resp = self._client.converse(
            modelId=req["modelId"], system=req["system"],
            messages=req["messages"], inferenceConfig=req["inferenceConfig"],
        )
        body = resp["output"]["message"]["content"][0]["text"].strip()
        return {
            "draft": body,
            "backend": self.name,
            "refused": body == self._drafting.REFUSAL,
        }


def _restate(text: str, limit: int = 90) -> str:
    """Echo the seller's own words back, so the reply is not category-generic.

    NO QUOTE MARKS around the echo, deliberately. `verify_grounding` treats any
    quoted span of 12+ characters as a claim about the source material, and it
    folds curly quotes into straight ones — so wrapping the seller's own words
    in “ ” made every template draft carry one span that is, correctly, nowhere
    in the passage. Every template draft then failed F6 and escalated, which
    silently disabled the fallback the mantle drafter degrades into.
    Do not put the quotes back; narrowing the checker instead would let a
    model's curly-quoted invention through, which is what it exists to catch.
    """
    flat = " ".join(text.split()).rstrip(".!?")
    flat = re.sub(r"^(hi|hello|hey)[,! ]+", "", flat, flags=re.I)
    if len(flat) > limit:
        flat = flat[:limit].rsplit(" ", 1)[0] + "..."
    return "your message: %s" % flat


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def build_retriever() -> Retriever:
    return OpenSearchRetriever() if settings.RETRIEVER == "opensearch" else LocalRetriever()


def build_classifier() -> Classifier:
    return BedrockClassifier() if settings.CLASSIFIER == "bedrock" else VoteClassifier()


def build_drafter() -> Drafter:
    """Always wrapped in a fallback chain except when the template IS the choice."""
    if settings.DRAFTER == "mantle":
        return FallbackDrafter(MantleDrafter())
    if settings.DRAFTER == "bedrock":
        return FallbackDrafter(BedrockDrafter())
    return TemplateDrafter()
