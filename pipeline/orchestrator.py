"""The pipeline: classify -> retrieve -> precedent -> standing -> draft -> gate -> log.

A FIXED SEQUENCE WITH EXACTLY ONE AGENTIC STEP
    The order never varies, which is what keeps a live demo deterministic. The
    single place the system decides something for itself is the re-query: when
    the first retrieval does not clear the confidence floor, it tries again
    with a synonym-expanded query and keeps whichever attempt was better.

    Both attempts are written to the trail. A judge can see the agent decide to
    look again, and see whether it helped.

WHAT GETS LOGGED
    Every step, in the DynamoDB single-table shape, including the steps that
    failed and the re-query that may not have helped. The trail is the product,
    not a debug artifact — so it records what happened rather than what we
    would prefer to show.
"""

import uuid

import backends
import coverage as coverage_mod
import intent as intent_mod
import precedent as precedent_mod
import settings
import standing as standing_mod
import trail as trail_mod

import config as os_config  # opensearch/config.py
import gate as gate_mod     # gate/gate.py


class Pipeline:
    def __init__(self, store=None):
        self.retriever = backends.build_retriever()
        self.classifier = backends.build_classifier()
        self.drafter = backends.build_drafter()
        self.trail = store or trail_mod.open_trail()
        self.precedents = precedent_mod.open_store()

    # -- the run ----------------------------------------------------------
    def run(self, text: str, ticket_id: str | None = None) -> dict:
        ticket_id = ticket_id or uuid.uuid4().hex[:8]
        steps: list[dict] = []
        n = 0

        def log(step: str, detail: dict) -> None:
            nonlocal n
            n += 1
            item = trail_mod.step_item(ticket_id, n, step, detail)
            self.trail.put(item)
            steps.append(item)

        meta = trail_mod.meta_item(ticket_id, text)
        self.trail.put(meta)

        # -- 1. classify --------------------------------------------------
        # The local classifier votes over a retrieval probe. That probe is a
        # real retrieval call, so it is named in the trail rather than hidden.
        # Intent rides along with the routing decision rather than taking a
        # stop of its own: both answer "what kind of ticket is this", and it
        # is a regex, not a step anyone waits for. It selects a paragraph in
        # the drafting prompt and nothing else — never a gate input. See
        # intent.py.
        how = intent_mod.explain(text)
        cls = self.classifier.classify(text, probe=self.retriever.search)
        category = cls["category"]
        log("classify", {
            "category": category,
            "backend": cls["backend"],
            "detail": cls.get("detail", ""),
            "votes": cls.get("votes"),
            "intent": how["intent"],
            "intent_why": how["why"],
            "intent_matched": how["matched"],
        })

        # -- 2. retrieve --------------------------------------------------
        found = self.retriever.search(text, category, settings.TOP_K)
        conf = found["confidence"]
        log("retrieve", {
            "backend": found["backend"],
            "best_cosine": conf["best_cosine"],
            "min_sim": conf["min_sim"],
            "confident": conf["confident"],
            "top": _summarize(found["results"]),
        })

        # -- 3. re-query (the one agentic step) ---------------------------
        requeried = False
        if not conf["confident"]:
            retry = self.retriever.search(text, category, settings.TOP_K, expand=True)
            requeried = True
            improved = retry["confidence"]["best_cosine"] > conf["best_cosine"]
            log("requery", {
                "reason": "best cosine %.4f below floor %.2f"
                          % (conf["best_cosine"], conf["min_sim"]),
                "strategy": "synonym-expanded query, re-embedded",
                "expanded_with": retry.get("expanded_with", [])[:12],
                "best_cosine": retry["confidence"]["best_cosine"],
                "improved": improved,
                "kept": "retry" if improved else "original",
                "top": _summarize(retry["results"]),
            })
            if improved:
                found, conf = retry, retry["confidence"]

        hits = found["results"]
        best = hits[0] if hits else None

        # -- 4. precedent: has another agent already answered this? -------
        # Compares the ticket, by meaning, against every past ticket. If the
        # same question was already answered and sent, the agent sees what
        # went out and which policy page it stood on; if today's page is a
        # DIFFERENT document, that is a conflicting answer and the gate (F7)
        # holds it. Excludes this ticket's own id so a re-run cannot be its
        # own precedent.
        prec = precedent_mod.lookup(self.precedents, text, best, exclude_id=ticket_id)
        log("precedent", precedent_mod.step_detail(prec))

        # -- 5. standing: the seller's own numbers against published limits -
        # Runs on every ticket and finds nothing in most of them. When the
        # seller has quoted a metric, the comparison is done here in Python
        # and handed to the drafter as a finished fact it may not recompute —
        # the model is bad at arithmetic and this arithmetic decides whether
        # someone keeps their shop.
        stand = standing_mod.check(text)
        log("standing", standing_mod.step_detail(stand))

        # -- 6. draft -----------------------------------------------------
        drafted = self.drafter.draft(text, best, context={
            "intent": how["intent"],
            "standing": stand,
        })
        grounding = _verify(drafted["draft"], best)
        log("draft", {
            "backend": drafted["backend"],
            "refused": drafted["refused"],
            "fallback_reason": drafted.get("fallback_reason"),
            "draft": drafted["draft"],
            "draft_quotes_policy": grounding["draft_quotes_policy"],
            "grounding_detail": grounding["detail"],
            "verified_quotes": grounding.get("verified", []),
            "urls_invented": grounding.get("urls_invented", []),
            "citation": _citation(best),
        })

        # -- 7. gate ------------------------------------------------------
        # has_citation is a fact about retrieval, not a hope: the passage must
        # carry a resolvable source_url or the reply cannot be checked.
        has_citation = bool(best and best.get("source_url"))
        ticket = gate_mod.build_ticket(
            text,
            category=category,
            confidence=conf["best_cosine"],
            has_citation=has_citation,
            draft_quotes_policy=grounding["draft_quotes_policy"],
            conflicts_with_precedent=prec["conflict"],
            standing_breach=stand["breach_selling"],
        )
        decision = gate_mod.decide(ticket, ticket_id=ticket_id)
        log("gate", {
            "decision": decision["decision"],
            "blocked_by": decision["blocked_by"],
            "reasons": decision["reasons"],
            "provenance": decision["provenance"],
            "topics": decision["topics"],
            "inputs": {
                "category": category,
                "retrieval_confidence": conf["best_cosine"],
                "has_citation": has_citation,
                "draft_quotes_policy": grounding["draft_quotes_policy"],
                "conflicts_with_precedent": prec["conflict"],
                "standing_breach": stand["breach_selling"],
            },
        })

        # -- close out ----------------------------------------------------
        outcome = dict(meta)
        outcome.update({
            "status": "complete",
            "decision": decision["decision"],
            "category": category,
            "best_cosine": conf["best_cosine"],
            "blocked_by": decision["blocked_by"],
            "requeried": requeried,
            "precedent_conflict": prec["conflict"],
            "steps": n,
            "completed_ms": trail_mod.now_ms(),
        })
        self.trail.put(outcome)

        # One row per run in the month's partition, so the coverage report is
        # a single query rather than a scan the function's role cannot make.
        # A failure here is never allowed to fail a ticket — the report is a
        # by-product, and the trail already holds everything it summarises.
        try:
            self.trail.put(coverage_mod.run_item(
                ticket_id, text, meta["created_ms"],
                category=category,
                decision=decision["decision"],
                blocked_by=decision["blocked_by"],
                confident=conf["confident"],
                has_citation=has_citation,
                draft_quotes_policy=grounding["draft_quotes_policy"],
                best_title=(best or {}).get("title"),
                best_cosine=conf["best_cosine"],
            ))
        except Exception as exc:                # noqa: BLE001
            outcome["coverage_error"] = "%s: %s" % (type(exc).__name__, str(exc)[:160])
            self.trail.put(outcome)

        # This ticket is now a precedent for the next one. Stored AFTER the
        # trail is complete, and never allowed to fail the run — the trail
        # row for the precedent step is amended with the error if it does.
        err = precedent_mod.remember_run(
            self.precedents, prec,
            ticket_id=ticket_id, text=text, created_ms=meta["created_ms"],
            category=category, decision=decision["decision"],
            blocked_by=decision["blocked_by"], best_hit=best, draft=drafted["draft"],
        )
        if err:
            row = next(s for s in steps if s["step"] == "precedent")
            row["remember_error"] = err
            self.trail.put(row)

        return {
            "ticket_id": ticket_id,
            "text": text,
            "category": category,
            "intent": how["intent"],
            "intent_why": how["why"],
            "standing": standing_mod.step_detail(stand),
            "confidence": conf,
            "requeried": requeried,
            "results": hits,
            "best": best,
            "draft": drafted["draft"],
            "grounding": grounding,
            "precedent": precedent_mod.public(prec),
            "decision": decision,
            "steps": steps,
            # Wall time of the run, so the console can say how long the decision took.
            "started_ms": meta["created_ms"],
            "completed_ms": outcome["completed_ms"],
        }


def _verify(draft: str, hit: dict | None) -> dict:
    import drafting

    if hit is None:
        return {"draft_quotes_policy": False, "detail": "nothing retrieved to ground against",
                "verified": []}
    # The title and heading path are shown to the model in the prompt, so a
    # quote lifted from them is grounded too — see verify_grounding's docstring.
    return drafting.verify_grounding(
        draft, hit.get("text", ""),
        extra_sources=[hit.get("title") or "", hit.get("heading_path") or ""],
        allowed_urls=[hit.get("source_url") or ""],
    )


def _summarize(results: list[dict]) -> list[dict]:
    """Trail rows keep the identity of every hit, not the passage text."""
    return [
        {
            "chunk_id": r["chunk_id"],
            "cosine": r["cosine"],
            "title": r["title"],
            "category": r["category"],
            "doc_type": r["doc_type"],
        }
        for r in results[:5]
    ]


def _citation(hit: dict | None) -> str | None:
    if not hit:
        return None
    label = hit.get("heading_path") or hit.get("title") or hit["chunk_id"]
    if hit.get("section_id"):
        label = "%s, %s" % (label, hit["section_id"])
    url = hit.get("source_url")
    return "%s%s" % (label, " — %s" % url if url else " [no source URL]")
