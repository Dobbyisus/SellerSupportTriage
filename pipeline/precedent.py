"""Precedent: has another agent already answered this question?

    python pipeline/precedent.py --backfill     # fill the store from the trail
    python pipeline/precedent.py --self-test
    python pipeline/precedent.py "my listing is not showing up in search"

WHY THIS STAGE EXISTS
    The top complaint in the July 2026 Seller Forums thread is not slow
    answers — it is DIFFERENT answers: two representatives giving conflicting
    information on the same issue. A per-ticket decision trail records what
    was said; this stage READS it back before the next reply is drafted.

    Every incoming ticket is compared, by meaning, against every past ticket.
    If the same question was answered before, the agent sees what was sent and
    which policy page it stood on. If today's retrieval landed on a DIFFERENT
    policy page than the one already sent for the same question, that is a
    conflicting answer in the making, and Cedar rule F7 holds it for a person
    rather than letting two contradictory replies go out under Amazon's name.

WHAT COUNTS AS "THE SAME QUESTION"
    Ticket-to-ticket cosine, same embedding model as retrieval. Two thresholds
    in opensearch/config.py: PRECEDENT_SIM (similar enough to show) and the
    stricter PRECEDENT_SAME (the same question — above the closest any two
    DISTINCT filter tickets get). Only a same-question precedent can raise a
    conflict, because a merely similar ticket may legitimately be about a
    different thing.

WHAT COUNTS AS "A DIFFERENT ANSWER"
    A different policy DOCUMENT, compared by doc_id. Two sections of the same
    help page are one answer. And only replies that were actually SENT
    (AUTO_SEND) set a precedent — a ticket that was escalated was answered by a
    person, and the system never saw what they wrote.

THE STORE FOLLOWS THE RETRIEVER
    OpenSearch when policy search is on OpenSearch (a second index on the same
    domain, `seller-tickets`); a local JSONL file when retrieval is in-process.
    Never DynamoDB: the function's role holds PutItem and Query, and finding
    the nearest past ticket is neither.

A STORE FAILURE NEVER FAILS THE TICKET
    lookup() and remember_run() catch everything and record the error in the
    trail. The gate then sees conflicts_with_precedent=false, which is the
    conservative reading of "we could not check" only in the sense that the
    other six rules still apply in full; the step row says plainly that the
    check did not run.
"""

import argparse
import json
import sys
from pathlib import Path

import settings

import config as os_config  # opensearch/config.py


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def doc_identity(hit: dict | None) -> str | None:
    """The document a hit belongs to, for the conflict comparison."""
    if not hit:
        return None
    if hit.get("doc_id"):
        return hit["doc_id"]
    cid = hit.get("chunk_id") or ""
    if "#" in cid:
        return cid.split("#", 1)[0]
    return hit.get("source_url") or hit.get("title") or None


def policy_of(hit: dict | None) -> dict:
    if not hit:
        return {"title": "", "path": "", "url": ""}
    return {
        "title": hit.get("title") or "",
        "path": hit.get("heading_path") or "",
        "url": hit.get("source_url") or "",
    }


def make_record(ticket_id: str, text: str, created_ms: int, category: str,
                decision: str, blocked_by: list[str], best_hit: dict | None,
                draft: str, embedding: list[float]) -> dict:
    """One past ticket, in the shape both stores hold."""
    pol = policy_of(best_hit)
    return {
        "ticket_id": ticket_id,
        "text": text,
        "created_ms": int(created_ms),
        "category": category,
        "decision": decision,
        "blocked_by": list(blocked_by or []),
        "sent": decision == "AUTO_SEND",
        "doc_id": doc_identity(best_hit),
        "policy_title": pol["title"],
        "policy_path": pol["path"],
        "policy_url": pol["url"],
        "draft": draft or "",
        "embedding": [float(x) for x in embedding],
    }


def embed(text: str) -> list[float]:
    import search as search_mod  # opensearch/search.py

    return search_mod.embed_query(text)


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------


class JsonlPrecedents:
    """Append-only local file, exact cosine in numpy. No AWS."""

    name = "jsonl"

    def __init__(self, path: Path | None = None):
        self.path = Path(path or settings.PRECEDENTS_PATH)
        self._rows: dict[str, dict] = {}
        self._stamp = None
        self._load()

    def _load(self) -> None:
        """(Re)read the file when another process has appended to it.

        The CLI, the test console and a local server all share one file.
        Reading it once at start-up meant a ticket run from the CLI was
        invisible to a server already running — which looks exactly like the
        check not working. Cheap: one stat per lookup, a re-read only on change.
        """
        try:
            st = self.path.stat()
            stamp = (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            stamp = None
        if stamp == self._stamp:
            return
        self._rows = {}
        if stamp is not None:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self._rows[r["ticket_id"]] = r  # last write wins, like put_item
        self._stamp = stamp

    def __len__(self) -> int:
        self._load()
        return len(self._rows)

    def remember(self, rec: dict) -> None:
        self._load()
        self._rows[rec["ticket_id"]] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        st = self.path.stat()
        self._stamp = (st.st_mtime_ns, st.st_size)

    def find(self, vec: list[float], k: int, exclude_id: str | None = None) -> list[dict]:
        import numpy as np

        self._load()
        rows = [r for r in self._rows.values() if r["ticket_id"] != exclude_id]
        if not rows:
            return []
        m = np.array([r["embedding"] for r in rows], dtype="float32")
        q = np.array(vec, dtype="float32")
        sims = m @ q
        order = np.argsort(-sims)[:k]
        out = []
        for i in order:
            r = dict(rows[int(i)])
            r.pop("embedding", None)
            r["similarity"] = round(float(sims[int(i)]), 4)
            out.append(r)
        return out


class OpenSearchPrecedents:
    """The `seller-tickets` index on the same domain as the policy corpus."""

    name = "opensearch"

    def __init__(self, os_client=None):
        import client

        self._client = os_client or client.connect()
        self.index = os_config.TICKETS_INDEX

    def remember(self, rec: dict) -> None:
        # POST, not PUT. opensearch-py's index() issues a PUT when given an id,
        # and the function's role has no es:ESHttpPut — it would 403 in
        # production and work on a laptop. refresh=true so the very next
        # ticket can find this one.
        self._client.transport.perform_request(
            "POST",
            "/%s/_doc/%s?refresh=true" % (self.index, rec["ticket_id"]),
            body=rec,
        )

    def find(self, vec: list[float], k: int, exclude_id: str | None = None) -> list[dict]:
        must_not = [{"term": {"ticket_id": exclude_id}}] if exclude_id else []
        body = {
            "size": k,
            "_source": {"excludes": ["embedding"]},
            "query": {
                "script_score": {
                    "query": {"bool": {"must_not": must_not}},
                    # Same script and the same +1.0 offset as search.py: the
                    # raw cosine, engine-independent, on the calibrated scale.
                    "script": {
                        "source": "1.0 + cosineSimilarity(params.query_value, doc['embedding'])",
                        "params": {"query_value": vec},
                    },
                }
            },
        }
        resp = self._client.transport.perform_request(
            "POST", "/%s/_search" % self.index, body=body
        )
        out = []
        for h in resp["hits"]["hits"]:
            r = dict(h["_source"])
            r["similarity"] = round(float(h["_score"]) - 1.0, 4)
            out.append(r)
        return out


def open_store():
    if settings.PRECEDENT == "opensearch":
        return OpenSearchPrecedents()
    if settings.PRECEDENT == "jsonl":
        return JsonlPrecedents()
    return None


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def _public(r: dict) -> dict:
    return {
        "ticket_id": r.get("ticket_id"),
        "text": r.get("text", ""),
        "created_ms": int(r.get("created_ms") or 0),
        "category": r.get("category"),
        "decision": r.get("decision"),
        "blocked_by": list(r.get("blocked_by") or []),
        "sent": bool(r.get("sent")),
        "similarity": r.get("similarity"),
        "same_question": bool((r.get("similarity") or 0.0) >= os_config.PRECEDENT_SAME),
        "doc_id": r.get("doc_id"),
        "policy": {
            "title": r.get("policy_title") or "",
            "path": r.get("policy_path") or "",
            "url": r.get("policy_url") or "",
        },
    }


def assess(matches: list[dict], best_hit: dict | None) -> dict:
    """Turn nearest past tickets into the facts the trail and the gate use.

    Pure — no I/O — so the self-test can exercise every branch.
    """
    similar = [_public(r) for r in matches
               if (r.get("similarity") or 0.0) >= os_config.PRECEDENT_SIM]
    same = [s for s in similar if s["same_question"]]
    sent = [s for s in same if s["sent"]]
    last = max(sent, key=lambda s: s["created_ms"]) if sent else None

    current_doc = doc_identity(best_hit)
    consistent = None
    if last is not None:
        consistent = bool(current_doc and last["doc_id"] and current_doc == last["doc_id"])

    if last is not None:
        src = next(r for r in matches if r.get("ticket_id") == last["ticket_id"])
        last = dict(last, draft=src.get("draft") or "")

    return {
        "checked": True,
        "similar": similar,
        "same_question": len(same),
        "last_sent": last,
        "current_doc_id": current_doc,
        "consistent": consistent,
        "conflict": consistent is False,
        "thresholds": {"similar": os_config.PRECEDENT_SIM, "same": os_config.PRECEDENT_SAME},
    }


def lookup(store, text: str, best_hit: dict | None, exclude_id: str | None = None) -> dict:
    """The stage. Never raises — a store failure is recorded, not propagated."""
    if store is None:
        return {"checked": False, "backend": "off", "reason": "precedent check is off",
                "similar": [], "same_question": 0, "last_sent": None,
                "current_doc_id": doc_identity(best_hit), "consistent": None,
                "conflict": False, "vector": None}
    try:
        vec = embed(text)
        matches = store.find(vec, os_config.PRECEDENT_K, exclude_id=exclude_id)
        out = assess(matches, best_hit)
        out["backend"] = store.name
        out["vector"] = vec
        return out
    except Exception as exc:  # noqa: BLE001 — see the module docstring
        return {"checked": False, "backend": getattr(store, "name", "?"),
                "reason": "%s: %s" % (type(exc).__name__, str(exc)[:160]),
                "similar": [], "same_question": 0, "last_sent": None,
                "current_doc_id": doc_identity(best_hit), "consistent": None,
                "conflict": False, "vector": None}


def step_detail(prec: dict) -> dict:
    """What the trail row holds. Compact: identities and outcomes, not drafts."""
    last = prec.get("last_sent")
    return {
        "backend": prec.get("backend"),
        "checked": prec.get("checked", False),
        "reason": prec.get("reason"),
        "similar": [
            {k: s[k] for k in ("ticket_id", "similarity", "same_question", "decision",
                               "sent", "doc_id", "created_ms")}
            for s in prec.get("similar", [])
        ],
        "same_question": prec.get("same_question", 0),
        "last_sent": (
            {k: last[k] for k in ("ticket_id", "created_ms", "doc_id", "policy")}
            if last else None
        ),
        "current_doc_id": prec.get("current_doc_id"),
        "consistent": prec.get("consistent"),
        "conflict": prec.get("conflict", False),
    }


def public(prec: dict) -> dict:
    """What the API returns: everything but the vector."""
    return {k: v for k, v in prec.items() if k != "vector"}


def remember_run(store, prec: dict, *, ticket_id: str, text: str, created_ms: int,
                 category: str, decision: str, blocked_by: list[str],
                 best_hit: dict | None, draft: str) -> str | None:
    """Store this ticket as a precedent for the next one. Returns an error or None."""
    if store is None:
        return None
    try:
        vec = prec.get("vector") or embed(text)
        store.remember(make_record(ticket_id, text, created_ms, category, decision,
                                   blocked_by, best_hit, draft, vec))
        return None
    except Exception as exc:  # noqa: BLE001
        return "%s: %s" % (type(exc).__name__, str(exc)[:160])


# ---------------------------------------------------------------------------
# Backfill from the decision trail
# ---------------------------------------------------------------------------


def _plain(v):
    """DynamoDB hands back Decimal; make rows JSON-plain."""
    from decimal import Decimal

    if isinstance(v, Decimal):
        return int(v) if v == v.to_integral_value() else float(v)
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_plain(x) for x in v]
    return v


def _hit_from_trail(rows: list[dict]) -> dict | None:
    """Rebuild the kept passage's identity from a ticket's step rows.

    The trail keeps identities, not passages: retrieve/requery rows carry
    chunk_id and title, and the draft row's citation carries the heading path
    and the URL. That is enough for doc_id, title, path and url.
    """
    by = {r.get("step"): r for r in rows if r.get("sk", "").startswith("STEP#")}
    ret, rq, dr = by.get("retrieve"), by.get("requery"), by.get("draft")
    top = None
    if rq and rq.get("kept") == "retry" and rq.get("top"):
        top = rq["top"][0]
    elif ret and ret.get("top"):
        top = ret["top"][0]
    if not top:
        return None
    path, url = "", ""
    cite = (dr or {}).get("citation") or ""
    if " — " in cite:
        path, url = cite.rsplit(" — ", 1)
    elif cite and not cite.endswith("[no source URL]"):
        path = cite
    return {
        "chunk_id": top.get("chunk_id"),
        "doc_id": (top.get("chunk_id") or "").split("#", 1)[0] or None,
        "title": top.get("title"),
        "heading_path": path,
        "source_url": url,
    }


def backfill(store, trail) -> tuple[int, int]:
    metas = [_plain(m) for m in trail.tickets()]
    done = skipped = 0
    for m in metas:
        if m.get("status") != "complete" or not m.get("decision"):
            skipped += 1
            continue
        rows = [_plain(r) for r in trail.query(m["ticket_id"])]
        hit = _hit_from_trail(rows)
        draft_row = next((r for r in rows if r.get("step") == "draft"), {})
        rec = make_record(
            m["ticket_id"], m["text"], m.get("created_ms") or 0,
            m.get("category") or "", m["decision"], m.get("blocked_by") or [],
            hit, draft_row.get("draft") or "", embed(m["text"]),
        )
        store.remember(rec)
        done += 1
        print("  %-9s %-10s %-9s %s" % (m["ticket_id"], m["decision"],
                                        (hit or {}).get("doc_id") or "-",
                                        m["text"][:52]))
    return done, skipped


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def self_test() -> int:
    import tempfile

    failed = 0

    def check(ok: bool, label: str) -> None:
        nonlocal failed
        print("%s %s" % ("ok  " if ok else "FAIL", label))
        if not ok:
            failed += 1

    print("precedent — store and conflict logic\n")

    a = "how do I change the handling time on my listings"
    b = "how can I edit the handling time for my products"
    c = "when will amazon transfer my money to my bank account"
    va, vb, vc = embed(a), embed(b), embed(c)
    hit_x = {"chunk_id": "help_G202167920#0001", "doc_id": "help_G202167920",
             "title": "Manage your delivery time", "heading_path": "Modify", "source_url": "u1"}
    hit_y = {"chunk_id": "help_G14911#0337", "doc_id": "help_G14911",
             "title": "When will I be paid?", "heading_path": "", "source_url": "u2"}

    with tempfile.TemporaryDirectory() as d:
        store = JsonlPrecedents(Path(d) / "p.jsonl")
        check(store.find(va, 3) == [], "empty store finds nothing")

        store.remember(make_record("t1", a, 1000, "shipping", "AUTO_SEND", [], hit_x, "sent A", va))
        store.remember(make_record("t3", c, 3000, "payments", "AUTO_SEND", [], hit_y, "sent C", vc))
        check(len(JsonlPrecedents(Path(d) / "p.jsonl")) == 2, "records survive a reload")

        m = store.find(vb, 3, exclude_id="t9")
        check(m and m[0]["ticket_id"] == "t1", "nearest past ticket is the paraphrase")
        check(m[0]["similarity"] >= os_config.PRECEDENT_SAME,
              "a paraphrase clears PRECEDENT_SAME (%.3f)" % m[0]["similarity"])
        check(all("embedding" not in r for r in m), "vectors never leave the store")

        # same question, same document -> consistent, no conflict
        out = assess(m, hit_x)
        check(out["last_sent"] and out["last_sent"]["ticket_id"] == "t1", "last sent reply is found")
        check(out["consistent"] is True and not out["conflict"], "same document -> consistent")
        check(out["last_sent"]["draft"] == "sent A", "what was sent travels with it")

        # same question, DIFFERENT document -> conflict
        out = assess(m, hit_y)
        check(out["conflict"] is True, "different document for the same question -> conflict")

        # same document, different chunk is still the same answer
        other_chunk = dict(hit_x, chunk_id="help_G202167920#0002")
        check(assess(m, other_chunk)["consistent"] is True, "another chunk of the same page is not a conflict")

        # an escalated precedent set no answer, so it cannot conflict
        store2 = JsonlPrecedents(Path(d) / "q.jsonl")
        store2.remember(make_record("t5", a, 1000, "shipping", "ESCALATE", ["F6_ungrounded"], hit_x, "held", va))
        out = assess(store2.find(vb, 3), hit_y)
        check(out["same_question"] == 1 and out["last_sent"] is None and not out["conflict"],
              "an escalated precedent is shown but sets no answer")

        # a merely similar ticket (below PRECEDENT_SAME) never conflicts
        low = [dict(make_record("t7", a, 1, "shipping", "AUTO_SEND", [], hit_x, "x", va),
                    similarity=os_config.PRECEDENT_SIM + 0.01)]
        out = assess(low, hit_y)
        check(len(out["similar"]) == 1 and out["same_question"] == 0 and not out["conflict"],
              "similar-but-not-same is shown, never a conflict")

        # the running ticket never matches itself
        check(all(r["ticket_id"] != "t1" for r in store.find(va, 3, exclude_id="t1")),
              "a ticket is excluded from its own lookup")

        # a broken store is reported, not raised
        class Broken:
            name = "broken"

            def find(self, *a, **k):
                raise RuntimeError("boom")

        out = lookup(Broken(), a, hit_x)
        check(out["checked"] is False and not out["conflict"] and "boom" in out["reason"],
              "a store failure is recorded and never raises")
        check(lookup(None, a, hit_x)["backend"] == "off", "PIPELINE_PRECEDENT=off is honest in the trail")

    print("\n%s" % ("all checks passed." if not failed else "%d FAILED" % failed))
    return failed


def main() -> None:
    ap = argparse.ArgumentParser(description="Precedent check over past tickets")
    ap.add_argument("ticket", nargs="*")
    ap.add_argument("--backfill", action="store_true",
                    help="fill the precedent store from the decision trail")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(1 if self_test() else 0)

    if args.backfill:
        import trail as trail_mod

        store = open_store()
        if store is None:
            raise SystemExit("PIPELINE_PRECEDENT=off — nothing to fill")
        trail = trail_mod.open_trail()
        print("trail=%s -> precedents=%s\n" % (settings.TRAIL, store.name))
        done, skipped = backfill(store, trail)
        print("\n%d tickets stored, %d skipped (incomplete)" % (done, skipped))
        return

    if not args.ticket:
        ap.error("give a ticket, or --backfill / --self-test")

    text = " ".join(args.ticket)
    store = open_store()
    out = lookup(store, text, None)
    print("\nticket   : %s" % text)
    print("store    : %s%s" % (out["backend"], "" if out["checked"] else "  (NOT CHECKED: %s)" % out.get("reason")))
    if not out["similar"]:
        print("precedent: none — first time this has been asked")
    for s in out["similar"]:
        print("  %.3f %-13s %-9s %s" % (s["similarity"], "SAME QUESTION" if s["same_question"] else "similar",
                                        "sent" if s["sent"] else "held", s["text"][:60]))
        if s["policy"]["title"]:
            print("        built on: %s" % s["policy"]["title"])


if __name__ == "__main__":
    main()
