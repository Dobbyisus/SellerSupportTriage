"""What the corpus could not answer — one row per run, one query per month.

    python pipeline/coverage.py                 # this month
    python pipeline/coverage.py --month 2026-09
    python pipeline/coverage.py --json

WHY THIS EXISTS
    The system already refuses to answer when no policy page is a close enough
    match, and writes down why. Those refusals are the most useful thing in
    the trail and nobody was reading them. Added up over a month they say:
    sellers asked about this fourteen times and we have no page that covers
    it. That turns a pile of escalations from a cost into a list of documents
    worth writing.

    It is also the honest answer to "what can this thing do?" — a system that
    can show the shape of its own ignorance is worth more than one that always
    has an answer.

THE ROW SHAPE, AND WHY IT IS NOT AN AGGREGATE
    pk = COVERAGE#<yyyy-mm>   sk = RUN#<ms>#<ticket_id>

    One small row per run, appended, never updated. The obvious alternative —
    keeping a running total per topic and incrementing it — needs a
    read-modify-write on every ticket, and the function's IAM role holds
    PutItem and Query and nothing else: no UpdateItem, no atomic counter, no
    Scan. Appending facts and adding them up at read time needs only the two
    verbs we have, and it cannot lose a count to a race.

    Every run is recorded, not just the failures. A gap count with no
    denominator ("14 gaps") means nothing; "14 of 90" is a number a person can
    act on.

WHAT COUNTS AS A GAP
    Not every escalation. A suspension ticket escalates because a human must
    see it, and that is the system working — writing a policy page would not
    change it. A gap is specifically a KNOWLEDGE failure: retrieval never
    cleared the confidence floor, or nothing came back with a citation, or the
    draft could not be tied to the passage. That is F6's territory, and F6 is
    the rule that reads those three facts.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import settings  # noqa: F401  (bootstraps gate/ and opensearch/ onto sys.path)
import topics as topics_mod  # gate/topics.py

# Enough of the question to recognise it in a report, not the whole ticket.
# The full text is already on the META row; this copy exists so the monthly
# report is one query instead of one query plus a fetch per gap.
EXCERPT = 160


def month_of(ms: int | None = None) -> str:
    when = datetime.fromtimestamp((ms or int(time.time() * 1000)) / 1000, timezone.utc)
    return when.strftime("%Y-%m")


def subject_of(text: str, category: str) -> str:
    """What the seller was asking about, in one label.

    Topics first, because they are the specific thing ("payment_hold") and
    they come from the seller's own words. Category second, because it is
    always present. Neither is a model's opinion about the subject matter.
    """
    found = topics_mod.extract(text)
    return found[0] if found else category


def run_item(ticket_id: str, text: str, created_ms: int, *, category: str,
             decision: str, blocked_by: list[str], confident: bool,
             has_citation: bool, draft_quotes_policy: bool,
             best_title: str | None, best_cosine: float) -> dict:
    """The row for one run. A gap is a knowledge failure, not any escalation."""
    reasons = []
    if not confident:
        reasons.append("no page matched closely enough")
    if not has_citation:
        reasons.append("nothing retrieved carried a source link")
    if not draft_quotes_policy:
        reasons.append("the draft could not be tied to the passage")

    return {
        "pk": "COVERAGE#%s" % month_of(created_ms),
        "sk": "RUN#%013d#%s" % (created_ms, ticket_id),
        "ticket_id": ticket_id,
        "at_ms": created_ms,
        "gap": bool(reasons),
        "reasons": reasons,
        "subject": subject_of(text, category),
        "category": category,
        "decision": decision,
        "blocked_by": blocked_by,
        "best_title": best_title,
        "best_cosine": best_cosine,
        "excerpt": " ".join(text.split())[:EXCERPT],
    }


def report(store, month: str | None = None, examples: int = 3) -> dict:
    """One query, aggregated in memory. No scan, no secondary index."""
    month = month or month_of()
    rows = [r for r in store.query_pk("COVERAGE#%s" % month) if r.get("sk", "").startswith("RUN#")]
    gaps = [r for r in rows if r.get("gap")]

    buckets: dict[str, dict] = {}
    for r in gaps:
        b = buckets.setdefault(r.get("subject") or "unknown", {
            "subject": r.get("subject") or "unknown", "gaps": 0, "examples": [],
        })
        b["gaps"] += 1
        if len(b["examples"]) < examples:
            b["examples"].append({
                "text": r.get("excerpt", ""),
                "at_ms": int(r.get("at_ms", 0)),
                "best_title": r.get("best_title"),
                "best_cosine": float(r.get("best_cosine") or 0),
                "reasons": list(r.get("reasons") or []),
            })

    # What retrieval kept reaching for when it came up short. A page that
    # appears here repeatedly is the nearest thing we have to the missing one,
    # which is a useful hint about what to write.
    near: dict[str, int] = {}
    for r in gaps:
        if r.get("best_title"):
            near[r["best_title"]] = near.get(r["best_title"], 0) + 1

    return {
        "month": month,
        "runs": len(rows),
        "gaps": len(gaps),
        "answered": len(rows) - len(gaps),
        "gap_rate": round(len(gaps) / len(rows), 4) if rows else 0.0,
        "subjects": sorted(buckets.values(), key=lambda b: (-b["gaps"], b["subject"])),
        "nearest_pages": [
            {"title": t, "times": n}
            for t, n in sorted(near.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        ],
    }


def render(rep: dict) -> None:
    print("coverage — %s" % rep["month"])
    print("%d run(s): %d answered from a policy page, %d could not be"
          % (rep["runs"], rep["answered"], rep["gaps"]))
    if not rep["runs"]:
        print("\nNothing has run this month.")
        return
    if not rep["gaps"]:
        print("\nEvery ticket this month found a page to stand on.")
        return

    print("\nwhat we could not answer")
    for b in rep["subjects"]:
        print("  %-24s %d" % (b["subject"].replace("_", " "), b["gaps"]))
        for ex in b["examples"]:
            print("      %s" % ex["text"])
            print("        nearest: %s (%.2f) — %s"
                  % (ex["best_title"] or "nothing", ex["best_cosine"],
                     "; ".join(ex["reasons"])))

    if rep["nearest_pages"]:
        print("\nwhat retrieval kept reaching for instead")
        for p in rep["nearest_pages"]:
            print("  %-52s %d" % (p["title"][:52], p["times"]))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    import trail as trail_mod

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--month", help="yyyy-mm (default: this month)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rep = report(trail_mod.open_trail(), args.month)
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
    else:
        render(rep)


if __name__ == "__main__":
    main()
