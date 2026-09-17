"""CLI for the pipeline.

    python pipeline/run.py "my payment is on hold and I don't know why"
    python pipeline/run.py --demo               # a small mixed batch
    python pipeline/run.py --queue              # the console's queue view
    python pipeline/run.py --show <ticket_id>   # replay one decision trail
    python pipeline/run.py --json "..."

Backends are chosen by environment variable — see settings.py. Defaults are all
local, so this runs with no AWS.
"""

import argparse
import json
import sys

# Windows consoles default to cp1252, which mangles the em dashes and curly
# quotes that come out of the corpus and the model. Same guard as
# test_final_pipeline.py — this is the script that runs on stage.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import settings  # noqa: E402,F401  (bootstraps sys.path before the rest import)
import trail as trail_mod
from orchestrator import Pipeline

# Fixtures for --demo. Deliberately NOT from filter_queries.tsv, which selected
# the corpus — reusing those would fit the demo to its own test set. These are
# working fixtures; the real demo tickets get sourced from the Seller Central
# forum threads in problem-and-sources.md.
DEMO = [
    "how do I change the handling time on my listings",
    "when will amazon transfer my money to my bank account",
    "my parcel is running late and the buyer is angry, and now amazon says my "
    "account could be deactivated if it happens again",
    "I received a trademark infringement complaint against one of my listings",
    "my listing is suppressed and not showing up in search",
]

BAR = "=" * 74


def render(out: dict, verbose: bool = True) -> None:
    d = out["decision"]
    conf = out["confidence"]

    print("\n" + BAR)
    print("TICKET %s" % out["ticket_id"])
    print(BAR)
    print(out["text"])

    print("\n  1. CLASSIFY   %s" % out["category"])
    print("  2. RETRIEVE   best cosine %.4f  floor %.2f  -> %s"
          % (conf["best_cosine"], conf["min_sim"],
             "confident" if conf["confident"] else "NOT confident"))
    if out["best"]:
        print("     %s" % out["best"]["title"])
        print("     %s" % (out["best"].get("heading_path") or ""))
        print("     %s" % (out["best"].get("source_url") or "[no source URL]"))
    if out["requeried"]:
        step = next(s for s in out["steps"] if s["step"] == "requery")
        print("  3. RE-QUERY   %s" % step["reason"])
        print("     kept the %s (best cosine %.4f, improved=%s)"
              % (step["kept"], step["best_cosine"], step["improved"]))
        if step.get("expanded_with"):
            print("     synonyms added: %s" % ", ".join(step["expanded_with"][:8]))

    g = out["grounding"]
    print("\n  4. DRAFT      grounding: %s" % g["detail"])
    if verbose:
        print()
        for line in out["draft"].splitlines():
            print("     | %s" % line)

    print("\n  5. GATE       %s" % d["decision"])
    print("     topics: %s" % (", ".join(d["topics"]) or "none"))
    for rid, reason in zip(d["blocked_by"], d["reasons"]):
        print("     %-26s %s" % (rid, reason))
    if d["allowed"]:
        print("     P0_routine_permit          no forbid rule matched")

    if d["provenance"]:
        for p in d["provenance"]:
            if p.get("source"):
                print("     why %s: %s" % (p["rule"], p["source_type"]))


def show_trail(ticket_id: str) -> None:
    store = trail_mod.open_trail()
    rows = store.query(ticket_id)
    if not rows:
        raise SystemExit("no trail for ticket %r" % ticket_id)
    print("\n%s\nDECISION TRAIL — %s\n%s" % (BAR, ticket_id, BAR))
    for r in rows:
        if r["sk"] == "META":
            print("\n[META] status=%s decision=%s backends=%s"
                  % (r.get("status"), r.get("decision", "-"), r.get("backends")))
            print("       %s" % r["text"])
            continue
        print("\n[%s] %s" % (r["sk"], r["step"]))
        for k, v in r.items():
            if k in ("pk", "sk", "ticket_id", "n", "step", "at_ms"):
                continue
            text = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
            if len(text) > 300:
                text = text[:300] + "..."
            print("       %-22s %s" % (k, text))


def show_queue() -> None:
    store = trail_mod.open_trail()
    if not hasattr(store, "tickets"):
        raise SystemExit("queue view needs the local jsonl trail")
    rows = store.tickets()
    if not rows:
        raise SystemExit("no tickets yet — run some first")
    print("\n%-10s %-11s %-10s %-8s %s" % ("TICKET", "DECISION", "CATEGORY", "COSINE", "TEXT"))
    print("-" * 96)
    auto = 0
    for r in rows:
        if r.get("decision") == "AUTO_SEND":
            auto += 1
        print("%-10s %-11s %-10s %-8s %s"
              % (r["ticket_id"], r.get("decision", "-"), r.get("category", "-"),
                 r.get("best_cosine", "-"),
                 (r["text"][:44] + "...") if len(r["text"]) > 44 else r["text"]))
    n = len(rows)
    print("-" * 96)
    print("%d tickets: %d auto-sent, %d escalated (%.0f%% held for a human)"
          % (n, auto, n - auto, 100 * (n - auto) / n))


def main() -> None:
    ap = argparse.ArgumentParser(description="Seller support triage pipeline")
    ap.add_argument("ticket", nargs="*")
    ap.add_argument("--demo", action="store_true", help="run the fixture batch")
    ap.add_argument("--queue", action="store_true", help="queue view of all tickets")
    ap.add_argument("--show", metavar="TICKET_ID", help="replay one decision trail")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="omit the draft body")
    args = ap.parse_args()

    if args.show:
        show_trail(args.show)
        return
    if args.queue:
        show_queue()
        return

    print(settings.summary())
    note = settings.banner()
    if note:
        print(note)

    pipe = Pipeline()

    if args.demo:
        outs = [pipe.run(t) for t in DEMO]
        for out in outs:
            render(out, verbose=not args.quiet)
        print("\n" + BAR)
        auto = sum(1 for o in outs if o["decision"]["allowed"])
        print("%d tickets: %d auto-sent, %d escalated" % (len(outs), auto, len(outs) - auto))
        print("trail written to %s" % settings.TRAIL_PATH.name)
        print(BAR)
        return

    if not args.ticket:
        ap.error("give a ticket, or --demo / --queue / --show")

    out = pipe.run(" ".join(args.ticket))
    if args.json:
        printable = {k: v for k, v in out.items() if k != "steps"}
        print(json.dumps(printable, indent=2, default=str))
    else:
        render(out, verbose=not args.quiet)
        print("\n  trail: python pipeline/run.py --show %s" % out["ticket_id"])


if __name__ == "__main__":
    main()
