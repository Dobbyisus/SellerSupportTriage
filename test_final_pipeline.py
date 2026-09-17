"""Interactive end-to-end test console for the Seller Support Triage pipeline.

    python test_final_pipeline.py

Loads the embedding model, the corpus, the Cedar policies and the drafter ONCE,
then stays open. Type a ticket, watch all five stages run, see the timings.
Type `exit` to leave — nothing else ends the session, including errors.

Commands
    <any text>      run it through the pipeline as a ticket
    demo            run the five fixture tickets
    queue           the console's queue view of everything run so far
    show <id>       replay one full decision trail
    last            re-print the previous result in full
    drafter <name>  switch live: mantle | template
    model <id>      switch the bedrock-mantle model live
    check           re-run the startup health checks
    stats           session totals
    help            this list
    exit            quit
"""

import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline"))

# Windows consoles default to cp1252, which mangles the em dashes and curly
# quotes that come out of the corpus and the model.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import settings          # noqa: E402  (also bootstraps gate/ and opensearch/ onto sys.path)
import backends          # noqa: E402
import trail as trail_mod  # noqa: E402
from orchestrator import Pipeline  # noqa: E402

BAR = "=" * 78
SUB = "-" * 78

DEMO = [
    "how do I change the handling time on my listings",
    "when will amazon transfer my money to my bank account",
    "my parcel is running late and the buyer is angry, and now amazon says my "
    "account could be deactivated if it happens again",
    "I received a trademark infringement complaint against one of my listings",
    "my listing is suppressed and not showing up in search",
]


# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------


def health_checks() -> bool:
    """Prove each dependency works before the first ticket, not during it."""
    ok = True

    def check(label, fn):
        nonlocal ok
        t0 = time.time()
        try:
            detail = fn()
            print("  [ ok ] %-34s %-38s %5.2fs" % (label, detail, time.time() - t0))
        except Exception as exc:
            ok = False
            print("  [FAIL] %-34s %s" % (label, str(exc)[:64]))

    print("\nHEALTH CHECKS")
    print(SUB)

    def corpus():
        import json
        n = sum(1 for line in (ROOT / "chunks.jsonl").open(encoding="utf-8") if line.strip())
        first = json.loads((ROOT / "chunks.jsonl").open(encoding="utf-8").readline())
        return "%d chunks, %d dims" % (n, len(first["embedding"]))

    def precedent():
        import precedent as precedent_mod
        store = precedent_mod.open_store()
        if store is None:
            return "off"
        if hasattr(store, "__len__"):
            return "%s, %d past tickets" % (store.name, len(store))
        n = store._client.count(index=store.index)["count"]
        return "%s/%s, %d past tickets" % (store.name, store.index, n)

    def cedar():
        import gate as gate_mod
        _, meta = gate_mod.load_policies()
        forbid = [m for m in meta.values() if m["id"].startswith("F")]
        return "%d rules (%d forbid), floor %.2f" % (
            len(meta), len(forbid), gate_mod.policy_floor())

    def bedrock():
        if settings.DRAFTER != "mantle":
            return "skipped (drafter=%s)" % settings.DRAFTER
        import json
        import boto3
        import requests
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        region = settings.MANTLE_REGION
        url = "https://bedrock-mantle.%s.api.aws/v1/chat/completions" % region
        creds = boto3.Session(region_name=region).get_credentials().get_frozen_credentials()
        body = {"model": settings.MANTLE_MODEL, "max_tokens": 4,
                "messages": [{"role": "user", "content": "Say OK"}]}
        req = AWSRequest(method="POST", url=url, data=json.dumps(body),
                         headers={"Content-Type": "application/json"})
        SigV4Auth(creds, "bedrock-mantle", region).add_auth(req)
        r = requests.post(url, data=req.body, headers=dict(req.headers), timeout=30)
        if r.status_code != 200:
            raise RuntimeError("%s %s" % (r.status_code, r.text[:80]))
        return settings.MANTLE_MODEL.split(".")[-1][:34]

    check("corpus", corpus)
    check("cedar policies", cedar)
    check("precedent store", precedent)
    check("bedrock-mantle", bedrock)
    print(SUB)
    if not ok:
        print("  Some checks failed. The session still starts — failures degrade")
        print("  gracefully (the drafter falls back to the template).")
    return ok


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def stage_times(out: dict, start_ms: int | None = None) -> dict:
    """Per-stage durations, read back off the trail's own timestamps.

    A step row is written AFTER its work finishes, so stage N's duration is
    at_ms[N] - at_ms[N-1] and belongs to N — not to N-1. The first stage
    measures from the wall-clock start the caller passes in.
    """
    times = {}
    prev = start_ms if start_ms is not None else out["steps"][0]["at_ms"]
    for s in out["steps"]:
        times[s["step"]] = max(0.0, (s["at_ms"] - prev) / 1000.0)
        prev = s["at_ms"]
    return times


def render(out: dict, elapsed: float, show_draft: bool = True,
           start_ms: int | None = None) -> None:
    d = out["decision"]
    conf = out["confidence"]
    t = stage_times(out, start_ms)

    print("\n" + BAR)
    print("TICKET %s     %.2fs total" % (out["ticket_id"], elapsed))
    print(BAR)
    print(out["text"])
    print()

    print("  1. CLASSIFY   %-12s %28s %6s"
          % (out["category"], "(no model - category vote)", _t(t.get("classify"))))

    verdict = "confident" if conf["confident"] else "NOT confident"
    print("  2. RETRIEVE   cosine %.4f  floor %.2f  -> %-14s %6s"
          % (conf["best_cosine"], conf["min_sim"], verdict, _t(t.get("retrieve"))))
    if out["best"]:
        b = out["best"]
        print("                %s" % (b.get("title") or ""))
        print("                %s" % (b.get("heading_path") or "")[:70])
        print("                %s" % (b.get("source_url") or "[no source URL]"))

    if out["requeried"]:
        step = next(s for s in out["steps"] if s["step"] == "requery")
        print("  3. RE-QUERY   %s" % step["reason"])
        print("                kept the %-8s cosine %.4f  improved=%-5s %6s"
              % (step["kept"], step["best_cosine"], step["improved"], _t(t.get("requery"))))
        if step.get("expanded_with"):
            print("                + %s" % ", ".join(step["expanded_with"][:10]))

    pr = out.get("precedent") or {}
    if not pr.get("checked"):
        print("  4. PRECEDENT  %-46s %6s"
              % (("not checked: %s" % (pr.get("reason") or pr.get("backend")))[:46], _t(t.get("precedent"))))
    elif not pr["similar"]:
        print("  4. PRECEDENT  %-46s %6s" % ("none - first time asked", _t(t.get("precedent"))))
    else:
        last = pr.get("last_sent")
        verdict = ("CONFLICT - sent before on a different page" if pr["conflict"]
                   else "consistent with what was sent" if last
                   else "seen before, nothing sent yet")
        print("  4. PRECEDENT  %-46s %6s" % (verdict[:46], _t(t.get("precedent"))))
        for s in pr["similar"][:3]:
            print("                %.3f %-13s %-4s %s" % (s["similarity"],
                  "same question" if s["same_question"] else "similar",
                  "sent" if s["sent"] else "held", s["text"][:44]))
        if last:
            print("                last sent stood on: %s" % (last["policy"]["title"] or last["doc_id"])[:50])

    g = out["grounding"]
    draft_step = next((s for s in out["steps"] if s["step"] == "draft"), {})
    print("  5. DRAFT      %-46s %6s" % (draft_step.get("backend", "?")[:46], _t(t.get("draft"))))
    print("                grounding: %s" % g["detail"])
    if draft_step.get("fallback_reason"):
        print("                FELL BACK: %s" % draft_step["fallback_reason"][:60])

    if show_draft:
        print()
        for line in out["draft"].splitlines():
            print("      | %s" % line)

    print("\n  6. GATE       %s" % d["decision"])
    print("                topics: %s" % (", ".join(d["topics"]) or "none"))
    for rid, reason in zip(d["blocked_by"], d["reasons"]):
        print("                %-26s %s" % (rid, reason[:44]))
    if d["allowed"]:
        print("                P0_routine_permit          no forbid rule matched")
    print("\n  trail: show %s" % out["ticket_id"])


def _t(seconds) -> str:
    return "%5.2fs" % seconds if seconds is not None else "     -"


def show_trail(ticket_id: str) -> None:
    import json

    rows = trail_mod.open_trail().query(ticket_id)
    if not rows:
        print("  no trail for %r" % ticket_id)
        return
    print("\n" + BAR)
    print("DECISION TRAIL  %s" % ticket_id)
    print(BAR)
    for r in rows:
        if r["sk"] == "META":
            print("\n[META]  status=%s  decision=%s" % (r.get("status"), r.get("decision", "-")))
            print("        backends: %s" % r.get("backends"))
            print("        %s" % r["text"])
            continue
        print("\n[%s]  %s" % (r["sk"], r["step"]))
        for k, v in r.items():
            if k in ("pk", "sk", "ticket_id", "n", "step", "at_ms"):
                continue
            text = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
            if len(text) > 260:
                text = text[:260] + "..."
            print("        %-20s %s" % (k, text))


def show_queue() -> None:
    store = trail_mod.open_trail()
    rows = store.tickets() if hasattr(store, "tickets") else []
    if not rows:
        print("  nothing run yet")
        return
    print("\n%-10s %-11s %-10s %-8s %s" % ("TICKET", "DECISION", "CATEGORY", "COSINE", "TEXT"))
    print(SUB)
    auto = sum(1 for r in rows if r.get("decision") == "AUTO_SEND")
    for r in rows:
        text = r["text"]
        print("%-10s %-11s %-10s %-8s %s"
              % (r["ticket_id"], r.get("decision", "-"), r.get("category", "-"),
                 r.get("best_cosine", "-"),
                 text[:40] + "..." if len(text) > 40 else text))
    print(SUB)
    print("%d tickets: %d auto-sent, %d escalated (%.0f%% held for a human)"
          % (len(rows), auto, len(rows) - auto, 100 * (len(rows) - auto) / len(rows)))


HELP = """
  <any text>      run it through the pipeline as a ticket
  demo            run the five fixture tickets
  queue           queue view of everything run this session
  show <id>       replay one full decision trail
  last            re-print the previous result in full
  drafter <name>  switch live: mantle | template
  model <id>      switch the bedrock-mantle model live
  check           re-run the health checks
  stats           session totals
  help            this list
  exit            quit
"""


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    print(BAR)
    print("SELLER SUPPORT TRIAGE  -  end-to-end pipeline test console")
    print(BAR)
    print(settings.summary())
    note = settings.banner()
    if note:
        print(note)

    health_checks()

    print("\nloading corpus, embedding model and policies ...", flush=True)
    t0 = time.time()
    pipe = Pipeline()
    print("ready in %.1fs.  type a ticket, or `help`.  `exit` to quit." % (time.time() - t0))

    ran, elapsed_all, last = 0, [], None

    def run_one(text: str, show_draft: bool = True):
        nonlocal ran, last
        t0 = time.time()
        out = pipe.run(text)
        dt = time.time() - t0
        ran += 1
        elapsed_all.append(dt)
        last = (out, dt, int(t0 * 1000))
        render(out, dt, show_draft, int(t0 * 1000))
        return out

    while True:
        try:
            line = input("\nticket> ").strip()
        except KeyboardInterrupt:
            print("\n  (type `exit` to quit)")
            continue
        except EOFError:
            break

        if not line:
            continue

        cmd, _, rest = line.partition(" ")
        low = cmd.lower()
        rest = rest.strip()

        try:
            if low in ("exit", "quit", ":q"):
                break

            elif low == "help":
                print(HELP)

            elif low == "demo":
                outs = [run_one(t, show_draft=False) for t in DEMO]
                auto = sum(1 for o in outs if o["decision"]["allowed"])
                print("\n" + BAR)
                print("%d tickets: %d auto-sent, %d escalated"
                      % (len(outs), auto, len(outs) - auto))
                print(BAR)

            elif low == "queue":
                show_queue()

            elif low == "show":
                show_trail(rest) if rest else print("  usage: show <ticket_id>")

            elif low == "last":
                if last:
                    render(last[0], last[1], True, last[2])
                else:
                    print("  nothing run yet")

            elif low == "drafter":
                if rest == "template":
                    pipe.drafter = backends.TemplateDrafter()
                elif rest == "mantle":
                    pipe.drafter = backends.FallbackDrafter(backends.MantleDrafter())
                else:
                    print("  usage: drafter mantle | template")
                    continue
                print("  drafter is now %s" % pipe.drafter.name)

            elif low == "model":
                if not rest:
                    print("  current: %s" % settings.MANTLE_MODEL)
                    continue
                pipe.drafter = backends.FallbackDrafter(backends.MantleDrafter(rest))
                print("  drafter is now %s" % pipe.drafter.name)

            elif low == "check":
                health_checks()

            elif low == "stats":
                if not elapsed_all:
                    print("  nothing run yet")
                else:
                    print("  %d tickets   mean %.2fs   median %.2fs   min %.2fs   max %.2fs"
                          % (ran, statistics.mean(elapsed_all), statistics.median(elapsed_all),
                             min(elapsed_all), max(elapsed_all)))

            else:
                run_one(line)

        except Exception:
            # A failure must never end the session — that is the whole point of
            # a console you can leave open while you iterate.
            print("\n  ERROR - the session is still alive, keep going\n")
            traceback.print_exc(limit=4)

    print("\n%s\nsession ended: %d tickets run%s" % (
        SUB, ran,
        ", mean %.2fs" % statistics.mean(elapsed_all) if elapsed_all else ""))
    print("trail saved to %s" % settings.TRAIL_PATH)


if __name__ == "__main__":
    main()
