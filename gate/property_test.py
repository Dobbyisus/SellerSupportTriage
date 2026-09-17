"""Exhaustive property tests over the gate's defined input space.

    python gate/property_test.py
    python gate/property_test.py --verbose

The 21 cases in gate.py are examples. These are properties: they enumerate the
whole input space we defined and assert things that must hold for every point
in it.

HONEST FRAMING — say it this way and no other way:
    "Exhaustively tested over the defined input space."
NOT "proven correct" and NOT "formally verified". Cedar supports symbolic
analysis; this is not that. What this is: every topic the extractor can emit,
crossed with both sides of every boolean and both sides of the confidence
floor, is actually evaluated through the real Cedar engine.

THE PROPERTY THAT EARNS ITS KEEP IS P1.
    topics.py and policies.cedar name their topics as strings, in two separate
    files, with no compiler between them. Rename a topic in one and not the
    other, or typo it, and a guardrail silently stops working — every example
    test still passes, because no example covers that topic. P1 and P2 compare
    the two sets directly and fail loudly on any drift.
"""

import argparse
import itertools
import sys

import gate as gate_mod
import topics as topics_mod

PASS, FAIL = "ok  ", "FAIL"
failures: list[str] = []


def report(ok: bool, label: str, detail: str = "") -> None:
    print("%s %s%s" % (PASS if ok else FAIL, label, ("  — " + detail) if detail else ""))
    if not ok:
        failures.append(label)


def evaluate(topic_list, confidence=0.95, citation=True, quotes=True, category="shipping",
             conflict=False):
    ticket = gate_mod.build_ticket(
        "", category, confidence, citation, quotes, topic_override=list(topic_list),
        conflicts_with_precedent=conflict,
    )
    return gate_mod.decide(ticket)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    extractor_topics = set(topics_mod.TRIGGERS)
    by_rule = gate_mod.policy_topics()
    policy_all = {t for topics in by_rule.values() for t in topics}

    print("gate property tests\n")
    print("topics.py emits      : %d topics" % len(extractor_topics))
    print("policies.cedar guards: %d topics across %d forbid rules"
          % (len(policy_all), len(by_rule)))
    print()

    # ---------------------------------------------------------------- P1
    print("P1  every topic named in policies.cedar can actually be produced")
    orphans = sorted(policy_all - extractor_topics)
    report(
        not orphans,
        "no policy guards a topic the extractor never emits",
        "orphaned: %s" % orphans if orphans else "",
    )
    if orphans:
        print("      A forbid rule naming a topic topics.py cannot produce is dead.")
        print("      Usually a rename or typo in one file and not the other.")

    # ---------------------------------------------------------------- P2
    print("\nP2  every topic the extractor emits is guarded by some rule")
    unguarded = sorted(extractor_topics - policy_all)
    report(
        not unguarded,
        "no extractable topic is silently unhandled",
        "unguarded: %s" % unguarded if unguarded else "",
    )
    if unguarded:
        print("      topics.py detects these but no rule blocks them — they")
        print("      auto-send. Either guard them or delete them from topics.py.")

    # ---------------------------------------------------------------- P3
    print("\nP3  every guarded topic, alone and with perfect retrieval, escalates")
    print("    (confidence 0.95, citation present, draft quotes the passage —")
    print("     every P0 condition satisfied, so only the topic can block it)")
    bad = []
    for topic in sorted(policy_all):
        out = evaluate([topic])
        if out["decision"] != "ESCALATE":
            bad.append(topic)
        elif args.verbose:
            print("      %-24s -> %s" % (topic, ",".join(out["blocked_by"])))
    report(not bad, "%d/%d guarded topics escalate" % (len(policy_all) - len(bad), len(policy_all)),
           "leaked: %s" % bad if bad else "")

    # ---------------------------------------------------------------- P4
    print("\nP4  the documented escalation list never auto-sends")
    documented = {
        "account suspension / deactivation": ["account_deactivation", "account_suspension"],
        "payment holds / disbursement delays": ["payment_hold", "disbursement_failure"],
        "incorrect listing suppressions": ["listing_suppressed", "listing_removed"],
    }
    for label, tlist in documented.items():
        results = [evaluate([t])["decision"] for t in tlist]
        report(all(r == "ESCALATE" for r in results), label, ", ".join(tlist))

    # ---------------------------------------------------------------- P5
    print("\nP5  the confidence floor is exact and inclusive")
    for conf, expected in [
        (0.7199, "ESCALATE"), (0.72, "AUTO_SEND"), (0.7201, "AUTO_SEND"),
        (0.0, "ESCALATE"), (1.0, "AUTO_SEND"),
    ]:
        got = evaluate([], confidence=conf)["decision"]
        report(got == expected, "confidence %.4f -> %s" % (conf, expected),
               "" if got == expected else "got %s" % got)

    # ---------------------------------------------------------------- P6
    print("\nP6  exhaustive: every single-topic state x three booleans x floor")
    combos = 0
    leaks = []
    subsets = [()] + [(t,) for t in sorted(extractor_topics)]
    for subset, conf, cite, quote, conflict in itertools.product(
        subsets, (0.7199, 0.72, 0.95), (True, False), (True, False), (False, True)
    ):
        out = evaluate(subset, conf, cite, quote, conflict=conflict)
        combos += 1
        # Auto-send is legal only when nothing is guarded, every F6 input is
        # clean, AND no already-sent reply would be contradicted (F7).
        should_allow = (
            not (set(subset) & policy_all) and conf >= 0.72 and cite and quote
            and not conflict
        )
        if out["allowed"] != should_allow:
            leaks.append((subset, conf, cite, quote, conflict, out["decision"]))
    report(not leaks, "%d combinations, all as expected" % combos,
           "%d mismatches" % len(leaks) if leaks else "")
    for leak in leaks[:5]:
        print("      %s" % (leak,))

    # ---------------------------------------------------------------- P7
    print("\nP7  monotonicity: adding a topic never unblocks a blocked ticket")
    violations = []
    guarded = sorted(policy_all)
    for a, b in itertools.combinations(guarded[:12], 2):
        if evaluate([a])["allowed"] or evaluate([b])["allowed"]:
            continue
        if evaluate([a, b])["allowed"]:
            violations.append((a, b))
    report(not violations, "no pair of blocked topics becomes allowed together",
           str(violations[:3]) if violations else "")

    # ---------------------------------------------------------------- P8
    print("\nP8  the classifier cannot influence the outcome")
    print("    (same ticket, every category label — the decision must not move)")
    moved = []
    for topic in ["account_deactivation", "payment_hold", "ip_infringement"]:
        seen = {
            evaluate([topic], category=c)["decision"]
            for c in ["account", "returns", "shipping", "payments", "listings"]
        }
        if len(seen) != 1:
            moved.append(topic)
    report(not moved, "decision is invariant across all 5 classifier labels",
           str(moved) if moved else "")
    print("      This is the independence property stated as a test: the gate")
    print("      holds even when the classifier is wrong.")

    # ---------------------------------------------------------------- P9
    print("\nP9  a conflicting precedent blocks on its own, and only F7 fires")
    out = evaluate([], conflict=True)
    report(out["decision"] == "ESCALATE" and out["blocked_by"] == ["F7_conflicting_precedent"],
           "clean ticket + conflict -> ESCALATE via F7 alone",
           "" if out["blocked_by"] == ["F7_conflicting_precedent"] else str(out["blocked_by"]))
    out = evaluate([], conflict=False)
    report(out["decision"] == "AUTO_SEND", "clean ticket + no conflict -> AUTO_SEND")
    print("      The check compares documents the system itself sent. It never")
    print("      reads the classifier, so P8 still holds with F7 in the set.")

    # ---------------------------------------------------------------- summary
    print()
    if failures:
        print("%d PROPERTY FAILURES:" % len(failures))
        for f in failures:
            print("   - %s" % f)
        sys.exit(1)
    print("all properties hold over %d evaluated states." % combos)
    print("\nClaim this as: \"exhaustively tested over the defined input space\".")
    print("Not \"formally verified\" — Cedar supports symbolic analysis, and")
    print("this is enumeration, not that.")


if __name__ == "__main__":
    main()
