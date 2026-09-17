"""The escalation gate: evaluates a ticket against policies.cedar.

    python gate/gate.py --self-test        # run the test matrix
    python gate/gate.py --validate         # type-check policies against schema
    python gate/gate.py --schema-json      # emit the AVP-format schema
    python gate/gate.py "my account was deactivated" --confidence 0.81

This runs the real Cedar engine (cedarpy), not a reimplementation, so the
`forbid`-overrides-`permit` semantics are Cedar's own. When the build moves to
Amazon Verified Permissions the same policy text and the same request shape
apply — only the transport changes.

WHAT THE GATE IS AND IS NOT
    It decides ONE thing: may this drafted reply go out without a human. It
    does not decide what the reply says. Everything denied goes to the
    escalation queue carrying the rule id and reason that stopped it, which is
    what the console renders on the ticket's decision trail.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cedarpy

import topics as topics_mod

HERE = Path(__file__).resolve().parent
POLICIES = HERE / "policies.cedar"
SCHEMA = HERE / "schema.cedarschema"

PRINCIPAL = 'SellerTriage::Agent::"seller-triage"'
ACTION = 'SellerTriage::Action::"AutoSend"'

# Cedar assigns positional ids (policy0, policy1, ...) when parsing a policy
# set from text; the @id annotation is documentation, not the id. Reading the
# annotation block that precedes each rule, in file order, rebuilds the mapping
# — so rule names, operator messages and provenance all live in the .cedar file
# rather than being duplicated here. Editing a rule edits everything about it.
_BLOCK = re.compile(
    r'((?:^@\w+\("[^"]*"\)\n)+)^(?:permit|forbid)\s*\(', re.MULTILINE
)
_ANNOTATION = re.compile(r'^@(\w+)\("([^"]*)"\)$', re.MULTILINE)


def load_policies(path=None, text=None) -> tuple[str, dict[str, dict]]:
    """Load from a file, or evaluate policy text held only in memory.

    `text` is how the demo and the tests try policy variants without ever
    writing to policies.cedar — see session.py.
    """
    if text is None:
        path = path or POLICIES
        text = path.read_text(encoding="utf-8")
    meta = {}
    for i, m in enumerate(_BLOCK.finditer(text)):
        ann = dict(_ANNOTATION.findall(m.group(1)))
        ann.setdefault("id", "policy%d" % i)
        ann.setdefault("reason", "")
        meta["policy%d" % i] = ann
    if not meta:
        raise SystemExit("No annotated rules found in %s" % (path or "<memory>"))
    return text, meta


def policy_floor(path=None, text=None) -> float:
    """The confidence floor the policy set actually enforces.

    Read from the policies rather than assumed, so swapping in an edited policy
    file reports the floor that file enforces — not a stale constant.
    """
    if text is None:
        text = (path or POLICIES).read_text(encoding="utf-8")
    m = re.search(r'decimal\("([0-9.]+)"\)', text)
    return float(m.group(1)) if m else 0.72


def policy_topics(path=None, text=None) -> dict[str, list[str]]:
    """Topic strings each forbid rule keys off, read straight from the policies.

    Used by property_test.py to prove the policy set and topics.py have not
    drifted apart — a typo in either file would otherwise silently disable a
    guardrail with no test failing.
    """
    if text is None:
        text = (path or POLICIES).read_text(encoding="utf-8")
    out = {}
    for m in _BLOCK.finditer(text):
        ann = dict(_ANNOTATION.findall(m.group(1)))
        tail = text[m.end() : m.end() + 900]
        body = tail.split("};")[0]
        found = re.findall(r'"([a-z_]+)"', body)
        if found:
            out[ann.get("id", "?")] = found
    return out


def build_ticket(
    text: str,
    category: str = "account",
    confidence: float = 0.80,
    has_citation: bool = True,
    draft_quotes_policy: bool = True,
    topic_override: list[str] | None = None,
) -> dict:
    """Assemble the Cedar resource for one ticket."""
    found = topic_override if topic_override is not None else topics_mod.extract(text)
    return {
        "text": text,
        "category": category,
        "topics": found,
        # Cedar's decimal extension needs a fixed-precision string, and it
        # rejects more than 4 decimal places.
        "retrieval_confidence": "%.4f" % confidence,
        "has_citation": has_citation,
        "draft_quotes_policy": draft_quotes_policy,
    }


def decide(ticket: dict, ticket_id: str = "t1", policy_path=None, policy_text=None) -> dict:
    policy_text, meta = load_policies(policy_path, policy_text)

    entities = [
        {
            "uid": {"type": "SellerTriage::Ticket", "id": ticket_id},
            "attrs": {
                "category": ticket["category"],
                "topics": ticket["topics"],
                "retrieval_confidence": {"__extn": {"fn": "decimal", "arg": ticket["retrieval_confidence"]}},
                "has_citation": ticket["has_citation"],
                "draft_quotes_policy": ticket["draft_quotes_policy"],
            },
            "parents": [],
        },
        {
            "uid": {"type": "SellerTriage::Agent", "id": "seller-triage"},
            "attrs": {},
            "parents": [],
        },
    ]

    request = {
        "principal": PRINCIPAL,
        "action": ACTION,
        "resource": 'SellerTriage::Ticket::"%s"' % ticket_id,
        "context": {},
    }

    result = cedarpy.is_authorized(request, policy_text, entities)
    allowed = result.decision == cedarpy.Decision.Allow
    firing = [meta.get(pid, {"id": pid, "reason": ""}) for pid in (result.diagnostics.reasons or [])]

    # On a Deny, Cedar reports the forbid rules that matched. On an Allow it
    # reports the permit. Only the forbids are worth showing an operator.
    blockers = [f for f in firing if f["id"].startswith("F")]

    return {
        "decision": "AUTO_SEND" if allowed else "ESCALATE",
        "allowed": allowed,
        "policies_fired": [f["id"] for f in firing],
        "blocked_by": [f["id"] for f in blockers],
        "reasons": [f["reason"] for f in blockers],
        # Provenance travels with the decision, so the console can show WHY a
        # rule exists — not just that it fired. This is what backs the claim
        # that the categories are documented rather than invented.
        "provenance": [
            {
                "rule": f["id"],
                "grounding": f.get("grounding", ""),
                "source_type": f.get("source_type", ""),
                "source": f.get("source", ""),
                "policy_ref": f.get("policy_ref", ""),
                "policy_url": f.get("policy_url", ""),
            }
            for f in blockers
        ],
        "topics": ticket["topics"],
        "category": ticket["category"],
        "retrieval_confidence": float(ticket["retrieval_confidence"]),
    }


# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------
# These are GATE fixtures, not demo tickets. Demo tickets are a separate job and
# must be sourced from the Seller Central forum threads in problem-and-sources.md
# (and must never reuse filter_queries.tsv, which selected the corpus).
#
# Each case names the rule it is meant to exercise, so a failure says which
# rule broke rather than just "expected ESCALATE".

CASES = [
    # ---- should auto-send ------------------------------------------------
    ("when will my next payment arrive in my bank account", "payments", 0.81, True, True,
     "AUTO_SEND", None, "routine payment-schedule question"),
    ("how do I change the handling time on my listings", "shipping", 0.79, True, True,
     "AUTO_SEND", None, "routine operational how-to"),
    ("what is the late shipment rate target", "shipping", 0.88, True, True,
     "AUTO_SEND", None, "metric definition, well grounded"),
    ("how do I issue a partial refund to a buyer", "returns", 0.83, True, True,
     "AUTO_SEND", None, "routine returns how-to"),

    # ---- F1 account enforcement ------------------------------------------
    ("my account was deactivated and I want to know why", "account", 0.85, True, True,
     "ESCALATE", "F1_account_enforcement", "explicit deactivation"),
    ("I received a policy violation notice on my account", "account", 0.80, True, True,
     "ESCALATE", "F1_account_enforcement", "enforcement action"),

    # ---- THE DEMO BEAT ---------------------------------------------------
    # Classified routine "shipping", high retrieval confidence, would sail
    # through P0 — but the seller mentions deactivation, so F1 catches it.
    # This works because topics are extracted independently of the classifier.
    ("my parcel is running late and the buyer is angry, and now amazon says my "
     "account could be deactivated if it happens again", "shipping", 0.86, True, True,
     "ESCALATE", "F1_account_enforcement", "DEMO BEAT: classified routine, caught anyway"),

    # ---- F2 payments -----------------------------------------------------
    ("my payment is on hold and I don't know why", "payments", 0.79, True, True,
     "ESCALATE", "F2_payments", "payment hold"),
    ("my disbursement keeps failing and I do not know what to fix", "payments", 0.75, True, True,
     "ESCALATE", "F2_payments", "disbursement failure"),
    ("why is there a reserve on my account balance", "payments", 0.82, True, True,
     "ESCALATE", "F2_payments", "account reserve"),

    # ---- F3 appeals ------------------------------------------------------
    ("how do I write a plan of action", "account", 0.61, True, True,
     "ESCALATE", "F3_appeals", "worst retrieval case in the corpus"),
    ("I want to appeal the decision and get my account back", "account", 0.78, True, True,
     "ESCALATE", "F3_appeals", "appeal and reinstatement"),

    # ---- F4 listings -----------------------------------------------------
    ("my listing is suppressed and not showing up in search", "listings", 0.70, True, True,
     "ESCALATE", "F4_listing_suppression", "suppression"),
    ("I cannot list in this category it says I need approval to sell", "listings", 0.69, True, True,
     "ESCALATE", "F4_listing_suppression", "gated category"),

    # ---- F5 legal --------------------------------------------------------
    ("I received a trademark infringement complaint against my listing", "listings", 0.84, True, True,
     "ESCALATE", "F5_legal", "IP claim"),
    ("someone reported my products as counterfeit and I will speak to my lawyer", "account", 0.80, True, True,
     "ESCALATE", "F5_legal", "counterfeit plus legal threat"),

    # ---- F6 ungrounded ---------------------------------------------------
    ("my buyer says the item smells strange, what should I tell them", "returns", 0.44, True, True,
     "ESCALATE", "F6_ungrounded", "retrieval below the floor"),
    ("how do I change the handling time on my listings", "shipping", 0.79, False, True,
     "ESCALATE", "F6_ungrounded", "no citation available"),
    ("how do I change the handling time on my listings", "shipping", 0.79, True, False,
     "ESCALATE", "F6_ungrounded", "draft does not quote the passage"),

    # ---- boundary --------------------------------------------------------
    ("what is the valid tracking rate requirement", "shipping", 0.72, True, True,
     "AUTO_SEND", None, "exactly at MIN_SIM — must pass, floor is inclusive"),
    ("what is the valid tracking rate requirement", "shipping", 0.7199, True, True,
     "ESCALATE", "F6_ungrounded", "one notch below the floor"),
]


def self_test() -> int:
    print("Cedar gate — test matrix\n")
    passed = failed = 0
    for text, cat, conf, cite, quotes, expected, expect_rule, label in CASES:
        ticket = build_ticket(text, cat, conf, cite, quotes)
        out = decide(ticket)
        ok = out["decision"] == expected
        if ok and expect_rule:
            ok = expect_rule in out["blocked_by"]

        mark = "ok  " if ok else "FAIL"
        print("%s %-10s %-28s %s" % (mark, out["decision"], ",".join(out["blocked_by"]) or "-", label))
        if not ok:
            print("       expected %s%s" % (expected, " via %s" % expect_rule if expect_rule else ""))
            print("       topics=%s conf=%.4f" % (out["topics"], out["retrieval_confidence"]))
            failed += 1
        else:
            passed += 1

    print("\n%d passed, %d failed" % (passed, failed))

    # The property the demo turns on: forbid beats permit even when every
    # permit condition is satisfied.
    demo = build_ticket(
        "my parcel is running late and the buyer is angry, and now amazon says my "
        "account could be deactivated if it happens again",
        "shipping", 0.86, True, True)
    out = decide(demo)
    print("\ndemo beat: classifier said '%s', retrieval %.2f (above floor), "
          "P0 conditions all met" % (out["category"], out["retrieval_confidence"]))
    print("           -> %s, blocked by %s" % (out["decision"], out["blocked_by"]))
    print("           -> %s" % (out["reasons"][0] if out["reasons"] else ""))
    return failed


def validate() -> int:
    policy_text, _ = load_policies()
    schema_text = SCHEMA.read_text(encoding="utf-8")
    result = cedarpy.validate_policies(policy_text, schema_text)
    if result.validation_passed:
        print("policies validate against the schema.")
        return 0
    print("VALIDATION FAILED:")
    for err in result.errors:
        print("   - %s" % err)
    return 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Cedar escalation gate")
    ap.add_argument("ticket", nargs="*", help="ticket text to evaluate")
    ap.add_argument("--category", default="account")
    ap.add_argument("--confidence", type=float, default=0.80)
    ap.add_argument("--no-citation", action="store_true")
    ap.add_argument("--no-quote", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--schema-json", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--why", action="store_true", help="show why each firing rule exists")
    ap.add_argument(
        "--policies",
        help="evaluate against a different .cedar file. The live-policy-edit demo: "
             "run a ticket, point at an edited policy set, run it again, watch the "
             "decision flip with no code change.",
    )
    args = ap.parse_args()

    if args.self_test:
        sys.exit(1 if self_test() else 0)
    if args.validate:
        sys.exit(validate())
    if args.schema_json:
        schema = cedarpy.Schema(SCHEMA.read_text(encoding="utf-8"))
        print(json.dumps(json.loads(schema.to_json_str()), indent=2))
        return
    if not args.ticket:
        ap.error("give a ticket, or --self-test / --validate / --schema-json")

    text = " ".join(args.ticket)
    ticket = build_ticket(
        text, args.category, args.confidence, not args.no_citation, not args.no_quote
    )
    policy_path = Path(args.policies) if args.policies else None
    out = decide(ticket, policy_path=policy_path)

    if args.json:
        print(json.dumps(out, indent=2))
        return

    if policy_path:
        print("\npolicies   : %s" % policy_path)
    print("\nticket     : %s" % text)
    print("category   : %s   (classifier)" % out["category"])
    print("topics     : %s   (deterministic, independent of the classifier)"
          % (", ".join(out["topics"]) or "none"))
    print("confidence : %.4f  (exact cosine, floor %.2f)"
          % (out["retrieval_confidence"], policy_floor(policy_path)))
    print("\nDECISION   : %s" % out["decision"])
    for rid, reason in zip(out["blocked_by"], out["reasons"]):
        print("  %-26s %s" % (rid, reason))
    if out["allowed"]:
        print("  P0_routine_permit          no forbid rule matched")

    if args.why:
        for prov in out["provenance"]:
            print("\n  why %s exists" % prov["rule"])
            print("     %s" % prov["grounding"])
            label = {
                "published_guidance": "published guidance (third-party, not Amazon first-party)",
                "design_decision": "our own design decision, not from published guidance",
                "design_decision_and_measurement": "our decision, backed by measurement",
            }.get(prov["source_type"], prov["source_type"])
            print("     basis  : %s" % label)
            if prov["source"]:
                print("     source : %s" % prov["source"])
            if prov["policy_ref"]:
                print("     policy : %s" % prov["policy_ref"])
                print("              %s" % prov["policy_url"])


if __name__ == "__main__":
    main()
