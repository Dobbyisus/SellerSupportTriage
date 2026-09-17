"""Deterministic topic extraction from raw ticket text.

No model runs here. Every topic is a word-boundary regex match against the
seller's own words, and `explain()` names the exact phrase that fired.

WHY THIS IS NOT A MODEL CALL
    The Cedar forbid rules read these topics. If a model produced them, the
    gate would depend on a model being right about the very cases where being
    wrong is most expensive. A safety control has to be independent of the
    thing it controls. It also means the gate's behaviour is reviewable by
    someone who does not read Python: the trigger lists below ARE the spec.

BIAS: TOWARDS ESCALATION
    False positives cost a human thirty seconds of reading. False negatives
    auto-send a wrong answer about someone's suspended account. The lists are
    therefore deliberately broad, and no negation handling is attempted — a
    ticket saying "my account was NOT deactivated" still escalates. That is the
    correct trade, and a known, deliberate limitation.
"""

import re

# topic -> trigger phrases. Matched case-insensitively on word boundaries, so
# "sue" does not fire on "issue" and "poa" does not fire on "poached".
TRIGGERS: dict[str, list[str]] = {
    # ---- F1: account enforcement -----------------------------------------
    "account_deactivation": [
        r"deactivat\w*",
        r"account (?:was |is |got )?closed",
        r"shut (?:down|off) my account",
        r"lost my selling privileges",
        r"selling privileges (?:were |have been )?(?:removed|revoked)",
        r"can(?:no|')?t sell (?:any ?more|on amazon)",
        r"banned",
    ],
    "account_suspension": [
        r"suspend\w*",
        r"account (?:is |was |got )?on hold",
    ],
    "enforcement_action": [
        r"policy violation",
        r"violation notice",
        r"performance notification",
        r"enforcement",
        r"strike against",
    ],
    "account_review": [
        r"account (?:is )?under review",
        r"account review",
        r"verification review",
    ],
    # ---- F2: money -------------------------------------------------------
    "payment_hold": [
        r"payment (?:is |was |are )?(?:on hold|held|being held)",
        r"funds? (?:on hold|held|frozen)",
        r"money (?:on hold|is held|is stuck|is frozen)",
        r"payments? (?:have been |been )?stopped",
    ],
    "disbursement_failure": [
        r"disbursement\w*",
        r"(?:transfer|payout|deposit) (?:failed|keeps failing|did ?n[o']t go through)",
        r"(?:have|has)(?:n't| not) (?:been )?(?:paid|received (?:my )?payment)",
        r"no payment (?:since|for)",
    ],
    "account_reserve": [
        r"reserve\w*",
        r"unavailable balance",
    ],
    "funds_withheld": [
        r"withh(?:eld|olding)",
    ],
    # ---- F3: appeals -----------------------------------------------------
    "appeal": [
        r"appeal\w*",
    ],
    "reinstatement": [
        r"reinstat\w*",
        r"(?:get|want) my account back",
        r"restore my account",
    ],
    "plan_of_action": [
        r"plan of action",
        r"\bpoa\b",
    ],
    # ---- F4: listings ----------------------------------------------------
    "listing_suppressed": [
        r"suppress\w*",
        r"(?:listing|offer|product) (?:is )?hidden",
        r"not (?:showing|appearing) (?:up )?in search",
        r"can(?:no|')?t (?:be )?find my (?:listing|product)",
        r"buy box (?:is )?(?:gone|lost|removed)",
    ],
    "listing_removed": [
        r"(?:listing|offer|asin|product) (?:was |got |been )?(?:removed|taken down|blocked|deleted)",
        r"delisted",
    ],
    "category_gated": [
        r"\bgat(?:ed|ing)\b",
        r"ungating",
        r"(?:need|require)s? approval to (?:sell|list)",
        r"category approval",
        r"restricted category",
    ],
    # ---- F5: legal -------------------------------------------------------
    "ip_infringement": [
        r"infring\w*",
        r"intellectual property",
        r"trademark",
        r"copyright",
        r"patent",
        r"\bip (?:claim|complaint|violation)",
    ],
    "counterfeit_claim": [
        r"counterfeit\w*",
        r"inauthentic",
        r"fake (?:product|item|goods)",
    ],
    "legal_threat": [
        r"lawyer",
        r"attorney",
        r"legal (?:action|counsel|proceedings)",
        r"\bsue\b",
        r"lawsuit",
        r"litigation",
        r"take (?:you|amazon) to court",
    ],
    "law_enforcement": [
        r"law enforcement",
        r"\bpolice\b",
        r"subpoena",
        r"warrant",
    ],
    "regulatory": [
        r"\bftc\b",
        r"regulator\w*",
        r"consumer protection",
        r"trading standards",
    ],
}

_COMPILED = {
    topic: [re.compile(r"\b%s\b" % pat, re.IGNORECASE) for pat in pats]
    for topic, pats in TRIGGERS.items()
}


def extract(text: str) -> list[str]:
    """Return the topics present in the ticket, sorted for stable logging."""
    return sorted(t for t, pats in _COMPILED.items() if any(p.search(text) for p in pats))


def explain(text: str) -> list[dict]:
    """Same, but naming the phrase that fired — this is the audit trail."""
    out = []
    for topic, pats in _COMPILED.items():
        for pat in pats:
            m = pat.search(text)
            if m:
                out.append({"topic": topic, "matched": m.group(0), "pattern": pat.pattern})
                break
    return sorted(out, key=lambda d: d["topic"])


def main() -> None:
    import sys

    if len(sys.argv) < 2:
        print("usage: python gate/topics.py \"ticket text\"")
        print("\n%d topics, %d trigger phrases"
              % (len(TRIGGERS), sum(len(v) for v in TRIGGERS.values())))
        raise SystemExit(1)

    text = " ".join(sys.argv[1:])
    hits = explain(text)
    print("ticket : %s" % text)
    if not hits:
        print("topics : (none) — no forbid rule keys off this ticket")
        return
    print("topics :")
    for h in hits:
        print("   %-22s matched %r" % (h["topic"], h["matched"]))


if __name__ == "__main__":
    main()
