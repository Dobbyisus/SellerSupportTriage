"""Is this ticket a problem being reported, or a question being asked?

    python pipeline/intent.py "how do I get the prime badge on my listings"

WHY THE DIFFERENCE MATTERS
    The queue was built for complaints: something broke, a seller wrote in.
    But the same retrieval, the same grounding check and the same gate answer
    "how do I qualify for the Prime badge?" just as well, and that is most of
    what a seller actually wants to know. The only thing that has to change is
    the shape of the reply — a seller asking a question should not be met with
    "I am sorry to hear about the problem you are having."

    So this returns one word, it goes in the trail, and it selects a paragraph
    in the drafting prompt. Nothing else.

WHAT IT DELIBERATELY DOES NOT DO
    It does not touch the gate. A question about a deactivated account is
    still an account ticket and still escalates — the seller's phrasing is not
    a safety signal, and letting a polite question route around a forbid rule
    would be exactly the hole the gate exists to close. Same reason the gate
    reads topics rather than the classifier.

NO MODEL RUNS HERE
    Regexes over the seller's own words, like gate/topics.py. A model call to
    decide the tone of a paragraph would be three seconds and a dollar for
    something a question mark already tells us.
"""

import re
import sys

# Asking for guidance: an interrogative opening, or a question mark.
ASKING = [
    r"^\s*(?:how|what|when|where|which|who|why)\b",
    r"^\s*(?:can|could|do|does|did|should|is|are|am|will|would|may)\s+(?:i|we|you|my|a|an|the)\b",
    r"\bhow (?:do|can|would|should) (?:i|we|you)\b",
    r"\bwhat (?:is|are|counts as|happens)\b",
    r"\b(?:am|are) (?:i|we) (?:eligible|able|allowed|required)\b",
    r"\bdo i (?:need|have to|qualify)\b",
    r"\bqualif(?:y|ies|ication)\b",
    r"\bwhat are the requirements\b",
    r"\bis there a way\b",
    r"\?",
]

# Reporting that something has gone wrong. These beat the question markers:
# "why has my account been suspended?" is a problem wearing a question mark.
REPORTING = [
    r"\b(?:isn'?t|aren'?t|wasn'?t|doesn'?t|didn'?t|won'?t|can'?t|cannot)\b",
    r"\bnot (?:working|showing|received|paid|arrived|able)\b",
    r"\b(?:still|yet) (?:no|not|waiting|haven'?t|hasn'?t)\b",
    r"\bno (?:one|body|response|reply|answer|payment|update)\b",
    r"\b(?:my|our) \w+ (?:is|was|has been|have been|got|keeps) (?:on hold|held|late|missing|blocked|removed|suspended|deactivated|stuck|frozen|failing|delayed)\b",
    r"\b(?:i|we) (?:have|'ve)? ?(?:been )?(?:waiting|charged|lost|received a)\b",
    r"\b(?:angry|upset|frustrat\w+|urgent|please help|help me)\b",
    r"\b(?:complain\w*|escalat\w*)\b",
    r"\bwent wrong\b",
    r"\bkeeps? (?:failing|happening)\b",
    # Word order moves around in a real ticket — "why has my account been
    # suspended?" is a question mark on top of a deactivated account. The
    # enforcement words are therefore matched near the seller's own noun
    # rather than inside one fixed phrase.
    r"\b(?:my|our) (?:account|listing|offer|product|payment|funds?|money|disbursement|balance)\b"
    r"(?:\W+\w+){0,6}?\W+(?:suspend\w*|deactivat\w*|block\w*|removed|withheld|frozen|"
    r"held|late|missing|stuck|gone)\b",
    r"\bbeen (?:suspend\w*|deactivat\w*|block\w*|removed|withheld|held|charged)\b",
]

_ASKING = [re.compile(p, re.IGNORECASE) for p in ASKING]
_REPORTING = [re.compile(p, re.IGNORECASE) for p in REPORTING]


def explain(text: str) -> dict:
    """The intent, plus the phrase that decided it — the audit trail."""
    asked = next((m.group(0) for p in _ASKING for m in [p.search(text)] if m), None)
    told = next((m.group(0) for p in _REPORTING for m in [p.search(text)] if m), None)

    if asked and not told:
        return {"intent": "question", "matched": asked.strip(),
                "why": "asks for guidance and reports nothing broken"}
    if told:
        return {"intent": "problem", "matched": told.strip(),
                "why": "reports something that has gone wrong"}
    # Neither marker: a bare statement of a situation. Treated as a problem,
    # which is the safer default — it produces the more careful reply.
    return {"intent": "problem", "matched": "",
            "why": "no question asked, so read as a report"}


def classify(text: str) -> str:
    return explain(text)["intent"]


# Pinned so a new pattern cannot quietly flip an existing reading. Where a
# ticket is genuinely both, the bias is towards "problem": the cost of being
# wrong that way is a reply that opens more carefully than it needed to.
CASES: list[tuple[str, str]] = [
    ("how do I get the prime badge on my listings", "question"),
    ("what is the late shipment rate target", "question"),
    ("am I eligible for seller fulfilled prime with 94% on-time delivery?", "question"),
    ("do I need to upload tracking for every order", "question"),
    ("what happens if my ODR goes above 1%", "question"),
    ("my payment is on hold and I don't know why", "problem"),
    ("why has my account been suspended?", "problem"),
    ("my listing is suppressed and not showing up in search", "problem"),
    ("my parcel is running late and the buyer is angry", "problem"),
    ("I still have not been paid for last month", "problem"),
    ("I need to change my handling time", "problem"),
]


def self_test() -> int:
    failed = 0
    for text, expected in CASES:
        out = explain(text)
        ok = out["intent"] == expected
        print("%s %-9s %s" % ("ok  " if ok else "FAIL", out["intent"], text))
        if not ok:
            failed += 1
            print("       expected %s — decided by %r" % (expected, out["matched"]))
    print("\n%d failure(s)" % failed)
    return 1 if failed else 0


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        raise SystemExit(self_test())
    if len(sys.argv) < 2:
        print('usage: python pipeline/intent.py "ticket text"')
        raise SystemExit(1)
    out = explain(" ".join(sys.argv[1:]))
    print("intent : %s" % out["intent"])
    print("why    : %s%s" % (out["why"],
                             " (%r)" % out["matched"] if out["matched"] else ""))


if __name__ == "__main__":
    main()
