"""Prompts for classify / draft / escalation-note, and the grounding check.

    python gate/drafting.py --show-prompts
    python gate/drafting.py --self-test

The reasoning behind every rule is in DRAFTING_INSTRUCTIONS.md. Change both
together.

Nothing here calls Bedrock — the prompts are assembled and the grounding check
runs locally, so all of it is testable while account authorization is pending.
`draft_ticket()` is the one function that will need a live client.
"""

import argparse
import json
import re
import sys

# Emitted instead of a reply when the passage does not answer the question.
# Routed straight to a human; never shown to a seller.
REFUSAL = "INSUFFICIENT_GROUNDING"

CLASSIFY_MODEL = "amazon.nova-micro-v1:0"
DRAFT_MODEL = "amazon.nova-pro-v1:0"

CATEGORIES = ["account", "returns", "shipping", "payments", "listings"]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM = """\
You label incoming Amazon seller support tickets for an internal triage queue.

Reply with exactly one word from this list and nothing else:
account, returns, shipping, payments, listings

Guidance:
- account   — account health, policy violations, verification, seller status
- returns   — returns, refunds, A-to-z claims, restocking fees
- shipping  — delivery, tracking, handling time, late shipments, cancellations
- payments  — disbursements, reserves, fees, currency conversion, statements
- listings  — creating or fixing listings, categories, detail pages, suppression

If a ticket touches more than one, choose the one the seller is asking about,
not the one they mention in passing.

Never explain. Never add punctuation. One word.\
"""


def build_classify_request(ticket_text: str) -> dict:
    """A Bedrock Converse request body for Nova Micro."""
    return {
        "modelId": CLASSIFY_MODEL,
        "system": [{"text": CLASSIFY_SYSTEM}],
        "messages": [{"role": "user", "content": [{"text": ticket_text}]}],
        "inferenceConfig": {"temperature": 0.0, "maxTokens": 5},
    }


def parse_category(raw: str) -> str:
    """Take the model at its word, but never let a bad label through."""
    word = raw.strip().lower().strip(".,\"'")
    return word if word in CATEGORIES else "account"


# ---------------------------------------------------------------------------
# Drafting
# ---------------------------------------------------------------------------

DRAFT_SYSTEM = """\
You draft replies for an Amazon Seller Support agent. The agent reads your
draft with the cited policy passage beside it, then sends or edits it. You are
writing TO the seller, FOR the agent to check.

THE PASSAGE IS YOUR ONLY SOURCE.
Every statement about Amazon policy must come from the passage supplied in the
user message. Not from general knowledge, not from what is probably true, not
from anything you recall about selling on Amazon. If it is not in the passage,
it does not go in the reply.

RULES

1. Quote at least one span from the passage verbatim, inside double quotation
   marks. Copy it exactly — this is checked automatically against the passage
   after you reply, and a draft whose quote does not match is discarded.

2. Cite the source on its own final line, exactly as given to you:
   Source: <heading path> <url>
   Never construct or guess a URL.

3. Address this seller's specific situation. Name the actual detail they
   raised. A reply that would fit any ticket in this category is a failure,
   even if every sentence in it is true.

4. If the passage does not answer their question, reply with exactly:
   INSUFFICIENT_GROUNDING
   and nothing else. Do not give partial answers, general advice, or
   "please contact support". Refusing is correct and expected.

5. Promise nothing the passage does not state. No timelines, disbursement
   dates, reinstatements, reversals or outcomes.

6. State no legal position. Nothing about infringement, counterfeit claims,
   regulators or law enforcement beyond what the passage says.

7. Invent no identifiers. No case numbers, order IDs, ASINs, amounts, dates or
   names that were not in the ticket or the passage.

8. When the passage gives a numeric threshold, use it exactly. Never round it,
   never paraphrase it as "low" or "acceptable".

STYLE
- 120 to 200 words.
- Address the seller as "you". Never mention AI, models, or this tool.
- Professional and direct. No effusive apology, no admission of fault.
- Expand jargon on first use: "Order Defect Rate (ODR)".
- Shape: acknowledge their specific problem, then what the policy says with
  your quote and citation, then what happens next or what they can do — that
  last part ONLY if the passage says so.\
"""

DRAFT_USER = """\
TICKET FROM SELLER
{ticket}

RETRIEVED POLICY PASSAGE
Title: {title}
Section: {heading_path}
URL: {url}

\"\"\"
{passage}
\"\"\"
{rules_block}
Write the reply now, or INSUFFICIENT_GROUNDING if the passage does not answer
this ticket.\
"""


def format_rules(rules: list[dict] | None) -> str:
    """Surface structured thresholds so the model quotes them exactly."""
    if not rules:
        return ""
    lines = ["", "EXACT THRESHOLDS IN THIS PASSAGE — reproduce these verbatim:"]
    for r in rules:
        lines.append(
            "  %s %s %s%s" % (r.get("metric"), r.get("operator"), r.get("value"), r.get("unit", ""))
        )
    return "\n".join(lines) + "\n"


def build_draft_request(ticket_text: str, hit: dict) -> dict:
    """A Bedrock Converse request body for Nova Pro.

    `hit` is one result from opensearch/search.py.
    """
    user = DRAFT_USER.format(
        ticket=ticket_text,
        title=hit.get("title") or "",
        heading_path=hit.get("heading_path") or "",
        url=hit.get("source_url") or "(no URL — do not cite a link)",
        passage=hit.get("text", "").strip(),
        rules_block=format_rules(hit.get("rules")),
    )
    return {
        "modelId": DRAFT_MODEL,
        "system": [{"text": DRAFT_SYSTEM}],
        "messages": [{"role": "user", "content": [{"text": user}]}],
        "inferenceConfig": {"temperature": 0.2, "maxTokens": 500},
    }


# ---------------------------------------------------------------------------
# Grounding check — this is what makes Cedar F6 real
# ---------------------------------------------------------------------------

# Curly quotes count as delimiters — models emit them freely.
_QUOTE = re.compile(r'["“]([^"“”]{12,})["”]')

# Same character, different codepoint. Folding these is not leniency: the
# corpus came from HTML and PDF, so it is full of curly quotes, en dashes and
# non-breaking spaces that a model retypes in their ASCII form.
_FOLD = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ",
})

# Punctuation a model adds at a quote boundary to fit its own sentence.
_EDGE = " .,;:!?\"'()[]-—–"


def _normalize(s: str) -> str:
    s = s.translate(_FOLD)
    s = s.replace("…", "...")
    s = re.sub(r"\s+", " ", s)
    # The extracted corpus has stray spaces before punctuation ("settings ."),
    # which no model reproduces when quoting.
    s = re.sub(r"\s+([.,;:!?)])", r"\1", s)
    return s.strip().lower()


# Trailing punctuation and markdown brackets are not part of the URL.
_URL = re.compile(r'https?://[^\s<>"\'\)\]]+')


def _canon_url(url: str) -> str:
    """Scheme + host + path, lowercased, no query, no fragment, no trailing slash.

    The query string is dropped on purpose: our citations carry `?locale=en-US`
    and a model that reproduces the link without it has pointed at the same
    document, not a different one. The PATH is what identifies the page, so a
    different help-article id (G14911 vs G200386250) still fails.
    """
    url = url.rstrip(".,;:!?")
    body = url.split("#", 1)[0].split("?", 1)[0]
    return body.rstrip("/").lower()


def check_urls(draft: str, allowed: list[str] | None) -> tuple[list[str], list[str]]:
    """Split the draft's links into ones it was given and ones it invented.

    verify_grounding validates quoted SPANS. It cannot see a fabricated link,
    and a plausible-looking Amazon help URL that goes nowhere is worse than a
    bad sentence — the seller clicks it. Observed in a real Mistral draft on
    13 Sept, which invented a working-looking /external/G200386250 link.
    """
    permitted = {_canon_url(u) for u in (allowed or []) if u}
    good, bad = [], []
    for raw in _URL.findall(draft):
        (good if _canon_url(raw) in permitted else bad).append(raw.rstrip(".,;:!?"))
    return good, bad


def verify_grounding(draft: str, passage: str, extra_sources: list[str] | None = None,
                     allowed_urls: list[str] | None = None) -> dict:
    """Does the draft actually quote the source material it was given?

    Produces `draft_quotes_policy` for the gate. A model that ignores the
    quoting rule fails this check and Cedar F6 escalates the ticket, so the
    rule is enforced rather than merely requested.

    WHAT COUNTS AS GROUNDED
        The passage, plus anything else the prompt put in front of the model —
        the title and the heading path. A model that quotes the section heading
        it was shown has not invented anything, and failing it would be wrong.
        `extra_sources` carries those.

    WHAT STILL FAILS
        Any span that is not verbatim in the supplied material, after folding
        unicode punctuation and trimming the boundary punctuation the model
        added to fit its sentence. Paraphrase fails. Invention fails. And one
        unverified span fails the whole draft, so a fabricated quote cannot
        ride along beside a real one.
    """
    if draft.strip() == REFUSAL:
        return {
            "draft_quotes_policy": False,
            "refused": True,
            "quotes_found": 0,
            "verified": [],
            "unverified": [],
            "detail": "model declined to answer — no grounding to check",
        }

    haystack = " || ".join(_normalize(s) for s in ([passage] + list(extra_sources or [])) if s)
    verified, unverified = [], []
    for q in _QUOTE.findall(draft):
        needle = _normalize(q).strip(_EDGE)
        (verified if needle and needle in haystack else unverified).append(q)

    # A link the model invented is as ungrounded as a quote it invented, and
    # more damaging — the seller clicks it. URLs inside the passage itself are
    # fine; the model was shown them.
    permitted = list(allowed_urls or []) + _URL.findall(passage)
    good_urls, bad_urls = check_urls(draft, permitted)

    ok = bool(verified) and not unverified and not bad_urls

    if bad_urls:
        detail = "%d invented URL(s): %s" % (len(bad_urls), ", ".join(u[:60] for u in bad_urls[:2]))
    elif not verified and not unverified:
        detail = "no quoted span in the draft"
    elif unverified:
        detail = "%d quoted span(s) not found in the passage" % len(unverified)
    else:
        detail = "%d quoted span(s) verified against the passage" % len(verified)
        if good_urls:
            detail += ", %d link(s) match the citation" % len(good_urls)

    return {
        "draft_quotes_policy": ok,
        "refused": False,
        "quotes_found": len(verified) + len(unverified),
        "verified": verified,
        "unverified": unverified,
        "urls_ok": good_urls,
        "urls_invented": bad_urls,
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# Escalation handover note
# ---------------------------------------------------------------------------

ESCALATION_SYSTEM = """\
You write internal handover notes for an Amazon Seller Support agent picking up
an escalated ticket. This is never shown to the seller.

Exactly four lines, each starting with the given label, no preamble:

Asked: what the seller wants, in one sentence, in your own words.
Found: the policy passage retrieved, named by its section, and whether it
       actually answers the question.
Held:  the rule that stopped auto-send and why, copied from what you are given.
Gap:   what a human still needs to determine that the passage does not settle.

Be blunt. If retrieval found nothing useful, say so plainly.\
"""


def build_escalation_request(ticket_text: str, hit: dict | None, gate_result: dict) -> dict:
    found = (
        "%s — %s" % (hit.get("title"), hit.get("heading_path"))
        if hit
        else "nothing above the confidence floor"
    )
    user = (
        "TICKET\n%s\n\nRETRIEVED\n%s\n\nCONFIDENCE\n%.4f (floor 0.72)\n\n"
        "GATE DECISION\n%s\n%s\n"
        % (
            ticket_text,
            found,
            gate_result.get("retrieval_confidence", 0.0),
            ", ".join(gate_result.get("blocked_by", [])) or "none",
            " ".join(gate_result.get("reasons", [])),
        )
    )
    return {
        "modelId": CLASSIFY_MODEL,
        "system": [{"text": ESCALATION_SYSTEM}],
        "messages": [{"role": "user", "content": [{"text": user}]}],
        "inferenceConfig": {"temperature": 0.0, "maxTokens": 250},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

PASSAGE = (
    "We require sellers to maintain an Order Defect Rate (ODR) less than 1% to sell in "
    "the Amazon store. An ODR above 1% may result in suspension of your Fulfilled by "
    "Merchant selling privileges."
)
CITATION = "https://sellercentral.amazon.com/help/hub/reference/external/G200285170?locale=en-US"

CASES = [
    (
        'Your account shows an Order Defect Rate above the threshold. Amazon policy states '
        'that sellers must maintain an "Order Defect Rate (ODR) less than 1%" to continue '
        'selling.',
        True, "verbatim quote present",
    ),
    (
        'Amazon requires you to keep your defect rate "below one percent or thereabouts" '
        'to stay active.',
        False, "quote is a paraphrase, not in the passage",
    ),
    (
        "Your ODR is too high. Please improve it to avoid problems with your account.",
        False, "no quoted span at all — the template-reply failure mode",
    ),
    (
        'Policy states an "Order Defect Rate (ODR) less than 1%" is required, and that an '
        '"ODR above 1% may result in suspension" of selling privileges.',
        True, "two verbatim quotes",
    ),
    (
        'Policy requires an "Order Defect Rate (ODR) less than 1%", and funds are '
        '"released within 24 hours" once resolved.',
        False, "one real quote, one fabricated — must fail",
    ),
    (REFUSAL, False, "refusal routes to a human"),
    (
        'Policy requires an "Order Defect Rate (ODR) less than 1%". See '
        'https://sellercentral.amazon.com/help/hub/reference/external/G200285170',
        True, "citation URL with the query string dropped - same page",
    ),
    (
        'Policy requires an "Order Defect Rate (ODR) less than 1%". See also '
        '[reserves](https://sellercentral.amazon.com/help/hub/reference/external/G200386250)',
        False, "INVENTED URL - different article id, must fail",
    ),
    (
        'Policy requires an "Order Defect Rate (ODR) less than 1%". More at '
        'https://amazon-seller-help.example.com/odr',
        False, "invented host, must fail",
    ),
]


def self_test() -> int:
    print("grounding check — the mechanism behind Cedar F6\n")
    failed = 0
    for draft, expected, label in CASES:
        got = verify_grounding(draft, PASSAGE, allowed_urls=[CITATION])
        ok = got["draft_quotes_policy"] == expected
        print("%s  quotes=%-5s %-46s %s"
              % ("ok  " if ok else "FAIL", got["draft_quotes_policy"], label, got["detail"]))
        if not ok:
            failed += 1
    print("\n%d/%d passed" % (len(CASES) - failed, len(CASES)))

    print("\nnote: two cases carry the weight.")
    print("  - one real quote + one fabricated quote must FAIL, so a plausible")
    print("    invention cannot ride along beside a real one.")
    print("  - a perfectly quoted draft carrying an INVENTED LINK must also fail.")
    print("    Observed in a real Mistral draft on 13 Sept: it produced a")
    print("    working-looking /external/G200386250 URL that the quote check")
    print("    could not see. The seller would have clicked it.")
    return failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-prompts", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(1 if self_test() else 0)

    if args.show_prompts:
        hit = {
            "title": "Order Defect Rate",
            "heading_path": "Order Defect Rate > Policy",
            "source_url": "https://sellercentral.amazon.com/help/hub/reference/external/G200285170",
            "text": PASSAGE,
            "rules": [{"metric": "ODR", "operator": "less than", "value": 1.0, "unit": "%"}],
        }
        req = build_draft_request("my odr went above the limit, what happens to my account?", hit)
        print("=" * 72)
        print("DRAFT — system (%s)" % DRAFT_MODEL)
        print("=" * 72)
        print(req["system"][0]["text"])
        print("\n" + "=" * 72)
        print("DRAFT — user")
        print("=" * 72)
        print(req["messages"][0]["content"][0]["text"])
        print("\n" + "=" * 72)
        print("CLASSIFY — system (%s)" % CLASSIFY_MODEL)
        print("=" * 72)
        print(CLASSIFY_SYSTEM)
        return

    ap.error("use --show-prompts or --self-test")


if __name__ == "__main__":
    main()
