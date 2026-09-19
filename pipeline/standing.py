"""Account standing — the seller's own numbers against published limits.

    python pipeline/standing.py "my ODR is 1.2% and my VTR is 91%"
    python pipeline/standing.py --self-test     # every quote, re-checked against the corpus
    python pipeline/standing.py --table         # the limits, with their sources

WHAT THIS STEP IS FOR
    Amazon runs a seller's shop on scorecards. Ship late too often, skip
    tracking numbers, cancel too many orders, and things are taken away: the
    Prime badge first, the right to sell eventually. The limits are published,
    and they are in our corpus. Most sellers do not know them, and find out
    when something has already gone.

    So when a ticket contains the seller's own numbers, we check them and say
    where they stand. That is the only honest form of "help me get seen more"
    this corpus supports: on Amazon, visibility is eligibility.

NO MODEL RUNS HERE, AND THAT IS THE POINT
    The numbers are pulled out with regexes and compared with `<` and `>=` in
    Python. A language model is confident and wrong at arithmetic, and this is
    arithmetic with a seller's business on the end of it. The model is told the
    finished comparison and is forbidden to redo it — see drafting.py.

    Same discipline as gate/topics.py: the tables below ARE the spec, readable
    by someone who does not read Python.

WHERE THE LIMITS COME FROM
    Every threshold carries the verbatim sentence that states it, the chunk it
    came from and the page it is on. Nothing here is typed from memory.
    `--self-test` re-reads chunks.jsonl and fails if the corpus has stopped
    saying any of it, so the table cannot quietly drift away from the source.

A DELIBERATE LIMITATION
    We check what the seller tells us. We do not have their real metrics —
    no Selling Partner API in this build — so a seller who mis-types their own
    ODR gets an answer about the number they typed. The reply says which
    numbers it used, which is what makes that safe to send.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "chunks.jsonl"


# ---------------------------------------------------------------------------
# The metrics we can read out of a ticket
# ---------------------------------------------------------------------------
# metric key -> how a seller writes it. Matched case-insensitively on word
# boundaries. Abbreviations are how sellers actually write these in the forums
# ("my ODR is 1.4%"), so they matter as much as the full names.

ALIASES: dict[str, list[str]] = {
    "order_defect_rate": [r"odr", r"order defect rate"],
    "late_shipment_rate": [r"lsr", r"late shipment rate", r"late dispatch rate"],
    "cancellation_rate": [r"\bcr\b", r"cancell?ation rate", r"pre-?fulfill?ment cancell?ation rate"],
    "valid_tracking_rate": [r"vtr", r"valid tracking rate", r"tracking rate"],
    "on_time_delivery": [r"\botd\b", r"on[- ]time delivery(?: rate)?"],
}

LABELS: dict[str, str] = {
    "order_defect_rate": "Order Defect Rate",
    "late_shipment_rate": "Late Shipment Rate",
    "cancellation_rate": "Cancellation Rate",
    "valid_tracking_rate": "Valid Tracking Rate",
    "on_time_delivery": "On-Time Delivery Rate",
}


# ---------------------------------------------------------------------------
# The limits
# ---------------------------------------------------------------------------
# kind:
#   "max" — the metric must stay BELOW value (defects, lateness, cancellations)
#   "min" — the metric must stay AT OR ABOVE value (tracking, on-time delivery)
#
# tier:
#   "selling" — the floor for selling in the Amazon store at all
#   "prime"   — the extra bar for the Prime badge on seller-fulfilled orders
#
# quote/chunk_id/title/url: the sentence in the corpus that says so. --self-test
# proves each quote is still present in that chunk.

THRESHOLDS: list[dict] = [
    {
        "metric": "order_defect_rate", "tier": "selling", "kind": "max", "value": 1.0,
        "at_stake": "selling in the Amazon store",
        "quote": "We require sellers to maintain their ODR less than 1% to sell in the Amazon store.",
        "chunk_id": "help_G200285170#0582",
        "title": "Order Defect Rate",
        "url": "https://sellercentral.amazon.com/help/hub/reference/external/G200285170?locale=en-US",
    },
    {
        "metric": "late_shipment_rate", "tier": "selling", "kind": "max", "value": 4.0,
        "at_stake": "selling in the Amazon store",
        "quote": "We require sellers to maintain a LSR less than 4% to sell in the Amazon store.",
        "chunk_id": "help_G200285190#0621",
        "title": "Late Shipment Rate",
        "url": "https://sellercentral.amazon.com/help/hub/reference/external/G200285190?locale=en-US",
    },
    {
        "metric": "cancellation_rate", "tier": "selling", "kind": "max", "value": 2.5,
        "at_stake": "selling in the Amazon store",
        "quote": "We require sellers to maintain a CR less than 2.5% to sell in the Amazon store.",
        "chunk_id": "help_G200285210#0289",
        "title": "Cancellation Rate",
        "url": "https://sellercentral.amazon.com/help/hub/reference/external/G200285210?locale=en-US",
    },
    {
        "metric": "valid_tracking_rate", "tier": "selling", "kind": "min", "value": 95.0,
        "at_stake": "selling seller-fulfilled orders in this category",
        "quote": "All sellers shipping seller-fulfilled orders (non-Fulfillment by Amazon) must "
                 "maintain a VTR greater than or equal to 95% at a product category level.",
        "chunk_id": "help_G201817070#0641",
        "title": "Valid tracking rate (VTR)",
        "url": "https://sellercentral.amazon.com/help/hub/reference/external/G201817070?locale=en-US",
    },
    {
        "metric": "on_time_delivery", "tier": "selling", "kind": "min", "value": 90.0,
        "at_stake": "the on-time delivery requirement",
        "quote": "Sellers must meet the 90% on-time delivery rate requirement without promise extension.",
        "chunk_id": "help_G200847280#0676",
        "title": "On-time delivery rate",
        "url": "https://sellercentral.amazon.com/help/hub/reference/external/G200847280?locale=en-US",
    },
    {
        "metric": "on_time_delivery", "tier": "prime", "kind": "min", "value": 93.5,
        "at_stake": "the Prime badge on seller-fulfilled orders",
        "quote": "The on-time delivery requirement currently in effect is 93.5% or higher",
        "chunk_id": "help_G202072550#0593",
        "title": "Seller Fulfilled Prime performance requirements",
        "url": "https://sellercentral.amazon.com/help/hub/reference/external/G202072550?locale=en-US",
    },
    {
        "metric": "valid_tracking_rate", "tier": "prime", "kind": "min", "value": 99.0,
        "at_stake": "the Prime badge on seller-fulfilled orders",
        "quote": "The valid tracking rate requirement currently in effect is 99% or higher",
        "chunk_id": "help_G202072550#0594",
        "title": "Seller Fulfilled Prime performance requirements",
        "url": "https://sellercentral.amazon.com/help/hub/reference/external/G202072550?locale=en-US",
    },
    {
        "metric": "cancellation_rate", "tier": "prime", "kind": "max", "value": 0.5,
        "at_stake": "the Prime badge on seller-fulfilled orders",
        "quote": "The cancellation rate requirement currently in effect is 0.5% or less",
        "chunk_id": "help_G202072550#0595",
        "title": "Seller Fulfilled Prime performance requirements",
        "url": "https://sellercentral.amazon.com/help/hub/reference/external/G202072550?locale=en-US",
    },
]

# How near the line counts as "close". One percentage point either side: close
# enough that a bad week crosses it, far enough that it is not noise.
MARGIN = 1.0


# ---------------------------------------------------------------------------
# Reading the numbers out of the ticket
# ---------------------------------------------------------------------------

_ALIAS_RE = [
    (metric, re.compile(r"\b%s\b" % pat, re.IGNORECASE))
    for metric, pats in ALIASES.items()
    for pat in pats
]

# 1.2%  ·  1.2 %  ·  1.2 percent  ·  91 per cent
_NUMBER_RE = re.compile(r"(\d{1,3}(?:\.\d{1,2})?)\s*(?:%|percent\b|per cent\b)", re.IGNORECASE)

# A number this far after the metric name is still about that metric
# ("my order defect rate has climbed to 1.4%"). Before it, less room: a
# trailing form ("1.4% ODR") puts the two side by side.
LOOK_AHEAD = 60
LOOK_BEHIND = 20


def extract(text: str) -> list[dict]:
    """Every metric reading in the ticket, with the words that produced it.

    Each percentage is attached to the nearest metric name that precedes it,
    and only if nothing else claimed that name first. A number with no metric
    near it is ignored — "I refunded 3% of orders" is not a scorecard.
    """
    mentions = []
    for metric, rx in _ALIAS_RE:
        for m in rx.finditer(text):
            mentions.append({"metric": metric, "start": m.start(), "end": m.end(),
                             "phrase": m.group(0)})
    if not mentions:
        return []
    # Longest alias wins where two overlap ("cancellation rate" over "cr").
    mentions.sort(key=lambda d: (d["start"], -(d["end"] - d["start"])))
    kept: list[dict] = []
    for m in mentions:
        if kept and m["start"] < kept[-1]["end"]:
            continue
        kept.append(m)

    out: list[dict] = []
    taken: set[int] = set()
    for num in _NUMBER_RE.finditer(text):
        before = [m for m in kept if m["end"] <= num.start()
                  and num.start() - m["end"] <= LOOK_AHEAD]
        after = [m for m in kept if m["start"] >= num.end()
                 and m["start"] - num.end() <= LOOK_BEHIND]
        owner = before[-1] if before else (after[0] if after else None)
        if owner is None or id(owner) in taken:
            continue
        taken.add(id(owner))
        out.append({
            "metric": owner["metric"],
            "label": LABELS[owner["metric"]],
            "value": float(num.group(1)),
            # The audit trail: the seller's own words that produced this number.
            "matched": text[owner["start"]:num.end()].strip(),
        })
    return out


# ---------------------------------------------------------------------------
# Comparing them
# ---------------------------------------------------------------------------


def _status(value: float, kind: str, limit: float) -> str:
    if kind == "max":
        if value >= limit:
            return "breach"
        return "close" if value >= limit - MARGIN else "meets"
    if value < limit:
        return "breach"
    return "close" if value < limit + MARGIN else "meets"


def assess(readings: list[dict]) -> list[dict]:
    """One finding per (reading, published limit). Plain `<` and `>=`, no model."""
    findings = []
    for r in readings:
        for t in THRESHOLDS:
            if t["metric"] != r["metric"]:
                continue
            status = _status(r["value"], t["kind"], t["value"])
            findings.append({
                "metric": r["metric"],
                "label": r["label"],
                "value": r["value"],
                "matched": r["matched"],
                "tier": t["tier"],
                "kind": t["kind"],
                "limit": t["value"],
                "at_stake": t["at_stake"],
                "status": status,
                "sentence": _sentence(r, t, status),
                "quote": t["quote"],
                "title": t["title"],
                "chunk_id": t["chunk_id"],
                "url": t["url"],
            })
    # Worst first, so the console and the prompt lead with what matters.
    order = {"breach": 0, "close": 1, "meets": 2}
    return sorted(findings, key=lambda f: (order[f["status"]], f["tier"] != "selling",
                                           f["label"]))


def _sentence(reading: dict, threshold: dict, status: str) -> str:
    """The finding in plain words. This is what the console shows an agent.

    A ceiling and a floor do not read the same way round. "above the 1% limit"
    and "below the 95% needed" are both failures, and writing both as "below
    the limit" would have an agent reading the wrong direction at a glance.
    """
    top = threshold["kind"] == "max"
    value = _pct(reading["value"])
    # "the 1% limit for selling in the store" / "the 95% needed to sell"
    bar = "%s %s for %s" % (_pct(threshold["value"]),
                            "limit" if top else "needed",
                            threshold["at_stake"])
    if status == "breach":
        return "%s of %s is %s the %s." % (
            reading["label"], value, "above" if top else "below", bar)
    if status == "close":
        return "%s of %s stays %s the %s, but only just." % (
            reading["label"], value, "under" if top else "above", bar)
    return "%s of %s is %s the %s." % (
        reading["label"], value, "within" if top else "comfortably above", bar)


def _pct(v: float) -> str:
    return ("%.1f%%" % v).replace(".0%", "%")


def check(text: str) -> dict:
    """The whole step: read the numbers, compare them, say what it means.

    `breach_selling` is the one field the gate reads. A seller who is under a
    floor that decides whether they can sell at all is in enforcement
    territory, and an automated reply is not what that moment needs. Missing
    the Prime bar is not the same thing — that is a fact we can quote, so it
    stays auto-sendable.
    """
    readings = extract(text)
    findings = assess(readings)
    breaches = [f for f in findings if f["status"] == "breach"]
    return {
        "checked": bool(readings),
        "readings": readings,
        "findings": findings,
        "breach_selling": any(f["tier"] == "selling" for f in breaches),
        "breach_prime": any(f["tier"] == "prime" for f in breaches),
        "summary": summarize(readings, findings),
    }


def summarize(readings: list[dict], findings: list[dict]) -> str:
    if not readings:
        return "No account numbers in this ticket — nothing to check."
    breaches = [f for f in findings if f["status"] == "breach"]
    close = [f for f in findings if f["status"] == "close"]
    n = len(readings)
    counted = "%d number%s checked" % (n, "" if n == 1 else "s")
    if breaches:
        worst = breaches[0]
        return "%s. %s is under the line for %s." % (counted, worst["label"], worst["at_stake"])
    if close:
        return "%s. Everything clears, but %s is close to the line." % (counted, close[0]["label"])
    return "%s. Every one clears its published limit." % counted


def step_detail(result: dict) -> dict:
    """What goes in the decision trail. The findings, not the whole table."""
    return {
        "checked": result["checked"],
        "summary": result["summary"],
        "breach_selling": result["breach_selling"],
        "breach_prime": result["breach_prime"],
        "readings": result["readings"],
        "findings": [
            {k: f[k] for k in
             ("metric", "label", "value", "tier", "limit", "kind", "status",
              "at_stake", "sentence", "matched", "title", "chunk_id", "url")}
            for f in result["findings"]
        ],
    }


def prompt_block(result: dict) -> str:
    """The findings, handed to the drafter as facts it may not recompute."""
    if not result["checked"]:
        return ""
    lines = [
        "",
        "THE SELLER'S OWN NUMBERS, ALREADY CHECKED AGAINST PUBLISHED LIMITS",
        "These comparisons were computed in code, not by you. Restate them if",
        "they are relevant. Never recalculate one, never contradict one, and",
        "never add a limit that is not listed here.",
    ]
    for f in result["findings"]:
        lines.append("  - %s  [%s]" % (f["sentence"], f["status"]))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Self-test — the table against the corpus it claims to come from
# ---------------------------------------------------------------------------


def _flat(s: str) -> str:
    return " ".join(s.split()).replace("’", "'").lower()


def self_test() -> int:
    if not CHUNKS.exists():
        print("chunks.jsonl not found at %s" % CHUNKS)
        return 1

    wanted = {t["chunk_id"] for t in THRESHOLDS}
    have: dict[str, str] = {}
    with CHUNKS.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["chunk_id"] in wanted:
                have[row["chunk_id"]] = _flat(row["text"])

    print("every published limit, re-read from chunks.jsonl\n")
    failed = 0
    for t in THRESHOLDS:
        body = have.get(t["chunk_id"])
        ok = body is not None and _flat(t["quote"]) in body
        print("%s %-22s %-8s %-6s %s" % (
            "ok  " if ok else "FAIL", t["metric"], t["tier"],
            _pct(t["value"]), t["chunk_id"]))
        if not ok:
            failed += 1
            print("       quote not found in that chunk: %r" % t["quote"][:70])

    print()
    for text, expect in _EXTRACTION_CASES:
        got = [(r["metric"], r["value"]) for r in extract(text)]
        ok = got == expect
        print("%s %-52s %s" % ("ok  " if ok else "FAIL", text[:52], got))
        if not ok:
            failed += 1
            print("       expected %s" % (expect,))

    print()
    for text, expect_selling, expect_prime, label in _GATE_CASES:
        out = check(text)
        ok = (out["breach_selling"], out["breach_prime"]) == (expect_selling, expect_prime)
        print("%s selling=%-5s prime=%-5s %s" % (
            "ok  " if ok else "FAIL", out["breach_selling"], out["breach_prime"], label))
        if not ok:
            failed += 1
            print("       expected selling=%s prime=%s" % (expect_selling, expect_prime))

    print("\n%d failure(s)" % failed)
    return 1 if failed else 0


# breach_selling is the only field the Cedar gate reads. Missing the Prime bar
# must never set it — that costs the badge, not the business, and holding back
# a quotable answer about it would make the whole check useless to the sellers
# it helps most. Pinned here because it is a one-word edit away from being
# wrong and no example test above would notice.
_GATE_CASES: list[tuple[str, bool, bool, str]] = [
    ("my ODR is 1.4%", True, False, "selling floor missed"),
    ("my on-time delivery is 92%", False, True, "Prime bar missed, selling floor fine"),
    ("my valid tracking rate is 97%", False, True, "same, on tracking"),
    ("my ODR is 0.4% and my cancellation rate is 0.2%", False, False, "everything clears"),
    ("how do I improve my ODR", False, False, "no numbers, nothing to breach"),
]


_EXTRACTION_CASES: list[tuple[str, list]] = [
    ("my ODR is 1.2%", [("order_defect_rate", 1.2)]),
    ("my ODR is 1.2% and my VTR is 91%",
     [("order_defect_rate", 1.2), ("valid_tracking_rate", 91.0)]),
    ("late shipment rate of 5 percent", [("late_shipment_rate", 5.0)]),
    ("on-time delivery has dropped to 88.5%", [("on_time_delivery", 88.5)]),
    ("94% on-time delivery rate this month", [("on_time_delivery", 94.0)]),
    # No metric named — not a scorecard number.
    ("I refunded 3% of my orders last week", []),
    # Named, but no number anywhere near it.
    ("what is a good order defect rate", []),
]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("text", nargs="*")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(self_test())

    if args.table:
        print("%-22s %-8s %-7s %s" % ("metric", "tier", "limit", "source"))
        for t in THRESHOLDS:
            print("%-22s %-8s %-7s %s" % (
                t["metric"], t["tier"],
                ("< " if t["kind"] == "max" else ">= ") + _pct(t["value"]),
                t["title"]))
        return

    if not args.text:
        print('usage: python pipeline/standing.py "my ODR is 1.2%"')
        raise SystemExit(1)

    out = check(" ".join(args.text))
    print(out["summary"])
    if not out["checked"]:
        return
    print()
    for f in out["findings"]:
        print("  %-6s %s" % (f["status"], f["sentence"]))
        print("         %s — %s" % (f["title"], f["url"]))
    print("\nbreach of a selling floor: %s" % out["breach_selling"])


if __name__ == "__main__":
    main()
