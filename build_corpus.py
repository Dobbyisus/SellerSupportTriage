#!/usr/bin/env python3
"""
build_corpus.py — Seller Support Triage Console, corpus collection tool.

Collects public Amazon Seller Central policy documents into a local folder,
records the source URL beside every file, and reports which of the priority
ticket categories are actually covered.

SCOPE: collection and verification only. This script does not chunk, embed,
index, or retrieve. Those are build-time concerns (Sept 17+).

Enforced constraints (from policy-corpus-sources.md):
  - US marketplace only. Any URL whose region prefix is not G/01 is refused.
  - Any filename containing REDLINE is skipped (tracked-change drafts, garbled).
  - First-party Amazon sources only. No blogs, no agency guides.
  - Source URL + fetch timestamp recorded per document in manifest.json.

Usage:
    python build_corpus.py --probe            # HEAD-check Tier A + Tier B PDFs
    python build_corpus.py --download         # download whatever resolved
    python build_corpus.py --crawl            # enumerate help-hub page IDs
    python build_corpus.py --render           # render help-hub pages to text
    python build_corpus.py --report           # gap check across categories
    python build_corpus.py --all              # everything, in order

Setup:
    pip install requests beautifulsoup4 pdfplumber playwright
    playwright install chromium
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

# Amazon page titles carry en-dashes and smart quotes. The default Windows
# console codepage (cp1252) raises UnicodeEncodeError on them mid-crawl.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace",
                           line_buffering=True)
except Exception:
    pass

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

OUT = Path("SellerSupport_Docs")
PDF_DIR = OUT / "pdf"
HELP_DIR = OUT / "help"
TEXT_DIR = OUT / "text"          # plain text, for reading during the gap check
MANIFEST = OUT / "manifest.json"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " \
     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"

PAUSE = 1.5  # seconds between requests; be a polite guest

MEDIA = "https://m.media-amazon.com/images"
HELP_HUB = "https://sellercentral.amazon.com/help/hub/reference/external/{}?locale=en-US"

# Tier A — verified live Sept 8 2026.
# NOTE: the trailing dot in "help./" is REAL. Do not "correct" it; it 404s.
TIER_A = [
    (f"{MEDIA}/G/01/rainier/help./1.1_BSA_PDF_English.pdf",
     "Amazon Services Business Solutions Agreement (US)"),
]

# Tier B — filenames seen under other region prefixes. G/01 equivalents unconfirmed.
# Both path forms are probed because the trailing-dot convention is inconsistent.
TIER_B_NAMES = [
    "Account_Health_Rating_program_policy_and_FAQ_EN.pdf",
    "AtoZ_Guarantee_Program_Policy_PDF.pdf",
    "Selling_Policies_and_Seller_Code_of_Conduct_EN.pdf",
    "Selling_Policies_and_Seller_Code_of_Conduct_English.pdf",
    "Selling_Policies_and_Seller_Code_of_Conduct_PDF.pdf",
]
TIER_B_PREFIXES = [f"{MEDIA}/G/01/rainier/help./", f"{MEDIA}/G/01/rainier/help/"]

# Seen under G/01 in a real path. Governs seller->buyer messaging — include if it
# resolves, but do not let it become the demo centrepiece (see corpus notes).
TIER_B_EXTRA = [
    (f"{MEDIA}/G/01/SellerCentral/CommunicationGuidelines/en_US_Communication_Guidelines.pdf",
     "Communication Guidelines (US) — seller-to-buyer messaging"),
]

# Tier C — confirmed help-hub node IDs. No PDF equivalent exists for these.
TIER_C_SEEDS = {
    "G1801": "Selling Policies and Seller Code of Conduct",
    "G200205250": "Account Health Rating program policy",
    "201824360": "Terms of Service",
}

# Crawl roots — harvest /help/hub/reference/external/G... links from these.
# NOTE: "https://sellercentral.amazon.com/help/hub/reference?locale=en-US" and
# the /help/hub/search endpoint both 302 to /ap/signin. They harvest ZERO nodes.
# That is why the first crawl returned only the Program Policies branch: every
# node came from the sidebars of G1801 and G200205250. Individual
# /external/G... pages are public. G2 is the help ROOT as a public node page,
# and G521 is Program Policies. Seed from those and walk the link graph.
CRAWL_ROOTS = [
    HELP_HUB.format("G2"),
    HELP_HUB.format("G521"),
    HELP_HUB.format("G1801"),
    HELP_HUB.format("G200205250"),
]

# Branch indexes reached from G2. These are the routes to the operational layer
# the Program Policies branch does not carry.
EXPAND_SEEDS = {
    "G2": "Seller Central Help (root)",
    "G69033": "Reference",
    "G69036": "Selling on Amazon reference",
    "G69032": "Tasks & tools",
    "G200342080": "Seller-fulfilled shipping",
    "G69126": "Seller-fulfilled returns, refunds, cancellations, and claims",
    "G200421970": "Selling on Amazon",
    "G200200040": "The order process",
}
MAX_EXPAND_VISITS = 700  # ceiling on link-harvest page loads per --expand run

# Frontier ordering. A ceilinged walk spends its budget in whatever order the
# frontier happens to be in, so bias it: hub/index pages first (they fan out
# widest), then titles touching a priority category, then everything else.
# This makes an interrupted walk still useful rather than merely alphabetical.
PRIORITY_HINTS = re.compile(
    r"performance|metric|shipment|shipping|tracking|defect|handling|deliver|"
    r"cancellation|fulfil|payment|disburs|reserve|remitt|payout|paid|fund|"
    r"suppress|listing|detail page|inactive|stranded|return|refund|a-to-z|"
    r"claim|chargeback|appeal|suspend|deactivat|account health|policy|polices",
    re.I)
HUB_HINTS = re.compile(r"reference|help|overview|index|guide|tasks|tools|"
                       r"about|resources|policies", re.I)


def _frontier_order(nid: str, title: str) -> tuple:
    """Sort key: hubs first, then priority-category titles, then the rest."""
    return (0 if HUB_HINTS.search(title) else 1,
            0 if PRIORITY_HINTS.search(title) else 1,
            title.lower())

# Gap check. Each priority category needs at least one document that actually
# answers a ticket of that type. Terms are deliberately narrow — a document that
# merely mentions "payment" once should not count as payments coverage.
CATEGORIES = {
    "payments": ["disbursement", "reserve", "payment hold", "withhold",
                 "remittance", "settlement period", "funds availability"],
    "shipping": ["late shipment rate", "valid tracking rate", "handling time",
                 "order defect rate", "on-time delivery", "ship by date"],
    "listings": ["suppress", "category restriction", "listing removal",
                 "restricted product", "detail page", "gated"],
    "returns":  ["return window", "refund", "return authorization",
                 "returnless", "prepaid return label", "a-to-z"],
}
MIN_HITS = 3  # distinct terms a document must hit to count for a category

# Liveness threshold for a rendered help page. See corpus-collection findings 5.4:
# the "enable JavaScript" <noscript> string SURVIVES successful hydration, so it
# is not a liveness test. Visible body text length is.
MIN_TEXT = 600


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------

def check_region(url: str) -> None:
    """Refuse anything outside the US marketplace prefix."""
    if "m.media-amazon.com" not in url:
        return
    m = re.search(r"/images/G/(\d{2})/", url)
    if not m:
        raise ValueError(f"No region prefix found in media URL: {url}")
    if m.group(1) != "01":
        raise ValueError(
            f"REFUSED: region G/{m.group(1)} is a different legal document "
            f"than G/01. Mixing regions puts contradictory passages in one "
            f"index.\n  {url}"
        )


def is_redline(name: str) -> bool:
    return "REDLINE" in name.upper()


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"documents": {}, "probe_results": {}, "help_ids": {}}


def save_manifest(m: dict) -> None:
    OUT.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Probe
# --------------------------------------------------------------------------

def probe() -> dict:
    """HEAD-check every candidate URL. Report which resolve."""
    m = load_manifest()
    candidates: list[tuple[str, str]] = list(TIER_A) + list(TIER_B_EXTRA)

    for name in TIER_B_NAMES:
        for prefix in TIER_B_PREFIXES:
            candidates.append((prefix + name, f"Tier B candidate: {name}"))

    print(f"Probing {len(candidates)} candidate URLs...\n")
    resolved = 0

    for url, label in candidates:
        fname = Path(urlparse(url).path).name
        if is_redline(fname):
            print(f"  SKIP (REDLINE)  {fname}")
            continue
        try:
            check_region(url)
        except ValueError as e:
            print(f"  {e}")
            continue

        try:
            r = requests.head(url, headers={"User-Agent": UA},
                              timeout=20, allow_redirects=True)
            ok = r.status_code == 200
            size = int(r.headers.get("content-length", 0))
            ctype = r.headers.get("content-type", "")
            # A 200 that returns HTML is an error page, not a PDF.
            if ok and "pdf" not in ctype.lower() and url.endswith(".pdf"):
                ok = False
                note = f"200 but content-type={ctype}"
            else:
                note = f"{size // 1024} KB" if ok else str(r.status_code)
        except requests.RequestException as e:
            ok, note = False, type(e).__name__

        m["probe_results"][url] = {"ok": ok, "note": note,
                                   "label": label, "checked": now()}
        print(f"  {'OK  ' if ok else 'MISS'}  {note:<14} {fname}")
        resolved += ok
        time.sleep(PAUSE)

    save_manifest(m)
    print(f"\n{resolved} of {len(candidates)} resolved. "
          f"Run --download to fetch them.")
    return m


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

def download() -> None:
    m = load_manifest()
    hits = [(u, d) for u, d in m["probe_results"].items() if d["ok"]]
    if not hits:
        print("Nothing resolved. Run --probe first.")
        return

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(hits)} documents...\n")

    for url, meta in hits:
        fname = Path(urlparse(url).path).name
        dest = PDF_DIR / fname
        if dest.exists():
            print(f"  HAVE  {fname}")
            continue
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            m["documents"][fname] = {
                "source_url": url,
                "label": meta["label"],
                "sha256": hashlib.sha256(r.content).hexdigest(),
                "bytes": len(r.content),
                "fetched": now(),
                "kind": "pdf",
            }
            print(f"  GOT   {fname}  ({len(r.content) // 1024} KB)")
        except requests.RequestException as e:
            print(f"  FAIL  {fname}  {e}")
        time.sleep(PAUSE)

    save_manifest(m)
    extract_pdf_text()


def extract_pdf_text() -> None:
    """Plain text alongside each PDF, so the gap check can be done by reading."""
    try:
        import pdfplumber
    except ImportError:
        print("\npdfplumber not installed — skipping text extraction.")
        return

    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        out = TEXT_DIR / (pdf.stem + ".txt")
        if out.exists():
            continue
        try:
            with pdfplumber.open(pdf) as doc:
                text = "\n\n".join(p.extract_text() or "" for p in doc.pages)
            out.write_text(text, encoding="utf-8")
            print(f"  TEXT  {out.name}  ({len(text):,} chars)")
        except Exception as e:
            print(f"  TEXT FAIL  {pdf.name}  {e}")


# --------------------------------------------------------------------------
# Crawl + render (help-hub pages)
# --------------------------------------------------------------------------

def _browser():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("Playwright not installed.\n"
                 "  pip install playwright && playwright install chromium")
    return sync_playwright()


def _launch(p, headless: bool = True):
    """Real installed Chrome, not bundled Chromium.

    Findings 5.3: Seller Central's app shell loads under bundled Chromium but
    never hydrates. Real Chrome renders the same URL fine. Symptom if this
    regresses: EVERY page reports EMPTY. Universal failure means the renderer;
    scattered failure means dead nodes.
    """
    try:
        browser = p.chromium.launch(headless=headless, channel="chrome")
    except Exception as e:
        print(f"  ! Chrome channel unavailable ({e}); falling back to bundled "
              f"Chromium. Expect help pages to render EMPTY.")
        browser = p.chromium.launch(headless=headless)
    ctx = browser.new_context(
        user_agent=UA,
        viewport={"width": 1400, "height": 900},
        locale="en-US",
    )
    return browser, ctx


def crawl() -> None:
    """Render the help index pages and harvest every external/G... node ID."""
    from bs4 import BeautifulSoup

    m = load_manifest()
    found: dict[str, str] = dict(TIER_C_SEEDS)

    with _browser() as p:
        browser, ctx = _launch(p)
        page = ctx.new_page()
        for root in CRAWL_ROOTS:
            print(f"  CRAWL  {root}")
            try:
                page.goto(root, wait_until="networkidle", timeout=45000)
                page.wait_for_timeout(2500)
                soup = BeautifulSoup(page.content(), "html.parser")
                for a in soup.find_all("a", href=True):
                    hit = re.search(r"/help/hub/reference/external/(G?\d+)",
                                    a["href"])
                    if hit:
                        nid = hit.group(1)
                        found.setdefault(nid, a.get_text(strip=True) or nid)
            except Exception as e:
                print(f"    failed: {e}")
            time.sleep(PAUSE)
        browser.close()

    m["help_ids"] = found
    save_manifest(m)
    print(f"\n{len(found)} help-hub node IDs enumerated. Run --render.")



# --------------------------------------------------------------------------
# Text extraction (shared by --render)
# --------------------------------------------------------------------------

FURNITURE_RE = re.compile(
    r"select your preferred language|was this (article|page) helpful|"
    r"listen to this|podcast episode|watch this video|"
    r"sign in|quick solutions|thank you for your feedback|"
    r"^(english|中文|한국어|Français|Italiano|Deutsch|Português|Español|"
    r"हिन्दी|ไทย|"
    r"தமிழ்|Tiếng Việt|日本語)",
    re.I)


def _is_link_only(el) -> bool:
    """True if the element is just a link (nav/related-links), not prose.

    Seller Central appends a long list of bare policy titles to every help
    page. Extracted naively those land in the body, and a region check then
    sees "Amazon Brazil" inside a clean US document. Prose containing an
    inline link is kept -- only elements that are ENTIRELY link text go.
    """
    anchors = el.find_all("a")
    if not anchors:
        return False
    txt = el.get_text(" ", strip=True)
    atxt = " ".join(a.get_text(" ", strip=True) for a in anchors)
    return bool(txt) and len(atxt) >= 0.9 * len(txt)


def extract_lines(soup) -> str:
    """Heading-marked body text with page furniture removed."""
    for tag in soup(["script", "style", "nav", "footer", "noscript",
                     "iframe", "form", "header", "select", "option"]):
        tag.decompose()

    lines = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        txt = el.get_text(" ", strip=True)
        if not txt or len(txt) < 3:
            continue
        if FURNITURE_RE.search(txt):
            continue
        if _is_link_only(el):
            continue
        lines.append(f"{'#' * int(el.name[1])} {txt}" if el.name.startswith("h")
                     else txt)

    # De-duplicate consecutive repeats (Amazon renders some blocks twice).
    out = []
    for ln in lines:
        if not out or out[-1] != ln:
            out.append(ln)
    return chr(10).join(out)


def expand(hops: int = 2) -> None:
    """Breadth-first walk of the public help link graph to enumerate node IDs.

    Harvests IDs and titles only -- no text is saved. --render does that, and
    only for the curated selection. Rationale: the corpus target is 30-50
    hand-checked documents, so enumerate broadly, then choose.
    """
    m = load_manifest()
    known: dict[str, str] = dict(m.get("help_ids") or {})
    known.update(TIER_C_SEEDS)
    known.update(EXPAND_SEEDS)

    visited: set[str] = set(m.get("expanded_from") or [])
    frontier = sorted((n for n in known if n not in visited),
                      key=lambda n: _frontier_order(n, known.get(n, n)))
    visits = 0

    with _browser() as p:
        browser, ctx = _launch(p)
        page = ctx.new_page()
        for hop in range(1, hops + 1):
            if not frontier:
                break
            print("")
            print(f"--- hop {hop}: {len(frontier)} pages to harvest ---")
            nxt: list[str] = []
            for nid in frontier:
                if visits >= MAX_EXPAND_VISITS:
                    print(f"  ! visit ceiling {MAX_EXPAND_VISITS} reached; stopping")
                    frontier = []
                    break
                try:
                    page.goto(HELP_HUB.format(nid), wait_until="networkidle",
                              timeout=45000)
                    page.wait_for_timeout(2000)
                    pairs = page.eval_on_selector_all(
                        "a[href]",
                        "els=>els.map(e=>[e.getAttribute('href'), e.innerText.trim()])")
                except Exception as e:
                    print(f"  FAIL  {nid}  {e}")
                    visited.add(nid)
                    visits += 1
                    m["help_ids"] = known
                    m["expanded_from"] = sorted(visited)
                    save_manifest(m)
                    continue

                fresh = 0
                for href, txt in pairs:
                    if not href:
                        continue
                    hit = re.search(r"/help/hub/reference/external/(G?\d+)", href)
                    if not hit:
                        continue
                    new_id = hit.group(1)
                    if new_id not in known:
                        known[new_id] = (txt or new_id).strip()
                        nxt.append(new_id)
                        fresh += 1
                visited.add(nid)
                visits += 1
                print(f"  {nid:14} +{fresh:3} new   (total {len(known)})")

                # Checkpoint every page. Resumable by design: --expand skips
                # anything already in expanded_from, so a killed run costs one
                # page load, not the whole walk.
                m["help_ids"] = known
                m["expanded_from"] = sorted(visited)
                save_manifest(m)
                time.sleep(PAUSE)
            frontier = sorted((n for n in nxt if n not in visited),
                              key=lambda n: _frontier_order(n, known.get(n, n)))

        browser.close()

    m["help_ids"] = known
    m["expanded_from"] = sorted(visited)
    save_manifest(m)
    print("")
    print(f"{len(known)} node IDs known ({visits} pages harvested this run).")
    print("Review titles, then --render.")


def render(only: list[str] | None = None) -> None:
    """Render each help page and save heading-marked text + raw HTML.

    `only` restricts the run to specific node IDs, so a curated shortlist can
    be rendered without pulling every enumerated node.
    """
    from bs4 import BeautifulSoup

    m = load_manifest()
    ids = m.get("help_ids") or TIER_C_SEEDS
    if only:
        ids = {k: v for k, v in ids.items() if k in set(only)}
        missing = set(only) - set(ids)
        for k in sorted(missing):
            ids[k] = k  # allow rendering an ID not yet in the manifest
    HELP_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Rendering {len(ids)} help pages...\n")

    with _browser() as p:
        browser, ctx = _launch(p)
        page = ctx.new_page()

        for nid, label in sorted(ids.items()):
            url = HELP_HUB.format(nid)
            stem = f"help_{nid}"
            if (TEXT_DIR / f"{stem}.txt").exists():
                print(f"  HAVE  {nid}")
                continue
            try:
                page.goto(url, wait_until="networkidle", timeout=45000)
                page.wait_for_timeout(2500)
                html = page.content()

                # Findings 5.4: check VISIBLE text, not the <noscript> string --
                # that string remains in the DOM after successful hydration.
                try:
                    visible = page.inner_text("body")
                except Exception:
                    visible = ""
                if len(visible) < MIN_TEXT:
                    print(f"  EMPTY {nid}  ({len(visible)} visible chars)")
                    continue

                (HELP_DIR / f"{stem}.html").write_text(html, encoding="utf-8")

                soup = BeautifulSoup(html, "html.parser")

                # Mark headings so structure survives into the text file. This is
                # for human reading during the gap check — the real heading_path
                # chunker is build-time work, not here.
                body = extract_lines(soup)
                header = f"SOURCE: {url}\nTITLE: {label}\nFETCHED: {now()}\n{'-' * 60}\n"
                (TEXT_DIR / f"{stem}.txt").write_text(header + body, encoding="utf-8")

                m["documents"][stem] = {
                    "source_url": url, "label": label, "kind": "help_page",
                    "chars": len(body), "fetched": now(),
                }
                print(f"  GOT   {nid}  {label[:44]}  ({len(body):,} chars)")
            except Exception as e:
                print(f"  FAIL  {nid}  {e}")
            time.sleep(PAUSE)

        browser.close()

    save_manifest(m)


# --------------------------------------------------------------------------
# Gap report
# --------------------------------------------------------------------------

def report() -> None:
    """Which priority categories have a document that actually answers them."""
    files = sorted(TEXT_DIR.glob("*.txt"))
    if not files:
        print("No extracted text found. Run --download / --render first.")
        return

    coverage = {c: [] for c in CATEGORIES}
    for f in files:
        low = f.read_text(encoding="utf-8", errors="ignore").lower()
        for cat, terms in CATEGORIES.items():
            hits = [t for t in terms if t in low]
            if len(hits) >= MIN_HITS:
                coverage[cat].append((f.stem, len(hits)))

    print(f"\n{'=' * 64}\nCORPUS GAP REPORT — {len(files)} documents\n{'=' * 64}\n")

    gaps = []
    for cat in ["payments", "shipping", "listings", "returns"]:
        docs = sorted(coverage[cat], key=lambda x: -x[1])
        if docs:
            print(f"  {cat.upper():<10} {len(docs)} doc(s)")
            for stem, n in docs[:4]:
                print(f"{'':14}{stem}  ({n} terms)")
        else:
            print(f"  {cat.upper():<10} *** EMPTY ***")
            gaps.append(cat)
        print()

    print(f"{'-' * 64}")
    if gaps:
        print(f"GAPS: {', '.join(gaps)}")
        print("A ticket in an empty category will make retrieval confidently\n"
              "return the wrong passage — worse than returning nothing.\n"
              "Find a document for each gap, or drop that ticket type from the demo.")
    else:
        print("All four priority categories covered.")
    print("\nKeyword hits are a screen, not a verdict. Open the top document for\n"
          "each category and confirm it answers a real ticket of that type.")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--crawl", action="store_true")
    ap.add_argument("--expand", type=int, nargs="?", const=2, default=0,
                    metavar="HOPS", help="BFS the help link graph N hops")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--only-file", default="",
                    help="file of node IDs (first tab-separated field) to render")
    ap.add_argument("--only", default="",
                    help="comma-separated node IDs to restrict --render to")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    if not any(vars(a).values()):
        ap.print_help()
        return

    OUT.mkdir(exist_ok=True)
    if a.probe or a.all:
        probe()
    if a.download or a.all:
        download()
    if a.crawl or a.all:
        crawl()
    if a.expand:
        expand(a.expand)
    if a.render or a.all:
        sel = [x.strip() for x in a.only.split(",") if x.strip()]
        of = getattr(a, "only_file", "")
        if of:
            for line in Path(of).read_text(encoding="utf-8").splitlines():
                nid = line.split("	")[0].strip()
                if nid:
                    sel.append(nid)
        render(sel or None)
    if a.report or a.all:
        report()


if __name__ == "__main__":
    main()
