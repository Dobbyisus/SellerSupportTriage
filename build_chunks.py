#!/usr/bin/env python3
"""Build chunks.jsonl -- the unit OpenSearch actually indexes.

Takes the 45 documents in knowledge_base.json and turns them into retrievable
passages. Three things this does that the filter's chunker did not:

  1. SENTENCE-AWARE PACKING. The filter split on a word count, which cut mid
     sentence -- one chunk ended "...Select the Late shipment rate" and the next
     began "number. Always monitor...". Harmless for matching, but the chunk is
     what gets QUOTED to the seller and shown to the agent, and a citation that
     opens mid-word destroys the credibility the citation exists to build.
     Chunks now pack whole sentences and overlap by whole sentences.

  2. BSA SECTIONING. The agreement is 111k characters with no heading markers
     (it comes from a PDF). Chunked generically it matched everything weakly.
     It is now split on its own section markers so a citation reads
     "Business Solutions Agreement > General Terms §2 Service Fee Payments".

  3. SHORT-CHUNK MERGING. Flushing at every heading produced 45-word chunks,
     which are too small to match well. Anything under MIN_WORDS is merged back
     into its predecessor within the same section.

The knowledge base itself is FROZEN at 45 documents. This does not re-run
selection -- it only prepares those 45 for indexing.

Usage:
  python build_chunks.py             # build and report
  python build_chunks.py --refresh   # ignore the embedding cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

KB = Path("knowledge_base.json")
QUERIES = Path("filter_queries.tsv")
OUT_JSONL = Path("chunks.jsonl")
CACHE = Path("SellerSupport_Docs/chunk_embeddings.npz")
MODEL = "BAAI/bge-small-en-v1.5"

# Retrieval-sized, not filter-sized. Smaller than the 220 used for selection:
# the chunk is quoted verbatim, so a tighter passage makes a cleaner citation.
TARGET_WORDS = 150
MIN_WORDS = 60           # below this, merge back into the previous chunk
OVERLAP_SENTENCES = 1
ANSWER_SIM = 0.72        # a chunk "answers" a ticket at or above this

BSA_FILE = "1.1_BSA_PDF_English.txt"
BSA_TITLE = "Amazon Services Business Solutions Agreement"

# Section markers actually present in the extracted BSA. Verified 2026-09-09:
# there are no P- sections in this document, despite an earlier note citing P-4.
BSA_PART = {"S": "Selling on Amazon Service Terms",
            "F": "Fulfillment by Amazon Service Terms",
            "API": "API Terms"}
BSA_SECTION_RE = re.compile(r"^\s*((?:S|F|API)-\d+(?:\.\d+)?)\s+(.*)$")
BSA_GENERAL_RE = re.compile(r"^\s*(\d{1,2})\.\s+([A-Z].*)$")

SEP = "-" * 60
# Cross-promotion, not policy. Two chunks opened with "listen to this Seller
# Central Podcast episode" -- both in payments, already the thinnest category.
# As the FIRST sentence it drags the whole chunk's meaning off-topic.
PROMO_RE = re.compile(r"listen to this|podcast episode|watch this video|"
                      r"for an overview.{0,30}(?:video|podcast)", re.I)

# --- A3: thresholds ----------------------------------------------------------
# "We require sellers to maintain a LSR less than 4%" -> a structured rule, so
# the draft can quote the exact figure and rule-bearing chunks can be boosted.
METRIC_RE = (r"LSR|ODR|VTR|AHR|CR|late shipment rate|order defect rate|"
             r"valid tracking rate|cancellation rate|account health rating|"
             r"on-time delivery rate|pre-fulfil?lment cancel")
RULE_RE = re.compile(
    r"(?P<metric>" + METRIC_RE + r")?"
    r"[^.]{0,90}?"
    r"(?P<op>less than or equal to|greater than or equal to|less than|greater than|"
    r"at least|no more than|no less than|below|under|above|at or below)\s*"
    r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>%)", re.I)


def extract_rules(text: str):
    """Structured thresholds, but only where the pairing is UNAMBIGUOUS.

    Bullet lists extract without punctuation, so several metrics and several
    thresholds land in one "sentence" and a naive regex mis-pairs them -- the
    Business Seller badge list produced "order defect rate greater than 97.5%",
    which is nonsense. A wrong-but-confident figure quoted to a seller is worse
    than no figure at all, so a sentence carrying more than one threshold is
    skipped entirely rather than guessed at. The metric must also sit within
    MAX_GAP characters of its comparator.
    """
    MAX_GAP = 80
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        hits = list(RULE_RE.finditer(sent))
        if len(hits) != 1:
            continue                      # ambiguous list -- do not guess
        m = hits[0]
        metric = m.group("metric")
        if not metric:
            near = sent[max(0, m.start("op") - MAX_GAP):m.start("op")]
            back = list(re.finditer(METRIC_RE, near, re.I))
            metric = back[-1].group(0) if back else None
        if not metric:
            continue
        out.append({"metric": metric.upper() if len(metric) <= 4 else metric.lower(),
                    "operator": m.group("op").lower(),
                    "value": float(m.group("val")),
                    "unit": m.group("unit"),
                    "evidence": sent.strip()[:220]})
    ded, seen = [], set()
    for r in out:
        k = (r["metric"], r["operator"], r["value"])
        if k not in seen:
            seen.add(k); ded.append(r)
    return ded


ABBREV = re.compile(r"\b(?:No|Inc|Ltd|Co|Corp|St|Mr|Mrs|Ms|Dr|vs|etc|e\.g|i\.e|U\.S|U\.K)\.$",
                    re.I)


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def sentences(text: str) -> list[str]:
    """Split into sentences, not perfectly but safely.

    A wrong split costs a slightly odd chunk boundary; the previous word-count
    splitter cut mid-word, which is worse. Abbreviations are re-joined.
    """
    parts = re.split(r"(?<=[.!?])\s+", norm(text))
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if PROMO_RE.search(p):
            continue
        if out and ABBREV.search(out[-1]):
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return out


def pack_document(secs):
    """Pack a whole document into chunks of (heading_path, section_id, text).

    Packing runs ACROSS section boundaries, not within them. Packing per section
    produced 152 chunks under 60 words, because many help-page headings hold a
    single short paragraph and a lone short chunk has no predecessor to merge
    into. A new chunk therefore starts at a heading change only once the current
    one is already big enough to stand on its own; otherwise the short section
    flows into what follows. The chunk keeps the heading of its FIRST sentence,
    so the citation still points at where the quote begins.

    Very long "sentences" are also hard-split: bullet lists extract without
    terminal punctuation and arrive as one 400-word blob that sentence packing
    cannot divide.
    """
    units = []
    for head, sid, text in secs:
        for s in sentences(text):
            w = s.split()
            if len(w) > TARGET_WORDS:
                for i in range(0, len(w), TARGET_WORDS):
                    piece = " ".join(w[i:i + TARGET_WORDS])
                    if piece.strip():
                        units.append((head, sid, piece))
            else:
                units.append((head, sid, s))

    chunks = []
    buf, n, chead, csid = [], 0, None, None

    def flush(overlap_head=None):
        nonlocal buf, n, chead, csid
        if buf:
            chunks.append({"heading_path": chead, "section_id": csid,
                           "text": " ".join(buf)})
        keep = buf[-OVERLAP_SENTENCES:] if (OVERLAP_SENTENCES and
                                            overlap_head == chead) else []
        buf = list(keep)
        n = sum(len(x.split()) for x in buf)
        if not buf:
            chead, csid = None, None

    for head, sid, s in units:
        w = len(s.split())
        if buf and (n + w > TARGET_WORDS or (head != chead and n >= MIN_WORDS)):
            flush(overlap_head=head)
        if not buf:
            chead, csid = head, sid
        buf.append(s)
        n += w
    flush()

    # Anything still short merges backwards, but never past a sane ceiling.
    ceiling = int(TARGET_WORDS * 1.6)
    merged = []
    for c in chunks:
        if (merged and len(c["text"].split()) < MIN_WORDS
                and len(merged[-1]["text"].split()) + len(c["text"].split()) <= ceiling):
            merged[-1]["text"] += " " + c["text"]
        else:
            merged.append(c)
    return merged


def sections_from_headings(body: str):
    """(heading_path, section_id, text) for a help page, using its # markers."""
    stack: list[str] = []
    buf: list[str] = []
    out = []

    def flush():
        if buf:
            t = norm(" ".join(buf))
            if t:
                out.append((" > ".join(stack) if stack else "", None, t))
            buf.clear()

    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            flush()
            depth = len(m.group(1))
            del stack[depth - 1:]
            stack.append(norm(m.group(2)))
        else:
            buf.append(line)
    flush()
    return out


def sections_from_bsa(body: str):
    """(heading_path, section_id, text) for the agreement, using S-/F-/API-/§N."""
    out = []
    part = "General Terms"
    sec_id = None
    sec_title = ""
    buf: list[str] = []

    def flush():
        if buf and sec_id:
            t = norm(" ".join(buf))
            if t:
                head = "%s > %s > %s %s" % (BSA_TITLE, part, sec_id, sec_title)
                out.append((norm(head), sec_id, t))
        buf.clear()

    for line in body.splitlines():
        raw = line.rstrip()
        if not raw.strip():
            continue
        m = BSA_SECTION_RE.match(raw)
        if m:
            flush()
            sec_id = m.group(1)
            part = BSA_PART.get(sec_id.split("-")[0], "Service Terms")
            sec_title = norm(m.group(2)).split(".")[0][:70]
            buf.append(norm(m.group(2)))
            continue
        g = BSA_GENERAL_RE.match(raw)
        if g and len(raw) < 90:          # a heading line, not a numbered clause mid-paragraph
            flush()
            part = "General Terms"
            sec_id = "§" + g.group(1)
            sec_title = norm(g.group(2)).rstrip(".")[:70]
            continue
        buf.append(raw.strip())
    flush()
    return out


def load_queries():
    rows = []
    for line in QUERIES.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith(">"):
            continue
        p = line.split("\t")
        if len(p) >= 2 and p[1].strip():
            rows.append((p[0].strip(), p[1].strip()))
    return rows


def embed(texts, queries, refresh=False):
    fp = hashlib.sha256(
        (MODEL + "|" + "|".join(texts) + "|" + "|".join(queries)).encode("utf-8", "replace")
    ).hexdigest()[:32]
    if CACHE.exists() and not refresh:
        z = np.load(CACHE, allow_pickle=False)
        if z["fp"].item().decode() == fp:
            print("  using cached embeddings")
            return z["C"], z["Q"]
    from fastembed import TextEmbedding
    threads = max(1, min(6, (os.cpu_count() or 4) // 3))
    print("  loading %s (threads=%d) ..." % (MODEL, threads))
    m = TextEmbedding(model_name=MODEL, threads=threads)
    print("  embedding %d chunks ..." % len(texts))
    C = np.array(list(m.embed(texts, batch_size=32, parallel=1)), dtype=np.float32)
    print("  embedding %d queries ..." % len(queries))
    Q = np.array(list(m.query_embed(queries, batch_size=32, parallel=1)), dtype=np.float32)
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, C=C, Q=Q, fp=np.bytes_(fp.encode()))
    return C, Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()

    kb = json.load(open(KB, encoding="utf-8"))["documents"]
    queries = load_queries()
    qtext = [q for _, q in queries]
    print("Knowledge base: %d documents (frozen) | %d filter queries"
          % (len(kb), len(queries)))

    records = []
    for d in kb:
        raw = Path(d["local_copy"]).read_text(encoding="utf-8", errors="replace")
        body = raw.split(SEP, 1)[-1].strip() if SEP in raw else raw
        nbody = norm(body)
        is_bsa = d["file"] == BSA_FILE
        secs = sections_from_bsa(body) if is_bsa else sections_from_headings(body)
        if not secs:
            secs = [(d["title"], None, nbody)]

        cursor = 0
        for ch in pack_document(secs):
                piece = ch["text"]
                if len(piece.split()) < 20:
                    continue
                probe = piece[:60]
                start = nbody.find(probe, cursor)
                if start < 0:
                    start = nbody.find(probe)
                if start >= 0:
                    cursor = start + 1
                records.append({
                    "chunk_id": "%s#%04d" % (d["file"].replace(".txt", ""), len(records)),
                    "doc_id": d["file"].replace(".txt", ""),
                    "title": BSA_TITLE if is_bsa else d["title"],
                    "text": piece,
                    "heading_path": ch["heading_path"] or d["title"],
                    "section_id": ch["section_id"],
                    "source_url": d["source_url"],
                    "local_copy": d["local_copy"],
                    "category": d["category"],
                    "doc_type": "agreement" if is_bsa else "help_page",
                    "word_count": len(piece.split()),
                    "rules": extract_rules(piece),
                    "norm_char_start": start if start >= 0 else None,
                    "norm_char_end": (start + len(piece)) if start >= 0 else None,
                })

    print("  chunked: %d passages from %d documents" % (len(records), len(kb)))

    C, Q = embed([r["text"] for r in records], qtext, refresh=a.refresh)
    S = C @ Q.T
    for i, r in enumerate(records):
        row = S[i]
        hits = np.where(row >= ANSWER_SIM)[0]
        order = hits[np.argsort(-row[hits])][:6]
        r["answers_questions"] = [qtext[j] for j in order]
        r["answer_categories"] = sorted({queries[j][0] for j in order})
        b = int(np.argmax(row))
        r["best_question"] = qtext[b]
        r["best_similarity"] = round(float(row[b]), 4)
        r["embedding"] = [round(float(x), 6) for x in C[i]]

    with OUT_JSONL.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    wc = [r["word_count"] for r in records]
    answered = sum(1 for r in records if r["answers_questions"])
    covered = len({q for r in records for q in r["answers_questions"]})
    print("\nWrote %s  (%.1f MB)" % (OUT_JSONL, OUT_JSONL.stat().st_size / 1e6))
    print("  chunks              : %d" % len(records))
    print("  words per chunk     : min %d / median %d / max %d"
          % (min(wc), int(np.median(wc)), max(wc)))
    print("  chunks under %d words: %d" % (MIN_WORDS, sum(1 for w in wc if w < MIN_WORDS)))
    print("  chunks answering >=1 ticket: %d of %d" % (answered, len(records)))
    print("  tickets with >=1 answering chunk: %d of %d" % (covered, len(qtext)))
    print("  by doc_type         : %s" % dict(Counter(r["doc_type"] for r in records)))
    nrules = sum(1 for r in records if r["rules"])
    print("  chunks with a structured rule: %d" % nrules)
    print("  BSA sections found  : %d"
          % len({r["section_id"] for r in records if r["doc_type"] == "agreement"}))


if __name__ == "__main__":
    main()
