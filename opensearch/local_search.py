"""Retrieval preview with no AWS and no OpenSearch. Retrieval only — no drafting.

    python opensearch/local_search.py                       # interactive
    python opensearch/local_search.py "my payment is on hold"
    python opensearch/local_search.py --hybrid "my LSR is bad"
    python opensearch/local_search.py --full -k 3 "reserve on my account"

WHAT IS EXACT AND WHAT IS NOT

  The vector half is EXACT. It scores the ticket against the same 384-dim
  vectors that go into the index, with the same model and the same query
  prefix, so the cosines printed here are the cosines OpenSearch will report
  and the ones MIN_SIM was calibrated against.

  --hybrid is an APPROXIMATION. It re-implements BM25 with the same field
  weights and the same synonym list, which is close enough to preview which
  tickets BM25 rescues — but Lucene's scoring, stemming and analyzer chain are
  not reproduced here. Use it to eyeball behaviour, never to tune weights.
  `evaluate.py` against the live index is the real measurement.
"""

import argparse
import json
import math
import re
import sys
from collections import Counter

import numpy as np

import config

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def load_chunks() -> list[dict]:
    return [
        json.loads(line)
        for line in config.CHUNKS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_matrix(chunks: list[dict]) -> np.ndarray:
    m = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


# ---------------------------------------------------------------------------
# Approximate BM25, mirroring the production field weights
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")

# Same fields and weights as config.LEXICAL_FIELDS.
FIELDS = [(spec.split("^")[0], float(spec.split("^")[1])) for spec in config.LEXICAL_FIELDS]


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def field_text(chunk: dict, field: str) -> str:
    value = chunk.get(field)
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value or "")


def load_synonym_map() -> dict[str, list[str]]:
    """Search-time synonym expansion, mirroring the synonym_graph filter."""
    mapping: dict[str, list[str]] = {}
    if not config.SYNONYMS_FILE.exists():
        return mapping
    for line in config.SYNONYMS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        terms = [t.strip() for t in line.split(",") if t.strip()]
        tokens = {tok for term in terms for tok in tokenize(term)}
        for term in terms:
            for tok in tokenize(term):
                mapping.setdefault(tok, [])
                mapping[tok] = sorted(set(mapping[tok]) | (tokens - {tok}))
    return mapping


class BM25Field:
    """One field's inverted statistics. k1=1.2, b=0.75 — Lucene's defaults."""

    def __init__(self, docs: list[list[str]]):
        self.n = len(docs)
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = float(self.len.mean()) if self.n else 0.0
        df = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {
            t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()
        }

    def score(self, terms: list[str], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        if self.avg == 0:
            return out
        for term in terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, counts in enumerate(self.tf):
                f = counts.get(term)
                if f:
                    denom = f + k1 * (1 - b + b * self.len[i] / self.avg)
                    out[i] += idf * (f * (k1 + 1)) / denom
        return out


def build_bm25(chunks: list[dict]) -> dict[str, BM25Field]:
    return {
        name: BM25Field([tokenize(field_text(c, name)) for c in chunks])
        for name, _ in FIELDS
    }


def bm25_scores(index: dict[str, BM25Field], terms: list[str]) -> np.ndarray:
    """best_fields with tie_breaker 0.3 — max field, plus a fraction of the rest."""
    per_field = []
    for name, weight in FIELDS:
        per_field.append(index[name].score(terms) * weight)
    stacked = np.vstack(per_field)
    best = stacked.max(axis=0)
    rest = stacked.sum(axis=0) - best
    return best + 0.3 * rest


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def retrieve(ticket, chunks, matrix, embed, k=5, hybrid=False, bm25=None, syn=None):
    qvec = np.array(embed(ticket), dtype=np.float32)
    qvec /= np.linalg.norm(qvec) + 1e-9
    cos = matrix @ qvec

    if not hybrid:
        order = np.argsort(-cos)[:k]
        return [(int(i), float(cos[i]), None, None) for i in order], None

    terms = tokenize(ticket)
    expanded = list(terms)
    for t in terms:
        expanded.extend(syn.get(t, []))
    lex = bm25_scores(bm25, expanded)

    combined = config.WEIGHT_LEXICAL * minmax(lex) + config.WEIGHT_VECTOR * minmax(cos)
    order = np.argsort(-combined)[:k]
    return (
        [(int(i), float(cos[i]), float(lex[i]), float(combined[i])) for i in order],
        sorted(set(expanded) - set(terms)),
    )


def show(rows, chunks, ticket, full=False, hybrid=False, added=None):
    best_cos = max(r[1] for r in rows) if rows else 0.0
    verdict = "CONFIDENT" if best_cos >= config.MIN_SIM else "RE-QUERY"

    print("\nticket     : %s" % ticket)
    print("best cosine: %.4f  vs MIN_SIM %.2f  ->  %s" % (best_cos, config.MIN_SIM, verdict))
    if hybrid and added:
        print("synonyms   : + %s" % ", ".join(added[:12]))
    print()

    for rank, (i, cos, lex, comb) in enumerate(rows, 1):
        c = chunks[i]
        head = "%d. cos %.4f" % (rank, cos)
        if comb is not None:
            head += "   bm25 %6.2f   hybrid %.4f" % (lex, comb)
        print(head)
        print("   %s  [%s / %s]" % (c.get("title"), c.get("category"), c.get("doc_type")))
        path = c.get("heading_path") or ""
        if c.get("section_id"):
            path += ", %s" % c["section_id"]
        print("   %s" % path)
        url = c.get("source_url") or "(no source_url — backfilled at upload for BSA chunks)"
        print("   %s" % url)
        print("   chunk_id: %s" % c["chunk_id"])
        if c.get("rules"):
            for r in c["rules"]:
                print("   RULE: %s %s %s%s" % (r.get("metric"), r.get("operator"),
                                               r.get("value"), r.get("unit", "")))
        body = " ".join(c.get("text", "").split())
        print("   %s" % (body if full else body[:260] + ("..." if len(body) > 260 else "")))
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Local retrieval preview (no AWS)")
    ap.add_argument("ticket", nargs="*")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--hybrid", action="store_true", help="approximate BM25 + synonyms too")
    ap.add_argument("--full", action="store_true", help="print the whole chunk")
    args = ap.parse_args()

    print(config.summary())
    chunks = load_chunks()
    matrix = load_matrix(chunks)
    print("loaded %d chunks, %d dims" % (len(chunks), matrix.shape[1]))

    from search import embed_query

    bm25 = syn = None
    if args.hybrid:
        print("building approximate BM25 index ...", flush=True)
        bm25, syn = build_bm25(chunks), load_synonym_map()
        print("  %d synonym tokens" % len(syn))

    def run(text: str) -> None:
        rows, added = retrieve(text, chunks, matrix, embed_query, args.k,
                               args.hybrid, bm25, syn)
        show(rows, chunks, text, args.full, args.hybrid, added)

    if args.ticket:
        run(" ".join(args.ticket))
        return

    print("\ninteractive — type a ticket, blank line or Ctrl-C to quit")
    if not args.hybrid:
        print("(vector only. add --hybrid to preview the BM25 half too)")
    while True:
        try:
            text = input("\nticket> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            return
        run(text)


if __name__ == "__main__":
    main()
