# hybrid-search

> Combine BM25 (keyword) and vector (semantic) retrieval to fix the vocabulary mismatch problem that pure vector search cannot solve.

## Showcase

- **What this proves:** I can implement BM25 from scratch and fuse keyword + vector retrieval two different ways.
- **Headline result:** BM25 + vector via RRF and a weighted blend; weighted (α=0.65) was the most consistent retriever across 8 vocab-mismatch / semantic queries, and exposed exactly where each signal fails alone.
- **Demo:** `python src/main.py`

## Concepts practiced

- [BM25 & hybrid search](../../concepts/bm25-and-hybrid-search.md)
- [Embeddings & cosine similarity](../../concepts/embeddings-and-cosine-similarity.md)

## Source

- **Wiki page:** `concepts/retrieval-spectrum` — Level 1 (keyword) + Level 2 (semantic) + hybrid variant
- **Why this concept:** Vector search finds *similar* content, not *correct* content. It fails when the user's words don't match the document's words. BM25 catches exact keyword matches that vector misses. Hybrid combines both signals.

## Goal

Show the vocabulary mismatch problem concretely, then fix it:
- Pure vector: query `"Gandhi cloth"` → should retrieve khadi, but vector may not connect "Gandhi cloth" to "khadi"
- BM25 alone: query `"breathable artisanal fabric"` → misses because none of those words appear verbatim
- Hybrid: catches both cases by combining scores

## Approach

- [x] Step 1 — BM25 from scratch: implement term frequency, IDF, and BM25 scoring
- [x] Step 2 — Vocabulary mismatch demo: show where pure vector fails, where BM25 fails
- [x] Step 3 — Hybrid (RRF): combine rankings using Reciprocal Rank Fusion
- [x] Step 4 — Hybrid (weighted): combine scores using α·vector + (1−α)·BM25
- [x] Step 5 — Benchmark: vocabulary mismatch queries vs semantic queries across all three
- [x] Step 6 — Failure analysis: what does hybrid still miss?

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python src/main.py
```

## Results

| Query | BM25 | Vector | RRF | Weighted |
|---|---|---|---|---|
| what is khadi | ✅ | ✅ | ✅ | ✅ |
| Gandhian textile | ❌ zero | ❌ Banarasi | ❌ Banarasi | ❌ Banarasi |
| cloth from freedom movement | ❌ | ❌ | ✅ | ✅ |
| breathable artisanal fabric | ✅ | ✅ | ✅ | ✅ |
| sherwani for wedding | ❌ | ✅ | ✅ | ✅ |
| Mughal era garment | ✅ | ✅ | ✅ | ✅ |
| fabric for summer | ❌ | ✅ | ❌ | ❌ |
| what is bandhgala | ❌ | ❌ | ❌ | ❌ |

Weighted hybrid is the most consistent. Remaining failures require query expansion (vocabulary mismatch) or re-ranking (BM25 false positives).

See `notes.md` for the full session log and failure analysis.
