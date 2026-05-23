# hybrid-search

> Combine BM25 (keyword) and vector (semantic) retrieval to fix the vocabulary mismatch problem that pure vector search cannot solve.

## Source

- **Wiki page:** `concepts/retrieval-spectrum` — Level 1 (keyword) + Level 2 (semantic) + hybrid variant
- **Why this concept:** Vector search finds *similar* content, not *correct* content. It fails when the user's words don't match the document's words. BM25 catches exact keyword matches that vector misses. Hybrid combines both signals.

## Goal

Show the vocabulary mismatch problem concretely, then fix it:
- Pure vector: query `"Gandhi cloth"` → should retrieve khadi, but vector may not connect "Gandhi cloth" to "khadi"
- BM25 alone: query `"breathable artisanal fabric"` → misses because none of those words appear verbatim
- Hybrid: catches both cases by combining scores

## Approach

- [ ] Step 1 — BM25 from scratch: implement term frequency, IDF, and BM25 scoring
- [ ] Step 2 — Vocabulary mismatch demo: show where pure vector fails, where BM25 fails
- [ ] Step 3 — Hybrid (RRF): combine rankings using Reciprocal Rank Fusion
- [ ] Step 4 — Hybrid (weighted): combine scores using α·vector + (1−α)·BM25
- [ ] Step 5 — Benchmark: vocabulary mismatch queries vs semantic queries across all three
- [ ] Step 6 — Failure analysis: what does hybrid still miss?

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

See `notes.md` for the running log.
