# Notes

Running log. One entry per working session. Newest at the top.

---

## 2026-05-23 — Scaffold

**What I tried:**
- Grounded project in `concepts/retrieval-spectrum` from the Zeno wiki — hybrid search is the natural next step after chunking: Level 1 (keyword) + Level 2 (semantic) combined.
- Scoped to BM25 from scratch, then two hybrid combination methods: Reciprocal Rank Fusion (RRF) and weighted scoring.
- Same corpus as naive-rag and advanced-chunking for continuity.

**Motivation:**
- advanced-chunking exposed the vocabulary mismatch problem: all four strategies use dense embeddings only. If the query uses different words than the document (synonyms, paraphrases), similarity drops even when the meaning matches.
- BM25 catches exact keyword hits that vector search misses. Hybrid combines both signals.

**Next:**
- Step 1: Implement BM25 from scratch — TF, IDF, length normalization. Score all chunks against a query and print ranked results.
