# Notes

Running log. One entry per working session. Newest at the top.

---

## 2026-05-23 — Steps 1–6: Full session

### Step 1 — BM25 from scratch
Built TF, IDF, and length normalization from scratch using only Python's `math` and `collections`.
- "what is khadi": BM25=4.05, correct chunk, massive score gap vs. everything else (next chunk: 0.70).
- "Gandhi cloth": BM25=4.27, correct — both "Gandhi" and "cloth" appear verbatim.
- "breathable artisanal fabric": BM25=5.66, correct — exact words in corpus.
- Hard zero behaviour confirmed: "Gandhi cloth" → Results 2 & 3 both 0.0000. No word overlap = no score at all.

### Step 2 — Vocabulary mismatch demo
Ran BM25 and vector side by side on queries designed to break each one.
- "Gandhian textile": BM25=0.0000 (zero — "Gandhian" not in corpus). Vector=0.6855, returned Banarasi silk (related but not khadi).
- "cloth from freedom movement": BM25 latched onto "cloth"+"from" → wrong chunk. Vector scored 0.5847 → wrong chunk. Both failed.
- "breathable artisanal fabric": both correct — exact words exist.
- Key insight: vector has no hard zeros — even unrelated chunks get a small non-zero score. BM25 is binary on missing words.

### Step 3 — Hybrid with RRF (Reciprocal Rank Fusion)
Combined BM25 and vector ranked lists using 1/(k+rank), k=60.
- "what is khadi": all three agree. ✅
- "Gandhian textile": BM25 all-zeros → arbitrary ranks → RRF promoted Bandhgala (noise injection). ❌
- "cloth from freedom movement": RRF surfaced khadi — neither BM25 nor vector found it alone. ✅ Real fusion win.
- RRF weakness: when one list carries no signal (all zeros), arbitrary ranks inject noise into the fusion result.

### Step 4 — Weighted hybrid (α × vector + (1−α) × BM25)
Normalized both score lists to [0,1] before combining with alpha=0.5.
- "Gandhian textile": BM25 all-zeros → normalizes to 0.0 everywhere → drops out cleanly. Weighted follows vector. No noise injection.
- "cloth from freedom movement": weighted=0.8767, khadi — correct. Confirmed the RRF win with a cleaner mechanism.
- Weighted scores 1.0000 when both methods agree on the same top chunk.
- Advantage over RRF: when BM25 has no signal, it contributes nothing instead of injecting noise.

### Step 5 — Interactive benchmark

| Query | BM25 | Vector | RRF | Weighted |
|-------|------|--------|-----|----------|
| sherwani for wedding | ❌ Banarasi | ✅ | ✅ | ✅ |
| Mughal era garment | ✅ | ✅ | ✅ | ✅ |
| fabric for summer | ❌ Banarasi | ✅ | ❌ | ❌ |
| what is bandhgala | ❌ | ❌ | ❌ | ❌ |

"fabric for summer": BM25 was confidently wrong (Banarasi has "fabric" prominently). RRF and weighted both followed BM25's signal and overrode the correct vector result. Hybrid hurt here.

"what is bandhgala": all four failed. Bandhgala definition is split or merged awkwardly by the recursive chunker. No retrieval strategy can fix a chunking problem.

### Step 6 — Failure analysis

**BM25**
- Hard zeros on vocabulary mismatch — if query words don't appear verbatim, score is 0.0.
- Latches onto high-frequency words ("fabric", "for") that appear in wrong chunks and scores them high.
- No semantic understanding — "Gandhian" and "Gandhi" are different tokens.

**Vector search**
- No hard zeros, but weak signal on unusual paraphrases ("Gandhian textile" → 0.6855 on wrong chunk).
- Single embedding averages chunk meaning — long chunks pull the vector away from the best sentence.
- Sensitive to chunk boundaries — if the right sentence is split, no embedding captures it cleanly.

**RRF**
- Noise injection: when one list has no signal (all zeros), arbitrary rank order poisons the fusion.
- Works well when both lists have real signal pointing at different chunks — fusion rescues weak agreement.
- Rank-based: doesn't know whether rank 1 had score 4.0 or 0.001 — treats them identically.

**Weighted hybrid**
- Better than RRF on zero-signal: normalization collapses all-zero BM25 to 0.0 contribution.
- Still fails when BM25 is confidently wrong — high BM25 score on wrong chunk overrides correct vector result ("fabric for summer").
- Alpha=0.5 is a guess. Optimal alpha depends on the query type — no single value works for all queries.

**Failures none of these fix:**
1. **Vocabulary mismatch at query time** — "Gandhian textile" fails everywhere because the query language doesn't match the corpus language. Fix: query expansion/rewriting — rephrase the query before retrieval.
2. **Chunking boundary failures** — "what is bandhgala" fails because the chunker split the Bandhgala definition. Fix: better chunking (sentence-window or semantic) before indexing.
3. **BM25 high-frequency false positives** — "fabric for summer" fails because BM25 scores wrong chunks high on common words. Fix: better stopword filtering, or lower BM25 alpha weight.
4. **No answer quality signal** — all strategies return the highest-scoring chunk regardless of whether it actually answers the question. Fix: re-ranking with a cross-encoder that reads query + chunk together.

**What comes next:**
- Query expansion: rewrite the query into multiple variants before retrieval, merge results.
- Re-ranking: retrieve top-20, re-score each with a cross-encoder, return true top-3.
