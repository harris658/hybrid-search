from dotenv import load_dotenv
from corpus import load_corpus, recursive_chunks
from bm25 import BM25
from embedder import embed_chunks, vector_search
from rrf import reciprocal_rank_fusion
from weighted import weighted_hybrid

load_dotenv()

CORPUS_PATH = "data/sample.txt"
TOP_K = 14
ALPHA = 0.5

QUERIES = [
    ("exact match",           "what is khadi"),
    ("synonym — BM25 gap",    "Gandhian textile"),
    ("paraphrase — both gap", "cloth from freedom movement"),
    ("exact words",           "breathable artisanal fabric"),
]


def run_comparison(query: str, label: str, bm25: BM25, chunks: list, embeddings: list) -> None:
    print(f"\n{'=' * 60}")
    print(f"[{label}]  Query: '{query}'")
    print("=" * 60)

    bm25_results = bm25.score(query)[:TOP_K]
    vec_results = vector_search(query, chunks, embeddings, top_k=TOP_K)
    rrf_results = reciprocal_rank_fusion([bm25_results, vec_results])
    weighted_results = weighted_hybrid(bm25_results, vec_results, alpha=ALPHA)

    print(f"\nBM25      score={bm25_results[0][1]:.4f}  →  {bm25_results[0][0][:80]}")
    print(f"VECTOR    score={vec_results[0][1]:.4f}  →  {vec_results[0][0][:80]}")
    print(f"RRF       score={rrf_results[0][1]:.4f}  →  {rrf_results[0][0][:80]}")
    print(f"WEIGHTED  score={weighted_results[0][1]:.4f}  →  {weighted_results[0][0][:80]}")


def main() -> None:
    text = load_corpus(CORPUS_PATH)
    chunks = recursive_chunks(text)

    print("Embedding chunks (one-time)...")
    embeddings = embed_chunks(chunks)
    bm25 = BM25(chunks)
    print(f"Ready. {len(chunks)} chunks  alpha={ALPHA}\n")

    for label, query in QUERIES:
        run_comparison(query, label, bm25, chunks, embeddings)


if __name__ == "__main__":
    main()
