from dotenv import load_dotenv
from corpus import load_corpus, recursive_chunks
from bm25 import BM25
from embedder import embed_chunks, vector_search
from rrf import reciprocal_rank_fusion

load_dotenv()

CORPUS_PATH = "data/sample.txt"
TOP_K = 5

QUERIES = [
    ("exact match",           "what is khadi"),
    ("synonym — BM25 gap",    "Gandhian textile"),
    ("paraphrase — both gap", "cloth from freedom movement"),
    ("exact words",           "breathable artisanal fabric"),
]


def run_comparison(
    query: str,
    label: str,
    bm25: BM25,
    chunks: list,
    embeddings: list,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"[{label}]  Query: '{query}'")
    print("=" * 60)

    bm25_results = bm25.score(query)[:TOP_K]
    vec_results = vector_search(query, chunks, embeddings, top_k=TOP_K)
    rrf_results = reciprocal_rank_fusion([bm25_results, vec_results])

    bm25_chunk, bm25_score = bm25_results[0]
    vec_chunk, vec_score = vec_results[0]
    rrf_chunk, rrf_score = rrf_results[0]

    print(f"\nBM25    score={bm25_score:.4f}")
    print(bm25_chunk)

    print(f"\nVECTOR  score={vec_score:.4f}")
    print(vec_chunk)

    print(f"\nRRF     score={rrf_score:.4f}")
    print(rrf_chunk)

    if rrf_chunk == bm25_chunk == vec_chunk:
        print("\n  ✓  All three agree")
    elif rrf_chunk != bm25_chunk and rrf_chunk != vec_chunk:
        print("\n  ↑  RRF surfaced a different top result")
    elif rrf_chunk == vec_chunk and bm25_score == 0.0:
        print("\n  ↑  RRF rescued vector result (BM25 was zero)")


def main() -> None:
    text = load_corpus(CORPUS_PATH)
    chunks = recursive_chunks(text)

    print("Embedding chunks (one-time)...")
    embeddings = embed_chunks(chunks)
    bm25 = BM25(chunks)
    print(f"Ready. {len(chunks)} chunks.\n")

    for label, query in QUERIES:
        run_comparison(query, label, bm25, chunks, embeddings)


if __name__ == "__main__":
    main()
