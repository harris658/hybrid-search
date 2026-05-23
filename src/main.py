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


def build_all(text: str):
    chunks = recursive_chunks(text)
    print("Embedding chunks (one-time)...")
    embeddings = embed_chunks(chunks)
    bm25 = BM25(chunks)
    print(f"Ready. {len(chunks)} chunks  alpha={ALPHA}\n")
    return chunks, embeddings, bm25


def query_all(query: str, chunks: list, embeddings: list, bm25: BM25) -> None:
    bm25_results = bm25.score(query)[:TOP_K]
    vec_results = vector_search(query, chunks, embeddings, top_k=TOP_K)
    rrf_results = reciprocal_rank_fusion([bm25_results, vec_results])
    weighted_results = weighted_hybrid(bm25_results, vec_results, alpha=ALPHA)

    print(f"\n{'=' * 60}")
    print(f"Query: '{query}'")
    print("=" * 60)

    for label, results in [
        ("BM25    ", bm25_results),
        ("VECTOR  ", vec_results),
        ("RRF     ", rrf_results),
        ("WEIGHTED", weighted_results),
    ]:
        chunk, score = results[0]
        print(f"\n{label}  score={score:.4f}")
        print(chunk)


def main() -> None:
    text = load_corpus(CORPUS_PATH)
    chunks, embeddings, bm25 = build_all(text)

    print("Type a query (or 'quit' to exit).\n")
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() == "quit":
            break
        query_all(query, chunks, embeddings, bm25)
        print()


if __name__ == "__main__":
    main()
