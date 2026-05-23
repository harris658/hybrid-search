from dotenv import load_dotenv
from corpus import load_corpus, recursive_chunks
from bm25 import BM25

load_dotenv()

CORPUS_PATH = "data/sample.txt"
TOP_K = 3
QUERIES = [
    "what is khadi",
    "Gandhi cloth",
    "breathable artisanal fabric",
]


def main() -> None:
    text = load_corpus(CORPUS_PATH)
    chunks = recursive_chunks(text)
    bm25 = BM25(chunks)

    print(f"Corpus: {len(chunks)} chunks  avgdl={bm25.avgdl:.1f} tokens\n")

    for query in QUERIES:
        print(f"{'=' * 60}")
        print(f"Query: '{query}'")
        print("=" * 60)
        results = bm25.score(query)[:TOP_K]
        for rank, (chunk, score) in enumerate(results, 1):
            print(f"\nResult {rank}  bm25={score:.4f}")
            print(chunk)
        print()


if __name__ == "__main__":
    main()
