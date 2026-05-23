import numpy as np
import ollama

MODEL = "nomic-embed-text"


def embed_chunks(chunks: list[str]) -> list[np.ndarray]:
    return [
        np.array(ollama.embeddings(model=MODEL, prompt=c).embedding)
        for c in chunks
    ]


def embed_query(query: str) -> np.ndarray:
    return np.array(ollama.embeddings(model=MODEL, prompt=query).embedding)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def vector_search(
    query: str,
    chunks: list[str],
    embeddings: list[np.ndarray],
    top_k: int = 3,
) -> list[tuple[str, float]]:
    q_vec = embed_query(query)
    scored = [(c, cosine_similarity(q_vec, e)) for c, e in zip(chunks, embeddings)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
