def normalize(scores: list[tuple[str, float]]) -> list[tuple[str, float]]:
    values = [s for _, s in scores]
    min_s, max_s = min(values), max(values)
    if max_s == min_s:
        return [(chunk, 0.0) for chunk, _ in scores]
    return [(chunk, (s - min_s) / (max_s - min_s)) for chunk, s in scores]


def weighted_hybrid(
    bm25_results: list[tuple[str, float]],
    vec_results: list[tuple[str, float]],
    alpha: float = 0.5,
) -> list[tuple[str, float]]:
    bm25_norm = dict(normalize(bm25_results))
    vec_norm = dict(normalize(vec_results))

    all_chunks = set(bm25_norm) | set(vec_norm)
    combined = {}
    for chunk in all_chunks:
        b = bm25_norm.get(chunk, 0.0)
        v = vec_norm.get(chunk, 0.0)
        combined[chunk] = alpha * v + (1 - alpha) * b

    return sorted(combined.items(), key=lambda x: x[1], reverse=True)
