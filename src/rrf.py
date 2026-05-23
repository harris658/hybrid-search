def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}

    for ranked in ranked_lists:
        for rank, (chunk, _) in enumerate(ranked, start=1):
            scores[chunk] = scores.get(chunk, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused
