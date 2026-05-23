import math
from collections import Counter


def tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25:
    def __init__(self, chunks: list[str], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokenized = [tokenize(c) for c in chunks]
        self.avgdl = sum(len(t) for t in self.tokenized) / len(self.tokenized)
        self.df = self._build_df()
        self.n = len(chunks)

    def _build_df(self) -> dict[str, int]:
        df: dict[str, int] = {}
        for tokens in self.tokenized:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        return df

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str) -> list[tuple[str, float]]:
        terms = tokenize(query)
        scores = []
        for i, tokens in enumerate(self.tokenized):
            tf_map = Counter(tokens)
            dl = len(tokens)
            s = 0.0
            for term in terms:
                tf = tf_map.get(term, 0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += self.idf(term) * numerator / denominator
            scores.append((self.chunks[i], s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
