def load_corpus(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def recursive_chunks(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    separators = ["\n\n", "\n", ". ", " "]

    def _split(text: str, separators: list[str]) -> list[str]:
        if not separators or len(text) <= size:
            return [text]
        sep = separators[0]
        parts = text.split(sep)
        chunks, current = [], ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > size:
                    chunks.extend(_split(part, separators[1:]))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)
        return chunks

    raw = _split(text, separators)
    result = []
    for i, chunk in enumerate(raw):
        if i == 0 or not overlap:
            result.append(chunk)
        else:
            result.append(raw[i])
    return result
