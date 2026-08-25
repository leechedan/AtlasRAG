from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def _enforce_chunk_size_cap(
    chunks: list[str], chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Safety net guaranteeing len(chunk) <= chunk_size for every returned
    chunk, regardless of which internal path (the custom-separator
    recursive branch, or the default RecursiveCharacterTextSplitter
    branch) produced it (Important Finding I2). ``separators=[""]`` means
    "split at any character boundary", so this is guaranteed to be able to
    split any oversized chunk further -- unlike a custom separator list,
    which may not appear in the text at all.
    """
    if all(len(c) <= chunk_size for c in chunks):
        return chunks

    hard_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[""],
    )
    result: list[str] = []
    for c in chunks:
        if len(c) <= chunk_size:
            result.append(c)
        else:
            result.extend(hard_splitter.split_text(c))
    return result


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: list[str] | None = None,
) -> list[str]:
    if not text or not text.strip():
        return []

    chunk_size_val = chunk_size if chunk_size is not None else settings.chunk_size
    chunk_overlap_val = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    separators_val = separators if separators is not None else settings.chunk_separators

    # If custom separators are provided (not the default), try splitting on them first
    if separators is not None and separators != settings.chunk_separators:
        # Try to split on the first separator (not the empty string)
        for sep in separators:
            if sep and sep in text:
                # Split on this separator and recursively chunk each part
                parts = text.split(sep)
                result = []
                for part in parts:
                    if part and part.strip():
                        # Recursively apply RecursiveCharacterTextSplitter to each part
                        sub_chunks = chunk_text(
                            part,
                            chunk_size=chunk_size_val,
                            chunk_overlap=chunk_overlap_val,
                            separators=[s for s in separators if s != sep]
                        )
                        result.extend(sub_chunks)
                return _enforce_chunk_size_cap(result, chunk_size_val, chunk_overlap_val)

    # Use RecursiveCharacterTextSplitter for default behavior
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_val,
        chunk_overlap=chunk_overlap_val,
        separators=separators_val,
    )
    return _enforce_chunk_size_cap(
        splitter.split_text(text), chunk_size_val, chunk_overlap_val
    )


def chunk_csv_rows(
    rows: list[dict], header: list[str], rows_per_chunk: int = 20, max_chars: int | None = None
) -> list[str]:
    max_chars = max_chars or settings.chunk_size
    header_line = f"columns: {', '.join(header)}"
    chunks: list[str] = []
    current_lines: list[str] = []

    def flush() -> None:
        if current_lines:
            chunks.append(header_line + "\n" + "\n".join(current_lines))

    for row in rows:
        line = ", ".join(f"{k}={row.get(k, '')}" for k in header)
        single_row_len = len(header_line) + 1 + len(line)

        if single_row_len > max_chars:
            flush()
            current_lines = []
            piece_budget = max(max_chars - len(header_line) - 1, 1)
            for piece in chunk_text(line, chunk_size=piece_budget, chunk_overlap=0, separators=[""]):
                chunks.append(header_line + "\n" + piece)
            continue

        candidate_len = len(header_line) + 1 + sum(len(l) + 1 for l in current_lines) + len(line) + 1

        if current_lines and (len(current_lines) >= rows_per_chunk or candidate_len > max_chars):
            flush()
            current_lines = []

        current_lines.append(line)

    flush()
    return chunks
