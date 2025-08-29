import warnings
from typing import Any


def chunk_examples(
    examples: dict[str, list[Any]],
    column: str | None = None,
    split_char: str | None = None,
    split_len: int | None = None,
) -> dict[str, list[Any]]:
    """Chunk an input example text into paragraphs.

    Args:
        examples (dict[str, list[Any]]): Input batch.
        column (str, optional): Text column name to chunk. Defaults to 'text'.
        split_char (str, optional): Character to split examples at. Only one of `split_char` and `split_len`
            should be specified. Defaults to None.
        split_len (int, optional): Number of characters to split examples at. Only one of `split_char` and
            `split_len` should be specified. Defaults to None.

    Returns:
        dict[str, list[Any]]: Chunked input batch, reduced to `column` only.
    """
    if split_char is not None and split_len is not None:
        warnings.warn("Both `split_char` and `split_len` are given, falling back to use `split_len`.", stacklevel=2)
    if split_char is None and split_len is None:
        raise ValueError("Either `split_char` or `split_len` are required.")

    column = column if column is not None else "text"

    chunks = []
    if split_len is not None:
        for sentence in examples[column]:
            chunks += [sentence[i : i + split_len] for i in range(0, len(sentence), split_len)]
    else:
        for sentence in examples[column or "text"]:
            chunks += sentence.split(split_char)
    return {column: chunks}


__all__ = ["chunk_examples"]
