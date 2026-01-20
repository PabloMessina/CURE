from nltk.tokenize import sent_tokenize
from typing import List


def split_text_into_chunks(text: str, max_length: int) -> List[str]:
    """
    Splits the input text into chunks, each not exceeding the specified maximum length,
    by grouping sentences together. Sentence splitting is performed first, and sentences
    are accumulated into a chunk as long as adding the next sentence does not exceed max_length.

    Args:
        text (str): The input text to split.
        max_length (int): The maximum length (in characters) of each chunk.

    Returns:
        list[str]: A list of text chunks, each of length <= max_length.
    """
    sentences = sent_tokenize(text)
    chunks: list[str] = []
    chunk = ''
    for sentence in sentences:
        if len(chunk) + len(sentence) > max_length and chunk != '':
            chunks.append(chunk)
            chunk = ''
        if chunk != '':
            if chunk[-1] != '.':
                chunk += '. '
            else:
                chunk += ' '
        chunk += sentence
    if chunk != '':
        chunks.append(chunk)
    return chunks