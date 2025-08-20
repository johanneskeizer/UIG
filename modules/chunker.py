def chunk_text(text, chunk_size=400, chunk_overlap=50):
    import re
    words = re.findall(r'\w+|\W+', text)
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = "".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks
