def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    cid = 1

    while start < len(words):
        end = start + chunk_size
        chunks.append({
            "id": cid,
            "text": " ".join(words[start:end])
        })
        start = end - overlap
        cid += 1

    return chunks
