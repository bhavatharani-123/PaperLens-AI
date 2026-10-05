def chunk_pages(pages, chunk_size=1200, overlap=200):

    chunks = []

    for page in pages:

        text = page["text"].strip()

        if not text:
            continue

        paper = page["paper"]
        page_number = page["page"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            # Try to finish at a sentence
            if end < len(text):

                last_period = chunk_text.rfind(". ")

                if last_period > chunk_size * 0.6:

                    chunk_text = (
                        chunk_text[:last_period + 1]
                    )

                    end = start + len(chunk_text)

            if chunk_text.strip():

                chunks.append({
                    "text": chunk_text.strip(),
                    "paper": paper,
                    "page": page_number
                })

            start = max(
                end - overlap,
                start + 1
            )

    return chunks