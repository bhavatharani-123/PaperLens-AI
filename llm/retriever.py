from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CREATE TF-IDF INDEX
# ============================================================

def create_retriever(chunks):

    if not chunks:
        raise ValueError(
            "No chunks were provided to the retriever."
        )

    texts = []

    for chunk in chunks:

        text = chunk.get("text", "")

        if text and text.strip():
            texts.append(text)

        else:
            texts.append("")


    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=30000,
        sublinear_tf=True
    )


    vectors = vectorizer.fit_transform(texts)


    return vectorizer, vectors


# ============================================================
# SEARCH RESEARCH PAPERS
# ============================================================

def search(
    query,
    chunks,
    vectorizer,
    vectors,
    top_k=5,
    min_score=0.03
):

    if not query or not query.strip():
        return []


    # Convert question into TF-IDF vector

    query_vector = vectorizer.transform(
        [query]
    )


    # TF-IDF vectors are L2 normalized by default.
    # Dot product therefore represents cosine similarity.

    scores = (
        vectors @ query_vector.T
    ).toarray().flatten()


    # Highest score first

    ranked_indices = scores.argsort()[::-1]


    results = []


    for index in ranked_indices:

        score = float(
            scores[index]
        )


        if score < min_score:
            continue


        result = chunks[index].copy()

        result["score"] = score

        results.append(result)


        if len(results) >= top_k:
            break


    return results