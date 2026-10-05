from pdf_processor import load_all_papers
from chunker import chunk_pages
from retriever import create_retriever, search
from llm import generate_answer


print("=" * 60)
print("PAPERLENS AI - RAG TEST")
print("=" * 60)


# --------------------------------
# 1. LOAD PAPERS
# --------------------------------

pages = load_all_papers()

print(f"\nTotal pages: {len(pages)}")


# --------------------------------
# 2. CREATE CHUNKS
# --------------------------------

chunks = chunk_pages(pages)

print(f"Total chunks: {len(chunks)}")


# --------------------------------
# 3. CREATE RETRIEVER
# --------------------------------

vectorizer, vectors = create_retriever(chunks)

print("\nRetriever created successfully")


# --------------------------------
# 4. USER QUESTION
# --------------------------------

question = (
    "What is the main idea of "
    "the Transformer architecture?"
)


print("\nQuestion:")
print(question)


# --------------------------------
# 5. RETRIEVE
# --------------------------------

results = search(
    question,
    chunks,
    vectorizer,
    vectors,
    top_k=5
)


print("\nRetrieved Results:")
print("-" * 60)


for i, result in enumerate(results, 1):

    print(
        f"\n{i}. Paper: {result['paper']}"
    )

    print(
        f"   Page: {result['page']}"
    )

    print(
        f"   Score: {result['score']:.4f}"
    )


# --------------------------------
# 6. GENERATE ANSWER
# --------------------------------

print("\nGenerating answer...")


answer = generate_answer(
    question,
    results
)


# --------------------------------
# 7. DISPLAY ANSWER
# --------------------------------

print("\n")
print("=" * 60)
print("PAPERLENS AI ANSWER")
print("=" * 60)

print(answer)


print("\n")
print("=" * 60)
print("RAG TEST COMPLETED")
print("=" * 60)