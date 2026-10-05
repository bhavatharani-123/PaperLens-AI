import os

from dotenv import load_dotenv
from google import genai


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:

    raise ValueError(
        "GEMINI_API_KEY is not configured in .env"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "gemini-3.6-flash"


# ============================================================
# GENERATE GROUNDED ANSWER
# ============================================================

def generate_answer(question, results):

    if not results:

        return (
            "The answer is not available in "
            "the indexed research papers."
        )

    context_parts = []

    for i, result in enumerate(results, start=1):

        context_parts.append(
            f"""
SOURCE {i}

Paper: {result["paper"]}
Page: {result["page"]}

Evidence:
{result["text"]}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are PaperLens, a research paper question-answering system.

Answer the user's question ONLY from the supplied
research-paper evidence.

RULES:

1. Use only the supplied evidence.
2. Do not use outside knowledge.
3. Do not invent information.
4. Do not combine unrelated papers.
5. Prefer the source that directly answers the question.
6. If the evidence does not contain the answer, say:

"The answer is not available in the indexed research papers."

7. Keep the answer concise and easy to understand.
8. Explain technical terms briefly when useful.
9. For important claims, mention the paper name and page.
10. Never mention similarity scores.
11. Never mention retrieval scores.
12. Do not say "according to my knowledge".

QUESTION:

{question}

RESEARCH-PAPER EVIDENCE:

{context}

ANSWER:
"""

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        if response and response.text:

            return response.text.strip()

        return (
            "The model returned an empty answer."
        )

    except Exception as e:

        return (
            "The answer could not be generated "
            "at this moment. Please try again."
        )