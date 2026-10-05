import re
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PaperLens
# Research Paper Intelligence & Evidence Retrieval
# ============================================================

st.set_page_config(
    page_title="PaperLens",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Configuration
# ============================================================

# IMPORTANT:
# Always resolve paths relative to app.py.
# This prevents "0 papers" when Streamlit is started from
# another working directory.
BASE_DIR = Path(__file__).resolve().parent

# Primary folder for research papers.
DATA_DIR = BASE_DIR / "data"

# Also support these folders automatically if they exist.
ALTERNATIVE_DIRS = [
    BASE_DIR / "papers",
    BASE_DIR / "pdfs",
]

TOP_K = 5
MIN_SCORE = 0.03
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
    /* ---------- Main application ---------- */
    .stApp {
        background: #08101f !important;
    }

    [data-testid="stAppViewContainer"] {
        background: #08101f !important;
    }

    [data-testid="stHeader"] {
        background: #08101f !important;
    }

    [data-testid="stToolbar"] {
        visibility: hidden;
    }

    /* ---------- Main content ---------- */
    .block-container {
        max-width: 1280px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: #111b2d !important;
        border-right: 1px solid #263957 !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: #111b2d !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #dbe5f2 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: #2b3d59 !important;
    }

    /* ---------- Brand ---------- */
    .brand {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.7px;
        margin-bottom: 0.15rem;
    }

    .brand-white {
        color: #f5f7fb;
    }

    .brand-blue {
        color: #3b82f6;
    }

    .subtitle {
        color: #8292ad;
        font-size: 0.86rem;
        margin-bottom: 1.8rem;
    }

    /* ---------- Metrics ---------- */
    div[data-testid="stMetric"] {
        background: #111d31 !important;
        border: 1px solid #263957 !important;
        border-radius: 12px !important;
        padding: 1.15rem 1.25rem !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #9aa9bf !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f3f7fd !important;
        font-weight: 750 !important;
    }

    /* ---------- Question input ---------- */
    div[data-testid="stTextInput"] input {
        background: #f4f6fa !important;
        color: #1c2533 !important;
        border: 1px solid #d5dbe5 !important;
        border-radius: 8px !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #7b8799 !important;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        background: #2563eb !important;
        color: #ffffff !important;
        border: 0 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        min-height: 42px !important;
    }

    .stButton > button:hover {
        background: #1d4ed8 !important;
        color: #ffffff !important;
    }

    /* ---------- Dark content boxes ---------- */
    .dark-box {
        background: #101c30;
        border: 1px solid #263957;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        margin-top: 0.8rem;
        margin-bottom: 1rem;
    }

    .answer-text {
        color: #e5ebf4;
        line-height: 1.75;
    }

    .footer {
        color: #667892;
        text-align: center;
        font-size: 0.72rem;
        margin-top: 2.2rem;
    }

    /* ---------- Expanders ---------- */
    [data-testid="stExpander"] {
        background: #0e1829 !important;
        border: 1px solid #22344f !important;
        border-radius: 9px !important;
    }

    /* Keep Streamlit's normal markdown readable */
    [data-testid="stMarkdownContainer"] {
        color: #e5ebf4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Text utilities
# ============================================================

def clean_text(text: str) -> str:
    """Clean PDF text and remove accidental HTML/entities."""
    if not text:
        return ""

    text = str(text)

    # Remove HTML tags only if they accidentally occur in PDF text.
    text = re.sub(r"<[^>]+>", " ", text)

    replacements = {
        "&nbsp;": " ",
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Split a page into overlapping retrieval chunks."""
    text = clean_text(text)

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def sentence_split(text: str) -> List[str]:
    """Split text into readable sentences."""
    text = clean_text(text)

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9])",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) >= 25
    ]


# ============================================================
# PDF discovery
# ============================================================

def find_pdf_files() -> List[Path]:
    """
    Find PDFs reliably.

    Priority:
      1. data/
      2. papers/
      3. pdfs/

    If none of those folders contain PDFs, also check the project
    root. This makes the app tolerant of different project layouts.
    """
    search_dirs = [DATA_DIR] + ALTERNATIVE_DIRS

    found: Dict[str, Path] = {}

    for folder in search_dirs:
        if not folder.exists() or not folder.is_dir():
            continue

        for pdf in folder.rglob("*.pdf"):
            if pdf.is_file():
                found[str(pdf.resolve())] = pdf

    # Fallback: PDFs directly beside app.py.
    if not found:
        for pdf in BASE_DIR.glob("*.pdf"):
            if pdf.is_file():
                found[str(pdf.resolve())] = pdf

    return sorted(found.values(), key=lambda p: p.name.lower())


def pdf_signature(pdf_files: List[Path]) -> Tuple[Tuple[str, int, int], ...]:
    """
    Create a cache signature from file path + modification time + size.

    This fixes the important problem where Streamlit keeps an old
    cached "0 papers" result after PDFs are added.
    """
    signature = []

    for pdf in pdf_files:
        try:
            stat = pdf.stat()
            signature.append(
                (
                    str(pdf.resolve()),
                    stat.st_mtime_ns,
                    stat.st_size,
                )
            )
        except OSError:
            continue

    return tuple(signature)


# ============================================================
# PDF loading
# ============================================================

@st.cache_data(show_spinner=False)
def load_papers(
    file_signature: Tuple[Tuple[str, int, int], ...]
) -> Tuple[List[Dict], Dict]:
    """
    Read all discovered PDFs.

    file_signature is deliberately part of the cache key so that
    adding/removing/changing PDFs automatically reloads the index.
    """
    chunks: List[Dict] = []
    paper_count = 0
    page_count = 0
    failed_files: List[str] = []

    for file_path, _, _ in file_signature:
        pdf_path = Path(file_path)

        try:
            reader = PdfReader(str(pdf_path))
        except Exception:
            failed_files.append(pdf_path.name)
            continue

        paper_count += 1
        page_count += len(reader.pages)

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
            except Exception:
                raw_text = ""

            text = clean_text(raw_text)

            # Scanned/image-only pages may contain no extractable text.
            if not text:
                continue

            page_chunks = split_into_chunks(text)

            for chunk_number, chunk in enumerate(page_chunks, start=1):
                chunks.append(
                    {
                        "paper": pdf_path.name,
                        "page": page_number,
                        "chunk": chunk_number,
                        "text": chunk,
                    }
                )

    stats = {
        "papers": paper_count,
        "pages": page_count,
        "chunks": len(chunks),
        "failed_files": failed_files,
    }

    return chunks, stats


# ============================================================
# TF-IDF retrieval
# ============================================================

@st.cache_resource(show_spinner=False)
def build_vector_index(texts: Tuple[str, ...]):
    """Build and cache the TF-IDF matrix."""
    if not texts:
        return None, None

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return None, None

    return vectorizer, matrix


def retrieve(
    query: str,
    chunks: List[Dict],
    top_k: int = TOP_K,
    min_score: float = MIN_SCORE,
) -> List[Dict]:
    """Retrieve the top matching chunks using cosine similarity."""
    query = clean_text(query)

    if not query or not chunks:
        return []

    texts = tuple(item["text"] for item in chunks)

    vectorizer, matrix = build_vector_index(texts)

    if vectorizer is None or matrix is None:
        return []

    query_vector = vectorizer.transform([query])

    if query_vector.nnz == 0:
        return []

    scores = cosine_similarity(query_vector, matrix).ravel()
    ranked_indices = scores.argsort()[::-1]

    results: List[Dict] = []

    for index in ranked_indices:
        score = float(scores[index])

        if score < min_score:
            continue

        item = dict(chunks[index])
        item["score"] = score
        results.append(item)

        if len(results) >= top_k:
            break

    return results


# ============================================================
# Extractive answer generation
# ============================================================

def generate_answer(query: str, results: List[Dict]) -> List[Dict]:
    """
    Create a transparent answer using only retrieved PDF text.

    No external LLM is required.
    """
    if not results:
        return []

    query_terms = {
        word.lower()
        for word in re.findall(r"[A-Za-z0-9]+", query)
        if len(word) >= 3
    }

    candidates = []

    for result in results:
        sentences = sentence_split(result["text"])

        if not sentences:
            sentences = [result["text"]]

        for sentence in sentences:
            words = {
                word.lower()
                for word in re.findall(r"[A-Za-z0-9]+", sentence)
            }

            overlap = len(query_terms.intersection(words))

            candidates.append(
                (
                    overlap,
                    result["score"],
                    sentence,
                    result["paper"],
                    result["page"],
                )
            )

    candidates.sort(
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )

    selected: List[Dict] = []
    seen = set()

    for overlap, score, sentence, paper, page in candidates:
        normalized = re.sub(r"\W+", "", sentence.lower())

        if not normalized or normalized in seen:
            continue

        seen.add(normalized)

        selected.append(
            {
                "text": sentence,
                "paper": paper,
                "page": page,
                "score": score,
                "overlap": overlap,
            }
        )

        if len(selected) >= 5:
            break

    return selected


# ============================================================
# Sidebar
# ============================================================

def render_sidebar(stats: Dict, pdf_files: List[Path]) -> None:
    """Render a clean native Streamlit sidebar."""
    with st.sidebar:
        st.markdown("## PaperLens")
        st.caption("Research Paper Intelligence")

        st.divider()

        st.markdown("### About")
        st.write(
            "PaperLens retrieves relevant information from research "
            "papers and presents evidence-based answers."
        )

        st.divider()

        st.markdown("### Retrieval")
        st.write("**Method:** TF-IDF + Cosine Similarity")
        st.write(f"**Top Results:** {TOP_K}")
        st.write(f"**Minimum Score:** {MIN_SCORE:.2f}")

        st.divider()

        # Useful status without exposing HTML/code.
        if stats["papers"] > 0:
            st.success(f"{stats['papers']} paper(s) loaded")
        else:
            st.warning("No PDF papers found")

        st.caption("PaperLens · Research Paper QA")


# ============================================================
# Main UI
# ============================================================

def main() -> None:
    # Discover PDFs BEFORE loading them.
    pdf_files = find_pdf_files()

    # Signature changes automatically when files are added/modified.
    signature = pdf_signature(pdf_files)

    chunks, stats = load_papers(signature)

    render_sidebar(stats, pdf_files)

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------
    st.markdown(
        '<div class="brand">'
        '<span class="brand-white">Paper</span>'
        '<span class="brand-blue">Lens</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Research Paper Intelligence &amp; Evidence Retrieval'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Papers", stats["papers"])

    with col2:
        st.metric("Pages", stats["pages"])

    with col3:
        st.metric("Chunks", stats["chunks"])

    # --------------------------------------------------------
    # Missing-PDF guidance
    # --------------------------------------------------------
    if stats["papers"] == 0:
        st.error(
            "No research PDFs were found. Put your PDF files inside "
            f"'{DATA_DIR.name}/' next to app.py, then click Refresh."
        )

        if stats["failed_files"]:
            st.warning(
                "Some PDF files could not be opened: "
                + ", ".join(stats["failed_files"])
            )

        refresh_col, path_col = st.columns([1, 5])

        with refresh_col:
            if st.button("Refresh Papers", use_container_width=True):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()

        with path_col:
            st.caption(
                f"Expected folder: {DATA_DIR}"
            )

    elif stats["chunks"] == 0:
        st.warning(
            "PDF files were found, but no selectable text could be extracted. "
            "If these are scanned/image-only PDFs, OCR is required."
        )

    # --------------------------------------------------------
    # Question
    # --------------------------------------------------------
    st.markdown("### Ask a Research Question")

    if "question" not in st.session_state:
        st.session_state.question = ""

    if "run_query" not in st.session_state:
        st.session_state.run_query = ""

    input_col, button_col = st.columns([6.5, 1.0])

    with input_col:
        question = st.text_input(
            "Research question",
            value=st.session_state.question,
            placeholder="Ask a question about the research papers...",
            label_visibility="collapsed",
            key="question_input",
        )

    with button_col:
        ask = st.button(
            "Ask PaperLens",
            type="primary",
            use_container_width=True,
        )

    if ask:
        cleaned_question = clean_text(question)

        if not cleaned_question:
            st.session_state.run_query = ""
            st.warning("Please enter a research question.")
        elif stats["chunks"] == 0:
            st.session_state.run_query = ""
            st.error(
                "No searchable paper text is available. "
                "Add PDFs to the data folder first."
            )
        else:
            st.session_state.question = cleaned_question
            st.session_state.run_query = cleaned_question

    active_query = st.session_state.get("run_query", "")

    # --------------------------------------------------------
    # Retrieval information
    # --------------------------------------------------------
    st.markdown('<div class="dark-box">', unsafe_allow_html=True)

    st.markdown(
        "**Retrieval:** Top 5 results &nbsp; | &nbsp; "
        "**Similarity:** Cosine Similarity (TF-IDF) &nbsp; | &nbsp; "
        f"**Min. Score:** {MIN_SCORE:.2f}"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Answer
    # --------------------------------------------------------
    st.markdown("### PaperLens Answer")

    if not active_query:
        st.info(
            "Ask a question about the research papers to retrieve "
            "evidence and generate an answer."
        )

    else:
        results = retrieve(
            active_query,
            chunks,
            top_k=TOP_K,
            min_score=MIN_SCORE,
        )

        if not results:
            st.warning(
                "No relevant evidence was found above the minimum "
                f"similarity score of {MIN_SCORE:.2f}."
            )
        else:
            answer_items = generate_answer(active_query, results)

            st.markdown('<div class="dark-box">', unsafe_allow_html=True)

            if answer_items:
                st.markdown("**Evidence-based answer**")

                for item in answer_items:
                    st.markdown(
                        f"{item['text']} "
                        f"(*{item['paper']}*, Page {item['page']})."
                    )
            else:
                st.write(
                    "Relevant evidence was retrieved, but a clean "
                    "sentence-level answer could not be generated."
                )

            st.markdown("</div>", unsafe_allow_html=True)

        # ----------------------------------------------------
        # Supporting evidence
        # ----------------------------------------------------
        st.markdown("### Supporting Evidence")

        if not results:
            st.info("No supporting evidence matched the question.")
        else:
            for number, result in enumerate(results, start=1):
                paper = result["paper"]
                page = result["page"]
                score = result["score"]

                with st.expander(
                    f"{number}. {paper}  ·  Page {page}  ·  "
                    f"Similarity {score:.4f}"
                ):
                    st.write(result["text"])

    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------
    st.markdown(
        '<div class="footer">'
        'PaperLens · Retrieval-Augmented Research Paper Question Answering'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()