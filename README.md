# PaperLens AI

PaperLens is a Streamlit app for searching research papers and answering questions with passages cited by paper and page.

## Requirements

- Python 3.10 or newer
- Research-paper PDFs that you have permission to use
- A Gemini API key only for the optional `llm/llm.py` command-line demo

## Run the app

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r llm\requirements.txt
streamlit run llm\app.py
```

Place PDFs in `llm/data/` (or `llm/papers/` or `llm/pdfs/`). The app indexes PDFs from those folders when it starts. Paper PDFs are not included in this repository; add only documents you are authorized to use.

## Optional Gemini demo

The separate `llm/test_rag.py` demo uses Gemini. Set `GEMINI_API_KEY` in an environment variable or in `llm/.env` before running it:

```powershell
$env:GEMINI_API_KEY = "your-api-key"
python llm\test_rag.py
```

Do not commit `.env` files or API keys.
