from pypdf import PdfReader
import os


def extract_text_from_pdf(pdf_path):

    pages = []

    reader = PdfReader(pdf_path)

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        if text.strip():

            pages.append({
                "paper": os.path.basename(pdf_path),
                "page": page_number,
                "text": text.strip()
            })

    return pages


def load_all_papers(folder="papers"):

    all_pages = []

    if not os.path.exists(folder):

        print("ERROR: papers folder not found")

        return []

    for filename in sorted(os.listdir(folder)):

        if filename.lower().endswith(".pdf"):

            path = os.path.join(folder, filename)

            try:

                pages = extract_text_from_pdf(path)

                all_pages.extend(pages)

                print(
                    f"Loaded: {filename} - "
                    f"{len(pages)} pages"
                )

            except Exception as e:

                print(
                    f"ERROR reading {filename}: {e}"
                )

    return all_pages