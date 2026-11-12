import os
import time
import pymupdf as fitz  # Modern PyMuPDF import (no warning!)
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load GEMINI_API_KEY from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file!")

client = genai.Client(api_key=api_key)

pdf_folder = "pdfs"
parsed_folder = "parsed_docs"
figures_folder = "extracted_figures"

for folder in [pdf_folder, parsed_folder, figures_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)


# 1. Universal Page Inspector: Catches ANY figure, pie chart, wrapped column, or rotated table
def needs_vision_extraction(page, raw_text):
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

    # Rule 1: Scanned page or very little text
    if len(lines) < 5 or len(raw_text) < 250:
        return True, "Low text density (full-page figure or scan)"

    # Rule 2: Page has embedded bitmap images (photos, SEM micrographs, scanned figures)
    if len(page.get_images()) > 0:
        return True, f"Contains {len(page.get_images())} embedded image(s)/figure(s) with wrapped text"

    # Rule 3: Page mentions a Figure, Table, or Chart, OR has vector drawings (pie charts, graphs)
    text_lower = raw_text.lower()
    has_visual_keyword = any(k in text_lower for k in [
                             "figure ", "fig. ", "table ", "chart", "graph"])
    has_vector_drawings = len(page.get_drawings()) > 15

    if has_visual_keyword or has_vector_drawings:
        return True, "Contains Table, Pie Chart, Graph, or Figure reference"

    # Rule 4: Check for narrow wrapped-text lines (3+ consecutive short lines) or garbled OCR
    consecutive_short = 0
    max_consecutive_short = 0
    tiny_lines = 0

    for line in lines:
        if len(line) <= 3:
            tiny_lines += 1
        if len(line) < 35 and not line.endswith("."):
            consecutive_short += 1
            if consecutive_short > max_consecutive_short:
                max_consecutive_short = consecutive_short
        else:
            consecutive_short = 0

    if (tiny_lines / len(lines)) > 0.25:
        return True, "Garbled or sideways rotated OCR detected"

    if max_consecutive_short >= 3:
        return True, "Narrow wrapped-column text detected"

    return False, "Pure clean text page"


# 2. Stitch clean text lines (only used for 100% pure text pages with no figures/tables)
def stitch_clean_text(raw_text):
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    paragraphs = []
    current = []

    for line in lines:
        current.append(line)
        if line.endswith(".") or line.endswith(":"):
            paragraphs.append(" ".join(current))
            current = []

    if len(current) > 0:
        paragraphs.append(" ".join(current))

    return "\n\n".join(paragraphs)


# 3. Vision AI Extractor using gemini-3.5-flash-lite
def extract_page_with_vision(png_bytes, page_num, pdf_name):
    prompt = f"""You are an expert metallurgical document parser reading Page {page_num} of '{pdf_name}'.
Convert this page into clean, accurate Markdown following these strict rules:
1. CONTINUOUS PARAGRAPHS: Read the body text naturally around any embedded figures. NEVER splice a figure caption into the middle of a sentence. Fix obvious OCR typos (e.g., 'II. Case Studies' instead of 'n. Case Studies', 'Fig. 1' instead of 'Fig. J', 'ISS' instead of 'LSS').
2. TABLES: If there is any table (upright or rotated sideways), output the complete table in clean Markdown pipe format (| Col 1 | Col 2 |).
3. CHARTS & PIE CHARTS: If there is a pie chart, bar chart, or S-N curve, extract all labels, categories, and numbers/percentages into a Markdown table or bullet list.
4. FIGURES & MICROGRAPHS: At the bottom of the page output, create a section titled '### Visual Figures & Engineering Description' that lists each Figure caption on the page followed by a clear description of what the photograph, arrows, notches, weld repairs, or SEM fractographs show.
Output ONLY the clean Markdown for this page."""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=[
                    types.Part.from_bytes(
                        data=png_bytes, mime_type="image/png"),
                    prompt
                ],
                config=types.GenerateContentConfig(temperature=0.0)
            )
            time.sleep(4)
            if response.text:
                return response.text.strip()
            return "*[Blank or cover page]*"
        except Exception as e:
            print(f"      [Retry {attempt + 1}/3] Vision API waiting ({e})...")
            time.sleep(10)

    return "*[Vision extraction unavailable for this page]*"


# 4. Universal PDF Parser
def parse_all_pdfs_universally():
    pdf_files = [f for f in os.listdir(
        pdf_folder) if f.lower().endswith(".pdf")]

    if len(pdf_files) == 0:
        print("No PDFs found in 'pdfs/' folder!")
        return

    print(
        f"\n--- Running Universal Smart Parser on {len(pdf_files)} PDF(s) ---")
    for pdf_name in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_name)
        base_name = pdf_name[:-4]
        save_path = os.path.join(parsed_folder, base_name + ".md")

        if os.path.exists(save_path):
            print(
                f" [Already parsed] {base_name}.md (Delete this .md file to re-parse)")
            continue

        print(f"\n-> Opening PDF: {pdf_name}")
        doc = fitz.open(pdf_path)
        pages_markdown = [f"# Document: {pdf_name}\n"]

        for page_index in range(len(doc)):
            page = doc[page_index]
            page_num = page_index + 1
            raw_text = page.get_text("text").strip()

            use_vision, reason = needs_vision_extraction(page, raw_text)

            if use_vision:
                print(
                    f"   Page {page_num}/{len(doc)}: [VISION AI] -> {reason}")
                pix = page.get_pixmap(dpi=150)
                png_bytes = pix.tobytes("png")

                img_path = os.path.join(
                    figures_folder, f"{base_name}_Page_{page_num}.png")
                pix.save(img_path)

                page_md = extract_page_with_vision(
                    png_bytes, page_num, pdf_name)
                section = f"\n## Page {page_num}\n{page_md}\n\n**[PAGE_SNAPSHOT: {img_path}]**\n"
            else:
                print(
                    f"   Page {page_num}/{len(doc)}: [FAST TEXT] -> Pure clean text page")
                clean_text = stitch_clean_text(raw_text)
                section = f"\n## Page {page_num}\n{clean_text}\n"

            pages_markdown.append(section)

        doc.close()
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(pages_markdown))

        print(f"Saved universally parsed Markdown: {save_path}")


if __name__ == "__main__":
    parse_all_pdfs_universally()
    print("\n=== UNIVERSAL STAGE 2 PARSING COMPLETE! ===")
