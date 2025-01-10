import fitz  # PyMuPDF
import unicodedata

def extract_page_text_utf8(page_num, pdf_path):
    """
    Extract text as individual UTF-8 glyphs (letters) to support all characters, including Greek.
    @param page_num: Page number (0-indexed).
    @param pdf_path: Path to PDF file.
    @return: Extracted UTF-8 text for the page.
    """
    doc = fitz.open(pdf_path)  # Open PDF file
    page = doc.load_page(page_num)
    
    # Extract glyphs (character-level information)
    glyphs = page.get_text("rawdict")["blocks"]

    page_text = f"\n--- Page {page_num + 1} ---\n"

    for block in glyphs:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text_utf8 = span.get("text", "").strip()  # Extract the text
                if text_utf8:
                    # Ensure UTF-8 decoding
                    normalised_text = unicodedata.normalize("NFC", text_utf8)  # Normalize Unicode to composed form
                    page_text += normalised_text + " "  # Add a space between spans

    return page_text + "\n\n"

def extract_full_text_utf8(pdf_path):
    """
    Extract full text as UTF-8 from the PDF.
    """
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    all_text = ""

    for page_num in range(total_pages):
        all_text += extract_page_text_utf8(page_num, pdf_path)

    return all_text

def pdf_to_text_utf8(pdf_path, output_txt_path):
    """
    Save extracted UTF-8 text to an output file.
    """
    utf8_text = extract_full_text_utf8(pdf_path)
    
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(utf8_text)

    print(f"Text extraction completed! Output saved to {output_txt_path}")

# Usage Example
if __name__ == "__main__":
    INPUT_PDF_PATH = "./input.pdf"
    OUTPUT_TXT_PATH = "./output_utf8.txt"
    pdf_to_text_utf8(INPUT_PDF_PATH, OUTPUT_TXT_PATH)