import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor

# Default paths
INPUT_PDF_PATH = "./input.pdf"
OUTPUT_TXT_PATH = "./output.txt"

def extract_page_text_ordered(page_num, pdf_path):
    """
    Extracts text from a PDF page in vertical-then-horizontal order.
    @param page_num: Page number (0-indexed).
    @param pdf_path: Path to PDF file.
    @return: Ordered page text.
    """
    doc = fitz.open(pdf_path)  # Open PDF file
    page = doc.load_page(page_num)
    blocks = page.get_text("blocks")  # Extract text blocks with positions (x, y, width, height)

    # Sort by y-axis (top-to-bottom) first, then x-axis (left-to-right) within the same vertical region
    sorted_blocks = sorted(blocks, key=lambda b: (round(b[1] / 20) * 20, b[0]))  # Rounding to avoid noise in coordinates

    page_text = f"\n--- Page {page_num + 1} ---\n"

    for block in sorted_blocks:
        page_text += block[4].strip() + "\n\n"  # Add the text content of the block with spacing

    return page_text

def parallel_text_extraction(pdf_path):
    """
    Extract text from PDF in parallel using threads.
    @param pdf_path: Path to the PDF file.
    @return: Full text of the PDF.
    """
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    results = [""] * total_pages

    # Use threads for parallel execution
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(extract_page_text_ordered, page_num, pdf_path): page_num for page_num in range(total_pages)}
        for future in futures:
            page_num = futures[future]
            results[page_num] = future.result()  # Get result for each page

    return "\n".join(results)

def pdf_to_text_no_ocr(pdf_path=INPUT_PDF_PATH, output_txt_path=OUTPUT_TXT_PATH):
    """
    Convert PDF to text (without OCR).
    @param pdf_path: Path to input PDF file.
    @param output_txt_path: Path to save the output .txt file.
    """
    print("Extracting structured text...")
    structured_text = parallel_text_extraction(pdf_path)

    # Save the text to the output .txt file
    with open(output_txt_path, 'w') as f:
        f.write(structured_text)

    print(f"Text extraction completed! Output saved to {output_txt_path}")

# Run the script
if __name__ == "__main__":
    pdf_to_text_no_ocr()