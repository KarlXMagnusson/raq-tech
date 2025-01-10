import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor
import os

# Set default input/output paths
INPUT_PDF_PATH = "./input.pdf"
OUTPUT_TXT_PATH = "./output.txt"

def pdf_to_images(pdf_path):
    """
    Convert PDF to images (for OCR).
    @param pdf_path: Path to input PDF file.
    @return: List of images (one per page).
    """
    images = convert_from_path(pdf_path)
    return images

def ocr_image(image):
    """
    Perform OCR on a single image using Tesseract.
    @param image: PIL Image object.
    @return: OCR-ed text.
    """
    text = pytesseract.image_to_string(image)
    return text

def extract_page_text(page_num, pdf_path):
    """
    Extract text from a PDF page and sort blocks to read in single-column order.
    @param page_num: Page number (for reference).
    @param pdf_path: Path to PDF file.
    @return: Extracted text.
    """
    doc = fitz.open(pdf_path)  # Open PDF within the function to avoid pickling issues
    page = doc.load_page(page_num)
    blocks = page.get_text("blocks")  # Extract blocks with coordinates
    sorted_blocks = sorted(blocks, key=lambda b: (b[1], b[0]))  # Sort y, then x for columns
    page_text = f"\n--- Page {page_num + 1} ---\n"

    for block in sorted_blocks:
        page_text += block[4].strip() + "\n"

    return page_text

def parallel_text_extraction(pdf_path):
    """
    Extract text from PDF in parallel and apply OCR.
    @param pdf_path: Path to the PDF file.
    @return: Full text of the PDF.
    """
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    results = [""] * total_pages

    with ThreadPoolExecutor() as executor:  # Use threads instead of processes
        futures = {executor.submit(extract_page_text, page_num, pdf_path): page_num for page_num in range(total_pages)}
        for future in futures:
            page_num = futures[future]
            results[page_num] = future.result()

    return "\n".join(results)

def pdf_to_text_with_ocr(pdf_path=INPUT_PDF_PATH, output_txt_path=OUTPUT_TXT_PATH):
    """
    Convert PDF to text (including OCR for images).
    @param pdf_path: Path to input PDF file.
    @param output_txt_path: Path to save the output .txt file.
    """
    # Convert PDF to images for OCR
    print("Performing OCR on PDF pages...")
    images = pdf_to_images(pdf_path)
    ocr_text = ""

    for i, img in enumerate(images):
        print(f"Performing OCR on page {i + 1}...")
        ocr_text += f"\n--- OCR Page {i + 1} ---\n"
        ocr_text += ocr_image(img)

    # Extract structured text from PDF using PyMuPDF
    print("Extracting structured text...")
    structured_text = parallel_text_extraction(pdf_path)

    # Combine OCR and structured text
    full_text = ocr_text + "\n\n--- Structured Text ---\n\n" + structured_text

    # Save the result to .txt file
    with open(output_txt_path, 'w') as f:
        f.write(full_text)

    print(f"Text extraction completed! Output saved to {output_txt_path}")

# Run the script
if __name__ == "__main__":
    pdf_to_text_with_ocr()