import os
import fitz  # PyMuPDF
import chromadb
from concurrent.futures import ThreadPoolExecutor
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Paths
INPUT_PDF_PATH = "./input.pdf"
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "snowflake-arctic-embed2:568m-l-fp16" #"snowflake-arctic-embed2:568m-l-fp16","all-MiniLM-L6-v2" 

def extract_page_text_ordered(page_num, pdf_path):
    """
    Extract text from a PDF page in vertical-then-horizontal order.
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
    @return: List of text chunks (per page).
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

    return results  # Return list of pages as chunks

def create_chroma_vector_store_from_pdf(pdf_text_chunks, embedding_model_name, db_directory):
    """
    Create a ChromaDB vector store and embed each page as a separate document.
    @param pdf_text_chunks: List of text chunks per page.
    @param embedding_model_name: The Ollama embedding model to use.
    @param db_directory: The directory to save the ChromaDB database.
    @return: Chroma collection with embedded vectors.
    """
    print(f"Initializing Ollama embedding model: {embedding_model_name}")
    embedding_model = OllamaEmbeddings(model=embedding_model_name)

    print(f"Setting up Chroma database at directory: {db_directory}")
    chroma_client = chromadb.PersistentClient(path=db_directory)
    collection = chroma_client.get_or_create_collection("MoD_database")

    print(f"Number of pages to embed: {len(pdf_text_chunks)}")
    for i, page_text in enumerate(pdf_text_chunks):
        if not page_text.strip():
            print(f"Skipping empty page {i + 1}.")
            continue

        try:
            print(f"Embedding page {i + 1}/{len(pdf_text_chunks)}...")
            embedding = embedding_model.embed_query(page_text)
            collection.add(documents=[page_text], embeddings=[embedding], ids=[f"page_{i + 1}"])
            print(f"Successfully stored page {i + 1} in the database.")
        except Exception as e:
            print(f"Error embedding or storing page {i + 1}: {e}")
            continue

    print(f"ChromaDB collection successfully created with {len(pdf_text_chunks)} pages.")
    return collection

def query_chroma_collection(collection, query_text, model_name=EMBEDDING_MODEL):
    """
    Query the ChromaDB collection using a given query text and return the response.
    @param collection: The ChromaDB collection to query.
    @param query_text: The query text.
    @param model_name: The Ollama model to use for querying.
    """
    print(f"\nQuerying database with text: '{query_text}'")
    print(f"Using model: {model_name}")
    embedding_model = OllamaEmbeddings(model=model_name)

    try:
        response = collection.query(query_texts=[query_text], n_results=5)
        print(f"Response:\n{response}")
    except Exception as e:
        print(f"Error querying database: {e}")

def main():
    """
    Main function to extract text from PDF, create ChromaDB collection, and run queries.
    """
    # Extract structured text from PDF
    print("Extracting text from PDF...")
    pdf_text_chunks = parallel_text_extraction(INPUT_PDF_PATH)

    # Create Chroma vector store from PDF pages
    chroma_collection = create_chroma_vector_store_from_pdf(
        pdf_text_chunks=pdf_text_chunks,
        embedding_model_name=EMBEDDING_MODEL,
        db_directory=CHROMA_DB_DIR
    )

    # Query the database
    query_text = "Tell me about the 'A Model of Design for Computing Systems: A Categorical Approach'"
    query_chroma_collection(chroma_collection, query_text)


if __name__ == "__main__":
    main()
