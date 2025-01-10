import os
import fitz  # PyMuPDF
import logging
from concurrent.futures import ThreadPoolExecutor
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.llms import Ollama
from langchain.runnables import RunnablePassthrough
from langchain.output_parsers import StrOutputParser

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Paths and models
INPUT_PDF_PATH = "./input.pdf"
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "snowflake-arctic-embed2:568m-l-fp16"  # Custom embedding model
LLM_MODEL_NAME = "phi4:14b-fp16"


# Step 1: PDF Text Extraction
def extract_page_text_ordered(page_num, pdf_path):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    blocks = page.get_text("blocks")
    sorted_blocks = sorted(blocks, key=lambda b: (round(b[1] / 20) * 20, b[0]))
    page_text = f"\n--- Page {page_num + 1} ---\n"

    for block in sorted_blocks:
        page_text += block[4].strip() + "\n\n"

    return page_text


def parallel_text_extraction(pdf_path):
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    results = [""] * total_pages

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(extract_page_text_ordered, page_num, pdf_path): page_num for page_num in range(total_pages)}
        for future in futures:
            page_num = futures[future]
            results[page_num] = future.result()

    return "\n".join(results)


# Step 2: Create Vector Store
def create_vector_store(text, embedding_model_name, db_directory):
    logging.info(f"Initializing embedding model: {embedding_model_name}")
    embedding_function = OllamaEmbeddings(model=embedding_model_name)

    logging.info(f"Setting up Chroma database at {db_directory}")
    vector_db = Chroma(persist_directory=db_directory, embedding_function=embedding_function)

    pages = text.split("--- Page")
    logging.info(f"Number of pages to embed: {len(pages)}")
    for i, page_text in enumerate(pages):
        if not page_text.strip():
            continue
        page_id = f"page_{i}"
        logging.info(f"Embedding page {i}/{len(pages)}...")
        vector_db.add_texts(texts=[page_text], ids=[page_id])

    vector_db.persist()
    logging.info("Vector store successfully created.")
    return vector_db


# Step 3: Create Retriever and Query Chain
def create_retriever(vector_db, llm_model):
    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""Generate five diverse versions of the given user question:
Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(vector_db.as_retriever(), llm_model, prompt=query_prompt)
    logging.info("Retriever created.")
    return retriever


def create_chain(retriever, llm_model):
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
    prompt = PromptTemplate.from_template(template)
    chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm_model | StrOutputParser()
    logging.info("Chain created successfully.")
    return chain


# Main Execution
def main():
    if not os.path.exists(INPUT_PDF_PATH):
        logging.error(f"PDF file not found at {INPUT_PDF_PATH}.")
        return

    # Extract text from PDF
    logging.info("Extracting text from PDF...")
    pdf_text = parallel_text_extraction(INPUT_PDF_PATH)

    # Create vector store
    vector_store = create_vector_store(pdf_text, EMBEDDING_MODEL, CHROMA_DB_DIR)

    # Load language model
    llm_model = ChatOllama(model=LLM_MODEL_NAME)

    # Create retriever and chain
    retriever = create_retriever(vector_store, llm_model)
    chain = create_chain(retriever, llm_model)

    # Example query
    query = "Tell me about 'A Model of Design for Computing Systems: A Categorical Approach'"
    response = chain.invoke({"question": query})
    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()
