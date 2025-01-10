import os
import chromadb
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Default paths
INPUT_TXT_PATH = "./output.txt"
CHROMA_DB_DIR = "./chroma_db"

def read_text_file(file_path):
    """
    Reads text from a file.
    @param file_path: Path to the text file.
    @return: Text content of the file.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return ""
    with open(file_path, 'r') as f:
        text = f.read()
    print(f"Successfully read text file: {file_path}")
    print(f"Text length: {len(text)} characters")
    return text


def create_chroma_vector_store(text, embedding_model_name, db_directory):
    """
    Create a ChromaDB vector store and embed the text using Ollama embeddings.
    @param text: The full text to be embedded.
    @param embedding_model_name: The Ollama embedding model to use.
    @param db_directory: The directory to save the ChromaDB database.
    @return: Chroma collection with embedded vectors.
    """
    print(f"Initializing Ollama embedding model: {embedding_model_name}")
    embedding_model = OllamaEmbeddings(model=embedding_model_name)

    print(f"Setting up Chroma database at directory: {db_directory}")
    chroma_client = chromadb.PersistentClient(path=db_directory)
    collection = chroma_client.get_or_create_collection("MoD_database")

    # Split text into sections for embedding (for efficient retrieval)
    print("Splitting text into sections for embedding...")
    chunks = text.split("\n\n")  # Split by paragraph
    documents = [chunk.strip() for chunk in chunks if chunk.strip()]  # Keep only non-empty chunks

    print(f"Number of text chunks to embed: {len(documents)}")
    print("Starting embedding and storing process...")

    for i, doc in enumerate(documents):
        try:
            print(f"Embedding chunk {i + 1}/{len(documents)}: {doc[:100]}...")
            embedding = embedding_model.embed_query(doc)  # Embed the string directly
            collection.add(documents=[doc], embeddings=[embedding], ids=[str(i)])  # Store in ChromaDB
            print(f"Successfully stored chunk {i + 1}/{len(documents)} in the database.")
        except Exception as e:
            print(f"Error embedding or storing chunk {i + 1}: {e}")
            continue

    print(f"ChromaDB collection successfully created with {len(documents)} chunks.")
    return collection


def query_chroma_collection(collection, query_text, model_name="snowflake-arctic-embed2:568m-l-fp16"):
    """
    Query the ChromaDB collection using a given query text and return the response.
    @param collection: The ChromaDB collection to query.
    @param query_text: The query text.
    @param model_name: The Ollama model to use for querying.
    """
    print(f"\nQuerying database with text: '{query_text}'")
    print(f"Using model: {model_name}")
    llm = OllamaEmbeddings(model=model_name)

    try:
        response = collection.query(query_texts=[query_text], n_results=5)
        print(f"Response:\n{response}")
    except Exception as e:
        print(f"Error querying database: {e}")

def main():
    """
    Main function to read text, create ChromaDB collection, and run queries.
    """
    # Read the text file
    text = read_text_file(INPUT_TXT_PATH)
    if not text:
        print("Error: No text loaded from the file. Exiting.")
        return

    # Create Chroma vector store
    chroma_collection = create_chroma_vector_store(
        text=text,
        embedding_model_name="snowflake-arctic-embed2:568m-l-fp16",
        db_directory=CHROMA_DB_DIR
    )

    # Query the database
    query_text = "Tell me about the 'A Model of Design for Computing Systems: A Categorical Approach'"
    query_chroma_collection(chroma_collection, query_text)


if __name__ == "__main__":
    main()
