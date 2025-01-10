# Imports
import nltk
import ssl
from langchain.document_loaders import PyMuPDFLoader  # Alternative PDF loader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama

# SSL fix for NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Explicit NLTK Path Configuration
nltk.data.path.append("/Users/tage/nltk_data")  # Ensure this points to your correct path

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# ==== PDF Ingestion ====
doc_path = "./data/input.pdf"  # Your input PDF path
model = "llama3.3:70b-instruct-q8_0"  # Model for embeddings

if doc_path:
    loader = PyMuPDFLoader(file_path=doc_path)  # Using PyMuPDFLoader for PDF loading
    try:
        data = loader.load()  # Load PDF content
        print("Done loading....")
    except Exception as e:
        print(f"Error loading PDF: {e}")
        exit(1)
else:
    print("Please upload a PDF file.")
    exit(1)

# Preview first page content
if data:
    content = data[0].page_content
    print(f"First 100 characters:\n{content[:100]}")

# ==== Text Splitting ====
try:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(data)
    print("Done splitting....")
except Exception as e:
    print(f"Error during text splitting: {e}")
    exit(1)

# ===== Add to Vector Database ====
try:
    ollama.pull("snowflake-arctic-embed2:568m-l-fp16")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="snowflake-arctic-embed2:568m-l-fp16"),
        collection_name="simple-rag",
    )
    print("Done adding to vector database....")
except Exception as e:
    print(f"Error setting up vector database: {e}")
    exit(1)

# ==== Retrieval ====
llm = ChatOllama(model=model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
)

template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# === Run the Query ===
try:
    res = chain.invoke(input=("how to use the Model of Design in my embedded system?",))
    print(res)
except Exception as e:
    print(f"Error during query execution: {e}")