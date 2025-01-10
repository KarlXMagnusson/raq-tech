# Imports
import nltk
import ssl
from langchain.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama
import time
from datetime import datetime
import random

# SSL fix for NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Explicit NLTK Path Configuration
nltk.data.path.append("/Users/tage/nltk_data")
nltk.download('punkt', quiet=True)

# ==== PDF Ingestion ====
doc_path = "./data/input.pdf"

if not doc_path:
    print("Please upload a PDF file.")
    exit(1)

loader = PyMuPDFLoader(file_path=doc_path)
data = loader.load()
print("Done loading PDF....")

# ==== Text Splitting ====
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks = text_splitter.split_documents(data)
print(f"Done splitting PDF into {len(chunks)} chunks....")

# ===== Add to Vector Database ====
ollama.pull("snowflake-arctic-embed2:568m-l-fp16")
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="snowflake-arctic-embed2:568m-l-fp16"),
    collection_name="simple-rag",
)
print("Done adding to vector database....")

# ==== Parameters ====
user_prompt = "tell me about 'A Model of Design for Computing Systems: A Categorical Approach'."

test_all_models = False        # Do not query all models by default
test_random_n_models = False   # Do not query random N models by default
num_random_models = 3          # Number of random models to pick

specific_models_to_test = [
    "phi4:14b-fp16",
    "qwen2.5:14b-instruct-fp16",
    "qwen2.5:32b-instruct-fp16",
    "tulu3:70b-q8_0",
    "reflection:70b-q8_0",
    "athene-v2:72b-q8_0",
    "qwen2.5-coder:32b-instruct-fp16",
    "qwen2-math:72b-instruct-q8_0",
    "qwen2.5:72b-instruct-q8_0",
    "llama3.3:70b-instruct-q8_0",
    "marco-o1:7b-fp16",
]

fact_check_model = "bespoke-minicheck:7b-fp16"
check_responses = True

# ==== Get Available Models ====
all_models_response = ollama.list()
all_models = all_models_response.models

# Print available models
print("=== List of Available Models ===")
header = f"{'Model':50s} | {'Modified':19s} | {'Size(GiB)':9s} | {'Params':7s} | {'Quant':8s}"
print(header)
print("-" * len(header))

for m in all_models:
    model_name = m.model
    modified_str = m.modified_at.strftime("%Y-%m-%d %H:%M:%S") if m.modified_at else "N/A"
    size_gib = m.size / (1024 ** 3) if m.size else 0.0
    param_size = m.details.parameter_size if (m.details and m.details.parameter_size) else "N/A"
    quant_level = m.details.quantization_level if (m.details and m.details.quantization_level) else "N/A"
    print(f"{model_name:50s} | {modified_str:19s} | {size_gib:9.2f} | {param_size:7s} | {quant_level:8s}")

# ==== Model Selection Logic ====
if test_random_n_models:
    models_to_query = random.sample(all_models, min(num_random_models, len(all_models)))
elif test_all_models:
    models_to_query = all_models
else:
    models_to_query = [m for m in all_models if m.model in specific_models_to_test]

print(f"\n--- Models to Query ({len(models_to_query)}) ---")
for m in models_to_query[:3]:  # Display the first 3 models for brevity
    print(f"   {m.model}")

# ==== Multi-query and Synthesis ====
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI assistant. Generate five different versions of the user question
    to retrieve relevant documents from the vector database.
    Original question: {question}""",
)

def synthesise_answers(prompt, check_responses=False):
    print(f"\nScript started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_script_time = time.time()

    combined_responses = []
    for m in models_to_query:
        print(f"\n--- Querying model: {m.model} ---")
        llm = ChatOllama(model=m.model)  # Use different LLMs for each retrieval
        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
        )
        synthesis_prompt = ChatPromptTemplate.from_template(
            "Synthesise the following answers into one coherent response:\n{context}"
        )
        start_time = time.time()
        try:
            query_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | synthesis_prompt
                | llm
                | StrOutputParser()
            )
            response = query_chain.invoke(input=prompt)
            combined_responses.append(f"Response from {m.model}:\n{response}")
            print(response)
        except Exception as e:
            print(f"Error querying {m.model}: {e}")
            combined_responses.append(f"ERROR from {m.model}: {str(e)}")
        print(f"Time: {time.time() - start_time:.2f} seconds")

    final_synthesis = ChatOllama(model="llama3.3:70b-instruct-q8_0").invoke({
        "role": "user",
        "content": f"Synthesise the following answers:\n\n{''.join(combined_responses)}",
    })
    print(f"\nFinal Synthesis:\n{final_synthesis}")

    # Fact-checking step
    if check_responses:
        print("\n--- Fact-checking synthesis ---")
        fact_check_prompt = f"Document: {prompt}\nClaim: {final_synthesis}"
        try:
            fc_res = ollama.chat(
                model=fact_check_model,
                messages=[{"role": "user", "content": fact_check_prompt}],
            )
            print(f"\nFact-check result: {fc_res['message']['content']}")
        except Exception as e:
            print(f"Fact-checking error: {e}")

    print(f"\nTotal execution time: {time.time() - start_script_time:.2f} seconds")


# Run the synthesis with the selected query
synthesise_answers(user_prompt, check_responses=check_responses)
