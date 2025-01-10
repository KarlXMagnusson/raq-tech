#!/usr/bin/env python3

import os
import subprocess
import shlex
from typing import Optional
import pandas as pd

# NOTE: We assume you have appropriate local packages that provide the classes LocalEmbeddingLLM and LocalChatLLM.
#       You may need to adjust imports according to your local environment and library structure.

#
# 1. Token-Based Splitting and Local LLM Prompting
#

#@cond INTERNAL
# If these imports do not exist, adapt them to your local text splitter and chain libraries.
from langchain_text_splitters import TokenTextSplitter

# This import references hypothetical modules for local LLM usage. Adjust them as needed for your environment.
from local_llm_clients import LocalChatLLM, LocalEmbeddingLLM

# A prompt and an output parser for constructing the RAG chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#@endcond

def main():
    """
    @brief The main function orchestrates the end-to-end process:
     1) Reads a text file (A_Model_of_Design_for_Computing_Systems_A_Categorical_Approach.txt)
     2) Splits it into tokens
     3) Applies an entity-relationship prompt to one of the chunks
     4) Retrieves data from local Parquet outputs
     5) Demonstrates calling a local GraphRAG CLI
     6) Demonstrates storing and retrieving embeddings from ChromaDB using a local embedding model
     7) Performs a final RAG query chain with a local generative model
    """

    # Step 1: Read the text file
    input_path = "./ragtest/input/A_Model_of_Design_for_Computing_Systems_A_Categorical_Approach.txt"
    with open(input_path, 'r') as file:
        content = file.read()

    # Step 2: Split text into tokens
    text_splitter = TokenTextSplitter(chunk_size=1200, chunk_overlap=100)
    texts = text_splitter.split_text(content)

    # Step 3: Construct the prompt template
    prompt_template = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalised
- entity_type: One of the following types: [large language model, differential privacy, federated learning, healthcare, adversarial training, security measures, open-source tool, dataset, learning rate, AdaGrad, RMSprop, adapter architecture, LoRA, API, model support, evaluation metrics, deployment, Python library, hardware accelerators, hyperparameters, data preprocessing, data imbalance, GPU-based deployment, distributed inference]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as (\"entity\"{{tuple_delimiter}}<entity_name>{{tuple_delimiter}}<entity_type>{{tuple_delimiter}}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: an integer score between 1 to 10, indicating strength of the relationship between the source entity and target entity
Format each relationship as (\"relationship\"{{tuple_delimiter}}<source_entity>{{tuple_delimiter}}<target_entity>{{tuple_delimiter}}<relationship_description>{{tuple_delimiter}}<relationship_strength>)

3. Return output in The primary language of the provided text is \"English.\" as a single list of all the entities and relationships identified in steps 1 and 2. Use **{{record_delimiter}}** as the list delimiter.

4. If you have to translate into The primary language of the provided text is \"English.\", just translate the descriptions, nothing else!

5. When finished, output {{completion_delimiter}}.

-Examples-
######################

Example 1:
[... truncated example for brevity in code snippet ...]

Example 2:
[... truncated example for brevity in code snippet ...]

-Real Data-
######################
entity_types: [large language model, differential privacy, federated learning, healthcare, adversarial training, security measures, open-source tool, dataset, learning rate, AdaGrad, RMSprop, adapter architecture, LoRA, API, model support, evaluation metrics, deployment, Python library, hardware accelerators, hyperparameters, data preprocessing, data imbalance, GPU-based deployment, distributed inference]
text: {input_text}
######################
output:
"""

    # Step 4: Prepare local generative LLM (instead of ChatOpenAI)
    # We use phi4:14b-fp16 for generation
    local_llm = LocalChatLLM(
        temperature=0.0,
        model="phi4:14b-fp16"  # Use the local generative LLM
    )

    # Step 5: Build the chain using the ChatPromptTemplate and StrOutputParser
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | local_llm | StrOutputParser()

    # Step 6: Run the chain on one chunk of text (e.g., texts[25] if it exists)
    if len(texts) > 25:
        response = chain.invoke({"input_text": texts[25]})
        print(response)
    else:
        print("Insufficient chunks; cannot query texts[25].")

    # Step 7: Demonstrate reading previously stored Parquet files
    # (These may or may not exist in your environment; adapt to your scenario)
    try:
        entities = pd.read_parquet('./ragtest/output/create_final_entities.parquet')
        relationships = pd.read_parquet('./ragtest/output/create_final_relationships.parquet')
        nodes = pd.read_parquet('./ragtest/output/create_final_nodes.parquet')
        community_reports = pd.read_parquet('./ragtest/output/create_final_community_reports.parquet')

        print(entities.head())
        print(relationships.head())
        print(nodes.head(10))

        print(community_reports["full_content"][0])
        print(community_reports["summary"][0])
    except FileNotFoundError:
        print("Some Parquet files not found. Skipping those demonstrations.")

    #
    # 8. GraphRAG CLI Query Function
    #

    def query_graphrag(
        query: str,
        method: str = "global",
        root_path: str = "./ragtest",
        timeout: Optional[int] = None,
        community_level: int = 2,
        dynamic_community_selection: bool = False
    ) -> str:
        """
        @brief Execute a GraphRAG query using the CLI tool.
        @details
         This function demonstrates a local call to GraphRAG CLI, using method, query, and other parameters.
         In case of errors, it raises exceptions with verbose messages.
        @param query The query string to process
        @param method One of "global", "local", or "drift" for the query approach
        @param root_path Path to the root directory where GraphRAG resources exist
        @param timeout Timeout in seconds for the command
        @param community_level The community level in the Leiden community hierarchy (default=2)
        @param dynamic_community_selection Whether or not to use global search with dynamic community selection
        @return The output from GraphRAG as a string
        @throws subprocess.CalledProcessError if the command fails
        @throws subprocess.TimeoutExpired if the command times out
        @throws ValueError if community_level is negative
        """

        if community_level < 0:
            raise ValueError("Community level must be non-negative")

        # Construct the base command
        command = [
            'graphrag', 'query',
            '--root', root_path,
            '--method', method,
            '--query', query,
            '--community-level', str(community_level)
        ]

        # Add dynamic community selection flag if enabled
        if dynamic_community_selection:
            command.append('--dynamic-community-selection')

        # Execute the command and capture output
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            result.check_returncode()
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            error_message = f"Command failed with exit code {e.returncode}\nError: {e.stderr}"
            raise subprocess.CalledProcessError(
                e.returncode,
                e.cmd,
                output=e.output,
                stderr=error_message
            )

    # 9. Execute some example GraphRAG queries
    try:
        result = query_graphrag(
            query="How does a company choose between RAG, fine-tuning, and different PEFT approaches?",
            method="local"
        )
        print("Query result (local method):")
        print(result)

        result = query_graphrag(
            query="How does a company choose between RAG, fine-tuning, and different PEFT approaches?",
            method="global"
        )
        print("Query result (global method):")
        print(result)

        result = query_graphrag(
            query="How does a company choose between RAG, fine-tuning, and different PEFT approaches?",
            method="drift"
        )
        print("Query result (drift method):")
        print(result)

    except FileNotFoundError:
        print("GraphRAG CLI not found or ragtest folder missing. Skipping GraphRAG queries.")

    #
    # 10. ChromaDB for Local Embeddings
    #
    try:
        import chromadb

        # @brief This demonstrates how to instantiate a local Chroma client
        #        and store embeddings from a local embedding model.
        chroma_client = chromadb.PersistentClient(path="./notebook/chromadb")

        # Create or get a collection
        paper_collection = chroma_client.get_or_create_collection(name="paper_collection")

        # Step 10a: Use local embedding model (snowflake-arctic-embed:latest)
        local_embedding_model = LocalEmbeddingLLM(model="snowflake-arctic-embed:latest")

        # We demonstrate the addition of chunks to our ChromaDB instance with local embeddings
        i = 0
        for text in texts:
            # We retrieve embedding from the local model (abstracted away)
            embedding_vector = local_embedding_model.embed(text)
            # Insert into ChromaDB
            paper_collection.add(
                documents=[text],
                embeddings=[embedding_vector],
                ids=[f"chunk_{i}"]
            )
            i += 1

        # Step 10b: Retrieval function from Chroma using local embeddings
        def chroma_retrieval(query, num_results=5):
            """
            @brief Retrieve the top-N similar documents from ChromaDB.
            @details
             The function queries the local ChromaDB collection using a local embedding model
             to find the documents with the highest similarity.
            @param query The query text
            @param num_results The number of top results to return
            @return A dictionary containing the relevant documents
            """
            # We embed the query with the same local embedding model
            query_emb = local_embedding_model.embed(query)
            results = paper_collection.query(
                query_embeddings=[query_emb],
                n_results=num_results
            )
            return results

        #
        # 11. RAG Prompt & Chain with Local Generative Model
        #
        rag_prompt_template = """
Generate a response of the target length and format that responds to the user's question, summarising all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

Context: {retrieved_docs}

User Question: {query}
"""
        rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)
        rag_chain = rag_prompt | local_llm | StrOutputParser()

        def chroma_rag(query):
            """
            @brief Performs a Retrieval-Augmented Generation with local embeddings and local generation.
            @details
             1) Retrieve top documents from Chroma using local embeddings.
             2) Pass the retrieved docs + user query to the local generative model for an answer.
            @param query The user query to answer
            @return The local LLM's response with references to the best supporting evidence
            """
            retrieved = chroma_retrieval(query)["documents"][0]
            response = rag_chain.invoke({"retrieved_docs": retrieved, "query": query})
            return response

        # Step 11a: Try a sample query
        sample_question = "How does a company choose between RAG, fine-tuning, and different PEFT approaches?"
        rag_response = chroma_rag(sample_question)
        print("\nFinal RAG Response:\n", rag_response)

    except ImportError:
        print("ChromaDB library or local embedding support not available. Skipping embedding steps.")

if __name__ == "__main__":
    main()

