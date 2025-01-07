import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration  # Add this import
from langchain.schema import Document
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Step 1: Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 2: Create a FAISS index and add normalized embeddings
# Load the FAISS index and metadata file
index = faiss.read_index(r'D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\data\document_embeddings.index')

with open(r'D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\data\document_metadata.json', 'r') as file:
    metadata = json.load(file)

# Generate the query embedding
query_text = "What is QS ASIA bricks ranking of IITH?"
query_embedding = model.encode(query_text).reshape(1, -1)  # Reshape for FAISS

# Normalize query embedding for cosine similarity
faiss.normalize_L2(query_embedding)  # Normalize query embedding

# Perform a similarity search in the vector store (FAISS)
k = 3  # Retrieve top 3 closest results
distances, indices = index.search(query_embedding, k)

# Convert distances to similarities
similarities = 1 - (distances / 2)

results = []
for i, idx in enumerate(indices[0]):
    if 0 <= idx < len(metadata):  # Ensure index is valid
        document = metadata[idx]  # Sentence-level metadata
        doc = Document(
            page_content=document['content'],  # Full sentence
            # metadata={'similarity': similarities[0][i]}  # Similarity score as metadata
        )
        results.append(doc)
    else:
        print(f"Invalid index: {idx}")

for result in results:
    # Access content and similarity from the Document object attributes
    print(f"Content: {result.page_content}")
    # print(f"Similarity: {result.metadata['similarity']:.4f}")
    print("-" * 80)
print(results)

# --- LangChain Query Generation ---

# Make sure to replace 'YOUR_HUGGINGFACE_API_KEY' with your actual API key
huggingface_api_key = "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"  # Replace with your Hugging Face API key

# Initialize T5 model from HuggingFaceHub
model_name = "google/flan-t5-large"  # Use FLAN-T5 small model (you can use base or large for better performance)

llm = HuggingFaceHub(
    repo_id=model_name,  # Model name on Hugging Face
    huggingfacehub_api_token=huggingface_api_key,  # Pass your HuggingFace API key here
    model_kwargs={"temperature": 0.7, "max_length": 200}  # Additional model settings
)

prompt = f"Given the following information, please provide the QS ASIA BRICS ranking of IITH in a full sentence based on the content provided:\n\n"

for idx, result in enumerate(results):
    prompt += f"Content {idx+1}: {result.page_content}\n\n"  # Include only content

prompt += f"User's Query: {query_text}\n\n"
prompt += "Please provide the answer as a complete sentence."

# Define a prompt template (you can modify it as needed)
prompt_template = PromptTemplate(
    input_variables=["query", "context"], 
    template=prompt
)

# Pass the prompt template to LangChain's LLMChain for generating the response
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# Execute the chain to get the generated response
final_answer = llm_chain.run({"query": query_text, "context": prompt})

# Print the final generated answer
print("Generated Answer using LangChain:")
print(final_answer)
