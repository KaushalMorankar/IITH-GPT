import faiss
import json
import torch
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from huggingface_hub import login
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer
from google_search import search_tool,format_search
import os
from tavily import TavilyClient
import torch
from utilis import classify_query

# Constants/Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LANGUAGE = "english"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("No CUDA-compatible GPU detected.")
# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
model.to(device)
# Load FAISS index
def load_faiss_index(index_path):
    return faiss.read_index(index_path)

# Retrieve documents from FAISS
def retrieve_documents_from_faiss(index, query_embedding, metadata, top_k=5):
    """Retrieve top-k documents using FAISS."""
    query_embedding = query_embedding.reshape(1, -1) 
    distances, indices = index.search(query_embedding, top_k)
    # if distances[0][0] > 0.7:
    #     web_results = search_tool(user_query,tavily)
    #     # print(user_query)
    #     formatted_results = format_search(web_results)
    #     return formatted_results
    retrieved_docs = [
        {"score": distances[0][i], "doc": metadata[indices[0][i]]}
        for i in range(len(indices[0]))
    ]
    # print(retrieved_docs)
    return retrieved_docs

def generate_subqueries(user_query, ollama_model):
    prompt = (
        """Query: {user_query}\n\n
        You are an expert at understanding and decomposing user queries into distinct sub-questions. 
        Your goal is to break down the query into meaningful and logically separate sub-questions.
        
        Guidelines:
        1. Do not split names or single entities unnecessarily.
        2. Maintain the context of the original question in the sub-questions.
        3. Ensure the sub-questions are simple, clear, and directly related to the query.
        4. Provide only the sub-questions without any additional text or explanations.
        
        ### Example 1:
        Input: "What is the QS and NIRF ranking of IITH?"
        Output:
        - "What is the QS ranking of IITH?"
        - "What is the NIRF ranking of IITH?"
        
        ### Example 2:
        Input: "Who is Narendra and Amit?"
        Output:
        - "Who is Narendra?"
        - "Who is Amit?"
        
        ### Example 3:
        Input: "Explain the differences between QS and NIRF rankings."
        Output:
        - "What are QS rankings?"
        - "What are NIRF rankings?"
        - "What are the differences between QS and NIRF rankings?"
        
        ### Example 4:
        Input: "Summarize the contributions of Mahatma Gandhi and Jawaharlal Nehru."
        Output:
        - "What are the contributions of Mahatma Gandhi?"
        - "What are the contributions of Jawaharlal Nehru?"
        
        Query: {user_query}
        Output:"""
    )

    formatted_prompt = prompt.format(user_query=user_query)
    response = ollama_model.invoke(formatted_prompt)
    
    # Extract sub-questions from the response
    subqueries = response.strip().split("\n")
    # Filter out explanatory text or any unwanted lines
    cleaned_subqueries = [
        subquery.strip() for subquery in subqueries 
        if subquery.strip() and not subquery.startswith(("**", "To answer", "These", "The next"))
    ]

    # If there's only one query or decomposition isn't meaningful, return the original query
    if len(cleaned_subqueries) == 0 or (len(cleaned_subqueries) == 1 and cleaned_subqueries[0] == user_query.strip()):
        return [user_query.strip()]

    return cleaned_subqueries

def load_llama_model(model_name="llama3.1"):
    """Initialize Ollama and load LLaMA model locally."""
    return OllamaLLM(model="llama3.1", device="cuda")  

# Main processing function
def process_query(query, index, metadata, ollama_model, embedder, num_clusters=5, sentences_count=5):
    subqueries = generate_subqueries(query, ollama_model)
    # print(subqueries)
    # llm=pipeline("text-analysis", model="gpt2", device=0)
    # query_type=classify_query(llm,query,"nope")
    # print(query_type)
    # Collect retrieved content for subqueries
    retrieved_context = []
    for subquery in subqueries:
        
        query_embedding = embedder.encode([subquery], device=device).flatten()
        retrieved_docs = retrieve_documents_from_faiss(index, query_embedding, metadata, top_k=5)
        print(retrieved_docs)

        # Append the entire retrieved_docs structure as context
        retrieved_context.append({
            "subquery": subquery,
            "context": retrieved_docs  # Keep the documents as-is
        })

    print("\n")
    # Combine retrieved contexts
    # We pass the retrieved_context as structured JSON-like data to the LLM
    combined_context = "\n".join(
        f"Subquery: {item['subquery']}\nContext: {item['context']}"
        for item in retrieved_context
    )
    print(combined_context)
    # Generate final response using LLM
    prompt = (
        "You are tasked with answering the following subqueries based on the provided context, set in the domain of Indian Institute of Technology Hyderabad (IITH):\n\n"
        "Query: {query}\n\n"
        "Context:\n{combined_context}\n\n"
        "Guidelines:\n"
        "1. Provide a direct answer providing the exact details asked in the query.\n"
        "2. Use a professional tone.\n"
        "3. Address all relevant aspects of the query.\n"
    )
    formatted_prompt = prompt.format(query=query, combined_context=combined_context)
    final_response = ollama_model.invoke(formatted_prompt)
    return final_response.strip()

# Paths
index_path = r"D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\output_multilevel_index\faiss_index.index"
metadata_path = r"D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\output_multilevel_index\metadata.json"

# Load FAISS index and metadata
index = load_faiss_index(index_path)
with open(metadata_path, "r", encoding="utf-8") as metadata_file:
    metadata = json.load(metadata_file)

# Load LLaMA model from Ollama
ollama_model = load_llama_model(model_name="llama3.1")  # Replace with LLaMA 3.1 model if different

while True:
    user_query = input("Enter your query: ")
    if user_query.lower() == "exit":
        break
    final_response = process_query(user_query, index, metadata, ollama_model, model)
    print("\nFinal Response:")
    print(final_response)