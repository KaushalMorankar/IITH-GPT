import faiss
import json
import torch
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from huggingface_hub import login
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from transformers import AutoTokenizer
import torch
from langchain_ollama import OllamaLLM
import os
from dotenv import load_dotenv
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Allow duplicate OpenMP libraries
os.environ["OMP_NUM_THREADS"] = "1"  # Limit to 1 thread to avoid conflicts

BASE_DIR = Path(__file__).resolve().parent.parent  # Moves up to root
dotenv_path = BASE_DIR / ".env"
load_dotenv(dotenv_path)

from tavily import TavilyClient
# from langgraph.checkpoint.memory import MemorySaver
import torch
from utils import classify_query_with_gemini
from prompts import summarize, fact_verification, search, exploration
# from lsa import clustered_rag_lsa, summarize_it
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

class MemorySaver:
    def __init__(self):
        self.memory = {}

    def save_document(self, doc_id, content, metadata=None):
        self.memory[doc_id] = {
            "query": content,
            "response": metadata or {}
        }

    def get_document(self, doc_id):
        return self.memory.get(doc_id, None)

    def exists(self, doc_id):
        return doc_id in self.memory

    def get_all_documents(self):
        return self.memory

    # def remove_document(self, doc_id):
    #     if doc_id in self.memory:
    #         del self.memory[doc_id]

    def to_json(self):
        return json.dumps(self.memory, indent=4)

    def load_from_json(self, json_str):
        self.memory = json.loads(json_str)

# Example usage
memory_store = MemorySaver()

# Tavily search tool setup
tavily_api_key = os.getenv('TAVILY_API_KEY')
hf_token = os.getenv('HF_TOKEN')
tavily = TavilyClient(api_key=tavily_api_key)

if not os.environ.get('AUTOGEN_USE_DOCKER'):
    os.environ['AUTOGEN_USE_DOCKER'] = '0'

# if not os.environ.get('GOOGLE_API_KEY'):
#     os.environ['GOOGLE_API_KEY'] = google_api_key

tavily = TavilyClient(tavily_api_key)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
login(token=hf_token)
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
model = SentenceTransformer('all-MiniLM-L6-v2')
model.to(device)


# Load a tokenizer (for example, BERT tokenizer)
hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

tavily = TavilyClient(tavily_api_key)
search = TavilySearchResults(max_results=2) 
tools = [search]

# agent_executor = create_react_agent(model, tools)
# agent_executor_with_memory = create_react_agent(model, tools, checkpointer=memory)

def execute_task(model, tools, user_query):
    """
    Executes the task by running tools and passing their output to the model.

    Args:
        model: The LLaMA model to generate responses.
        tools: A list of tools to retrieve additional information.
        user_query: The user's input query.

    Returns:
        The model's response.
    """
    tool_results = [tool.run(user_query) for tool in tools]
    
    combined_query = f"User Query: {user_query}\n\nTool Results:\n"
    for i, result in enumerate(tool_results, 1):
        combined_query += f"{i}. {result}\n"
    
    response = model.generate(prompts=[combined_query])
    return response.generations[0][0].text  

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def retrieve_documents_with_memory(index, query_embedding, memory_store, metadata, top_k):
    """
    Retrieve top-k documents using FAISS while incorporating the 5 most recent memory entries.
    
    Args:
        index: FAISS index for document retrieval.
        query_embedding: Embedding of the current query.
        memory_store: MemorySaver instance storing past queries and responses.
        metadata: Metadata containing document references.
        top_k: Number of documents to retrieve.
    
    Returns:
        List of retrieved documents including memory context.
    """
    query_embedding = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_docs = [
        {"score": distances[0][i], "doc": metadata[indices[0][i]]}
        for i in range(len(indices[0]))
    ]
    all_memory = memory_store.get_all_documents()
    recent_memories = list(all_memory.items())[-5:]
    memory_docs = [
        {"score": 0.0, "doc": f"Memory: {doc_id} -> Query: {content['query']}, Response: {content['response']}"}
        for doc_id, content in recent_memories
    ]
    combined_results = memory_docs + retrieved_docs
    combined_results = sorted(combined_results, key=lambda x: x["score"], reverse=True)

    return combined_results[:top_k]


def generate_subqueries(user_query, ollama_model, memory_store):
    # Get the last 5 queries from memory
    memory_keys = list(memory_store.memory.keys())[-5:]  # Get last 5 query IDs
    previous_queries = [memory_store.memory[key]["query"] for key in memory_keys]

    # Format memory context
    memory_context = "\n".join([f"Previous Query: {query}" for query in previous_queries])

    prompt = f"""
    {memory_context}

    Current Query: {user_query}

    You are an expert at understanding and correlating user queries. If the query consists of distinct sub-questions, and a clear distinction is observed, break it down into meaningful and logically separate sub-questions **only if necessary.** Otherwise, retain the query as is. **If the query is already a complete and meaningful statement, return it without changes.** Minor grammatical adjustments are allowed if required.

    Guidelines:
    1. **If the query is already a valid and complete question or statement, return it as is without splitting.**  
    2. **Break down the query only when it contains multiple, distinct parts that can stand alone as sub-questions.**  
    3. **Maintain the original intent and context of the query when creating sub-questions.**  
    4. **Provide only the output without any additional explanations or comments.**
    5. **There may be typos or grammatical mistakes. Fix them as per necessity.**
    6. **If the query appears to reference a previous query, use the provided memory to frame it as a complete and independent question.**

    ### Example 1:  
    **Input:** "What is the QS and NIRF ranking of IITH?"  
    **Output:**  
    - "What is the QS ranking of IITH?"  
    - "What is the NIRF ranking of IITH?"  

    ### Example 2:  
    **Input:** "Summarize about IIT."  
    **Output:**  
    - "Summarize about IIT."  

    ### Example 3:  
    **Input:** "Explain the differences between QS and NIRF rankings."  
    **Output:**  
    - "What are QS rankings?"  
    - "What are NIRF rankings?"  
    - "What are the differences between QS and NIRF rankings?"  

    ### Example 4:  
    **Input:** "Who is Rajesh Kedia?"  
    **Output:**  
    - "Who is Rajesh Kedia?"  

    ### Example 5:  
    **Input:** "What is Lambda IITH?"  
    **Output:**  
    - "What is Lambda IITH?"  

    Query: {user_query}  
    Output:
    """
    # print("Query type: ",classify_query_with_gemini(user_query))

    formatted_prompt = prompt.format(user_query=user_query)
    response = ollama_model.invoke(formatted_prompt)
    
    # Extract sub-questions from the response
    subqueries = response.strip().split("\n")
    # Filter out explanatory text or any unwanted lines
    cleaned_subqueries = [
        subquery.strip() for subquery in subqueries 
        if subquery.strip() and not subquery.startswith(("**", "To answer", "These", "The next"))
    ]

    if len(cleaned_subqueries) == 0 or (len(cleaned_subqueries) == 1 and cleaned_subqueries[0] == user_query.strip()):
        return [user_query.strip()]

    return cleaned_subqueries

def load_llama_model(model_name="llama3.1",device=device):
    """Initialize Ollama and load LLaMA model locally."""
    return OllamaLLM(model=model_name, device="cuda" if torch.cuda.is_available() else "cpu")


def validate_relevance_llmresp(subquery, final_resp, llm):
    """
    Validates the relevance of retrieved documents to the subquery using the LLM.

    Args:
        subquery: The subquery to validate against.
        retrieved_docs: List of retrieved documents with scores.
        llm: The LLM used for relevance validation.

    Returns:
        Boolean indicating whether any of the documents are relevant.
    """
    prompt = (
        f"Query: {subquery}\n\n"
        f"Response Generated:\n{final_resp}\n\n"
        "Task:\n"
        "Evaluate if the response generated is relevant to the query and answers it in a satisfactory manner. Respond with:\n"
        "- 'Relevant' if it matches the query.\n"
        "- 'Not Relevant' if it does not match the query.\n"
        "Only provide the response, no additional text."
    )
    response = llm.invoke(prompt).strip()
    if response.lower() != "not relevant":
        return True
    return False



def validate_relevance(subquery, retrieved_docs, llm):
    """
    Validates the relevance of retrieved documents to the subquery using the LLM.

    Args:
        subquery: The subquery to validate against.
        retrieved_docs: List of retrieved documents with scores.
        llm: The LLM used for relevance validation.

    Returns:
        Boolean indicating whether any of the documents are relevant.
    """
    for doc in retrieved_docs:
        prompt = (
            f"Query: {subquery}\n\n"
            f"Retrieved Document:\n{doc['doc']}\n\n"
            "Task:\n"
            "Evaluate if the retrieved documents are relevant to the query, and if the query can be satisfactorily answered using information from the documents. Respond with:\n"
            "- 'Relevant' if yes.\n"
            "- 'Not Relevant' if the query cannot be answered in a satisfactory manner.\n"
            "Only provide the response, no additional text."
        )
        response = llm.invoke(prompt).strip()
        if response.lower() == "relevant":
            return True
    return False


def process_query_with_validation(query, index, metadata, ollama_model, embedder, tavily, top_k=5):
    """
    Process the query by retrieving documents, validating their relevance, and falling back to Tavily if needed.

    Args:
        query: User's query.
        index: FAISS index for retrieval.
        metadata: Metadata associated with the FAISS index.
        ollama_model: LLaMA model for subqueries and response generation.
        embedder: Embedding model to encode queries.
        tavily: Tavily client for fallback search.
        top_k: Number of documents to retrieve initially.

    Returns:
        Final response generated by the LLM.
    """

    # Check if the main query or any subqueries already exist in memory
    subqueries = generate_subqueries(query, ollama_model,memory_store)
    # print("Subqueries:", subqueries)
    retrieved_context = []

    for subquery in subqueries:
        print(type(subquery))
        query_embedding = embedder.encode([subquery], device=device).flatten()
        query_type = classify_query_with_gemini(subquery)
        if query_type == "summarization":
            # Retrieve more documents for comprehensive coverage
            top_k = 20
        elif query_type == "question_answering":
            # Focus on precise and concise documents
            top_k = 5
        elif query_type == "search":
            # Balance between precision and coverage
            top_k = 10
        elif query_type == "fact_verification":
            # Retrieve documents explicitly supporting or refuting the fact
            top_k = 8
        elif query_type == "exploration":
            # Retrieve a wide variety of documents for broader coverage
            top_k = 15
        # elif query_type == "comparison":
        #     # Retrieve a wide variety of documents for broader coverage
        #     top_k = 10
        else:
            # Fallback logic for unknown query types
            top_k = 5
        retrieved_docs = retrieve_documents_with_memory(index, query_embedding,memory_store, metadata, top_k)
        is_relevant = validate_relevance(subquery, retrieved_docs, ollama_model)
        # print(retrieved_docs)
        # Fallback logic
        if not is_relevant:
            print(f"Subquery '{subquery}' documents not relevant, retrying...")
            
            retrieved_docs = retrieve_documents_with_memory(index, query_embedding,memory_store, metadata, top_k * 2)
            is_relevant = validate_relevance(subquery, retrieved_docs, ollama_model)
            # # Final fallback to Tavily if still not relevant
            # if not is_relevant:
            #     print(f"Subquery '{subquery}' failed validation. Performing Tavily search...")
            #     tavily_search = TavilySearchResults(max_results=2)
            #     retrieved_docs = [{"doc": result} for result in tavily_search.run(subquery)]

        retrieved_context.append({
            "subquery": subquery,
            "context": retrieved_docs
        })

    combined_context = "\n".join(
        f"Subquery: {item['subquery']}\nContext: {item['context']}"
        for item in retrieved_context
    )

    prompt = (
        "You are an intelligent AI assistant. You will be provided with a user query divided into subqueries and a context. Your task is to generate proper response for the subqueries provided with the context(documents retrived) and answer as a whole together.\n\n"
        "Remember, the context is set in the domain of Indian Institute of Technology Hyderabad (IITH).\n\n"
        f"Original Query: {query}\n\n"
        f"Subqueries: {', '.join(subqueries)}\n\n"
        f"Context:\n{combined_context}\n\n"
        "Guidelines:\n"
        "1. Carefully analyse both the query and the context that has been provided to you. Do not lose relevant information while generating the response.\n"
        "2. Provide direct and concise answers while combining all relevant aspects related to the query.\n"
        "3. Maintain a professional tone and do not mention the reasonings behind which you chose the content, only the content itself in a informative manner.\n"
        "4. Address each subquery comprehensively without omitting details.\n"
        "5. Provide precise and accurate responses, ensuring that only relevant information directly related to the query is included.\n"
    )
    if query_type == "summarization":
        final_response = ollama_model.invoke(summarize.format(user_query=query, context=combined_context, subqueries=subqueries))
    elif query_type == "question_answering":
        # print("question answer query detected. Truncating context for exploration...")
        # truncated_context = clustered_rag_lsa(embedder, combined_context, num_clusters=5, sentences_count=5)
        final_response = ollama_model.invoke(exploration.format(user_query=query, context=combined_context, subqueries=subqueries))
    elif query_type == "search":
        # print("Search query detected. Truncating context for exploration...")
        # truncated_context = clustered_rag_lsa(embedder, combined_context, num_clusters=5, sentences_count=5)
        final_response = ollama_model.invoke(exploration.format(user_query=query, context=combined_context, subqueries=subqueries))
    elif query_type == "fact_verification":
        final_response = ollama_model.invoke(fact_verification.format(user_query=query, context=combined_context, subqueries=subqueries))
    elif query_type == "exploration":
        # print("Exploration query detected. Truncating context for exploration...")
        # truncated_context = clustered_rag_lsa(embedder, combined_context, num_clusters=5, sentences_count=5)
        final_response = ollama_model.invoke(exploration.format(user_query=query, context=combined_context, subqueries=subqueries))
    else:
        final_response = ollama_model.invoke(prompt)
    is_relevant_llmresp = validate_relevance_llmresp(query, final_response, ollama_model)
    retry_count = 0
    max_retries = 3
    while not is_relevant_llmresp and retry_count < max_retries:
        print(f"Final response not relevant, retrying... Attempt {retry_count + 1}")
        final_response = ollama_model.invoke(prompt)
        is_relevant_llmresp = validate_relevance_llmresp(query, final_response, ollama_model)
        retry_count += 1
    memory_store.save_document(query, final_response)
    return final_response.strip()
    # query_embedding = embedder.encode([query], device=device).flatten()
    # retrieved_docs = retrieve_documents_with_memory(index, query_embedding, metadata, top_k=5)

    # # # Step 2: Extract document content for clustering and summarization
    # # context_docs = [doc['doc']['content'] for doc in retrieved_docs]
    # # final_context = "\n".join(context_docs)
    
    # # Step 4: Generate the final response using the summarized context
    # prompt = (
    #     "You are tasked with answering the following query based on the provided context, set in the domain of Indian Institute of Technology Hyderabad (IITH):\n\n"
    #     "Query: {query}\n\n"
    #     "Context:\n{combined_content}\n\n"
    #     "Guidelines:\n"
    #     "1. Provide a direct answer providing the exact details asked in the query.\n"
    #     "2. Use a professional tone.\n"
    #     "3. Address all relevant aspects of the query.\n"
    #     "4. The title_1 is a field which states who is the person or what it signifies. Use that information to know if a person is associated with it and include its relevance in the answer"
    # )
    # formatted_prompt = prompt.format(
    #     query=query,
    #     combined_content=retrieved_docs  # Use 'combined_content' here to match the placeholder
    # )
    # response = ollama_model.invoke(formatted_prompt)  # Use the model object directly as it should be callable
    
    # # Assuming 'response' is a string-like object
    # return response.strip()

index_path = r'D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\output_multilevel_index\faiss_index.index'
metadata_path = r'D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\output_multilevel_index\metadata.json'

index = load_faiss_index(index_path)
with open(metadata_path, "r", encoding="utf-8") as metadata_file:
    metadata = json.load(metadata_file)

ollama_model = load_llama_model(model_name="llama3.1",device=device)  # Replace with LLaMA 3.1 model if different


while True:
    user_query = input("Enter your query: ")
    if user_query.lower() == "exit":
        break

    final_response = process_query_with_validation(
        query=user_query,
        index=index,
        metadata=metadata,
        ollama_model=ollama_model,
        embedder=model,  
        tavily=None
    )
    print("\nFinal Response:")
    print(final_response)