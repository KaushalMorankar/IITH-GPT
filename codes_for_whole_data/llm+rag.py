import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def load_index_tracker(file_path):
    with open(file_path, 'r') as file:
        index_tracker = json.load(file)
    return index_tracker

def get_query_embedding(query, model):
    query_embedding = model.encode(query)
    return query_embedding

def search_in_faiss(query_embedding, index_file, top_k=5):
    index = faiss.read_index(index_file)
    query_embedding = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    return distances, indices

def get_top_5_sentences_from_one_file(query, index_tracker, model, top_k=5):
    query_embedding = get_query_embedding(query, model)
    most_relevant_doc = None
    best_distance = float('inf')
    best_results = None

    # Iterate over all tracked FAISS indices
    for doc_info in index_tracker:
        index_file = doc_info['index_file']
        metadata_file = doc_info['metadata_file']

        # Search in the current FAISS index
        distances, indices = search_in_faiss(query_embedding, index_file, top_k)

        # Find the document with the minimum distance (most relevant)
        if distances[0][0] < best_distance:
            best_distance = distances[0][0]
            most_relevant_doc = doc_info
            best_results = (distances, indices, metadata_file)
    
    # Retrieve the top-5 sentences from the most relevant document
    if most_relevant_doc and best_results:
        distances, indices, metadata_file = best_results

        # Load metadata to fetch sentences
        with open(metadata_file, 'r') as file:
            metadata = json.load(file)

        top_sentences = [metadata[i] for i in indices[0]]
        print(top_sentences)
        return {
            'document': most_relevant_doc['json_file'],
            'top_sentences': top_sentences,
            'distances': distances[0]
        }
    return None


def convert_to_documents(top_sentences):
    documents = []
    for sentence in top_sentences:
        if isinstance(sentence, dict) and 'content' in sentence:
            content = sentence['content']
        elif isinstance(sentence, str):
            content = sentence
        else:
            raise ValueError(f"Unexpected sentence format: {sentence}")
        
        doc = Document(page_content=content)
        documents.append(doc)
    return documents

def generate_response_with_langchain(documents, query_text, api_key):
    """
    Generates a response using LangChain by integrating retrieved documents into a prompt.
    """
    # Initialize the HuggingFaceHub LLM
    model_name = "google/flan-t5-large"  # Replace with your desired HuggingFace model
    llm = HuggingFaceHub(
        repo_id=model_name,
        huggingfacehub_api_token=api_key,
        model_kwargs={"temperature": 0.7, "max_length": 800}
    )

    # Build the prompt using the retrieved documents
    prompt = f"Given the following content, please answer the user's query in a concise and informative manner:\n\n"
    for idx, doc in enumerate(documents):
        prompt += f"Content {idx + 1}: {doc.page_content}\n\n"

    prompt += f"User's Query: {query_text}\n\n"
    prompt += "Please provide the answer in a clear and detailed response."

    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=["query", "context"],
        template=prompt
    )

    # Create a LangChain LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    # Execute the chain to get the response
    response = llm_chain.run({"query": query_text, "context": prompt})
    return response


def main_pipeline():
    # Configurations
    index_tracker_file = r"D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\index_tracker.json"
    query_text = "NIRF ranking of IITH?"
    top_k = 5
    huggingface_api_key = "YOUR HUGGING FACE API KEY"  # Replace with your API key

    # Load SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load index tracker
    with open(index_tracker_file, 'r') as file:
        index_tracker = json.load(file)

    # Step 1: Retrieve top 5 sentences from the most relevant document
    result = get_top_5_sentences_from_one_file(query_text, index_tracker, model, top_k=top_k)

    if result:
        print(f"Most Relevant Document: {result['document']}")

        # Step 2: Convert top sentences to Document objects
        documents = convert_to_documents(result['top_sentences'])

        # Step 3: Generate a response using LangChain
        final_response = generate_response_with_langchain(documents, query_text, huggingface_api_key)
        print("Generated Response:", final_response)
    else:
        print("No relevant document found.")

# Execute the pipeline
if __name__ == "__main__":
    main_pipeline()