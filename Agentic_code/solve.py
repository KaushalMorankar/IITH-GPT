import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import pdfplumber
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# Initialize the model once globally
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 1: Extract text from PDF using pdfplumber
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""  # If extraction fails, we append an empty string
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

# Step 2: Preprocess the extracted text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Clean up whitespace
    return text.strip()

# Step 3: Split text into chunks (e.g., by paragraphs or set number of words)
def chunk_text(text, chunk_size=500):  # Adjust chunk_size as needed
    sentences = [sentence for sentence in text.split("\n") if sentence]
    chunks = []
    current_chunk = []

    for sentence in sentences:
        # If the chunk is too big, push the current chunk and start a new one
        if len(" ".join(current_chunk).split()) + len(sentence.split()) > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]  # Start a new chunk with the current sentence
        else:
            current_chunk.append(sentence)

    # Append the last chunk if any sentences are remaining
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

# Step 4: Generate embeddings for the extracted text using Sentence-Transformers
def generate_embeddings(text, model, chunk_size=500):
    chunks = chunk_text(text, chunk_size)  # Split the text into chunks
    embeddings = model.encode(chunks, show_progress_bar=True)  # added show_progress_bar for user feedback
    return chunks, np.array(embeddings)

# Step 5: Perform similarity search using FAISS
def search_with_faiss(query, embeddings, chunks, model, k=3):
    query_embedding = model.encode([query])  # Embedding of the query
    
    # Initialize FAISS index and add document embeddings
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance for similarity search
    index.add(embeddings)  # Add document embeddings to the FAISS index
    
    # Search for the most similar chunks
    D, I = index.search(query_embedding, k)  # D -> distances, I -> indices of nearest neighbors
    
    # Get the corresponding chunks
    result = [chunks[i] for i in I[0]]
    return result

# Step 6: Summarize the retrieved content using Hugging Face summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text_hf(text):
    try:
        summary = summarizer(text, max_length=200, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return "Could not summarize the text."

# Step 7: Run the entire workflow
def process_pdf_for_query(pdf_path, user_query, chunk_size=500):
    # Step 1: Extract and preprocess the text from the PDF
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return "Error: No text extracted from the PDF."
    
    preprocessed_text = preprocess_text(text)
    
    # Step 2: Generate embeddings for the text
    chunks, embeddings = generate_embeddings(preprocessed_text, model, chunk_size)
    
    # Step 3: Perform the search to find relevant chunks based on the query
    relevant_chunks = search_with_faiss(user_query, embeddings, chunks, model, k=3)
    
    # Step 4: Summarize the relevant chunks or return as is
    combined_relevant_text = " ".join(relevant_chunks)
    summary = summarize_text_hf(combined_relevant_text)
    
    return summary

# Example usage
pdf_path = r'D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\data\Academic_Handbook_2022_(50th_senate).pdf'
user_query = 'Give information about the courses offered in the academic handbook.'

summary = process_pdf_for_query(pdf_path, user_query, chunk_size=500)
print(summary)
