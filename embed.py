import faiss
import json
import os
import torch
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('all-MiniLM-L6-v2')
model.to(device)

# Initialize the FAISS index
def initialize_faiss_index(dim):
    index = faiss.IndexFlatL2(dim)  # L2 distance-based index
    return index

def processjson(json_file, index, metadata):
    with open(json_file, 'r', encoding='utf-8', errors='ignore') as file:
        print(file)
        data = json.load(file)
    
    for entry in data:
        title = entry.get('title', '')
        content = ' '.join(entry.get('content', []))
        
        name_without_extension = os.path.splitext(os.path.basename(json_file))[0]

        # Get title and content embeddings
        title_embedding = model.encode([title], device=device)
        content_embedding = model.encode([content], device=device)
        file_embedding = model.encode(name_without_extension)

        # Convert embeddings to torch tensors
        title_embedding = torch.tensor(title_embedding).to(device)
        content_embedding = torch.tensor(content_embedding).to(device)
        file_embedding = torch.tensor(file_embedding).to(device)
        
        # Split content into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
        sections = text_splitter.split_text(content)
        
        section_embeddings = []
        for section in sections:
            section_embedding = model.encode([section], device=device)
            section_embedding = torch.tensor(section_embedding).to(device)
            section_embeddings.append(section_embedding.cpu().numpy())  # Move tensor to CPU and convert to numpy
        
        # Flatten title embedding and section embeddings, then combine them
        title_embedding_flat = title_embedding.flatten().cpu().numpy()
        section_embeddings_flat = [embedding.flatten() for embedding in section_embeddings]
        
        # Combine title and sections' embeddings
        combined_embedding = (
            1.2 * file_embedding.cpu().numpy() +  # Convert tensor to NumPy array
            1.2 * title_embedding_flat +  # Already a NumPy array
            1.1 * sum(section_embeddings_flat)  # Already a list of NumPy arrays
        ) / (2.4 + 1.1 * len(section_embeddings_flat))


        # Add the combined embedding to the FAISS index
        index.add(combined_embedding.reshape(1, -1))  # Add the embedding as a 2D array (1 x dim)
        
        # Add metadata for this embedding
        metadata.append({
            "title_1":name_without_extension,
            "title": title,
            "sections": sections
        })

# Define paths
directory_path = r'D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\data'
output_directory = r'D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\output_multilevel_index'

# Create output directory if not exists
os.makedirs(output_directory, exist_ok=True)

# Initialize the FAISS index (dimension of 'all-MiniLM-L6-v2' embeddings is 384)
index = initialize_faiss_index(384)  # The dimension of the embeddings is 384
metadata = []

# Process all JSON files in the directory
for file_name in os.listdir(directory_path):
    if file_name.endswith('.json'):
        file_path = os.path.join(directory_path, file_name)
        
        # Process the file and add embeddings to FAISS index
        processjson(file_path, index, metadata)
        print(f"Processed {file_name} and embeddings added to FAISS index.")

# Save the FAISS index to disk
index_file_path = os.path.join(output_directory, 'faiss_index.index')
faiss.write_index(index, index_file_path)
print(f"FAISS index saved to {index_file_path}")

# Save the metadata to a JSON file
metadata_file_path = os.path.join(output_directory, 'metadata.json')
with open(metadata_file_path, 'w', encoding='utf-8') as metadata_file:
    json.dump(metadata, metadata_file, ensure_ascii=False, indent=4)
print(f"Metadata saved to {metadata_file_path}")
