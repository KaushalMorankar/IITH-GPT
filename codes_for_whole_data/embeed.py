import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Step 1: Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 2: Function to handle hierarchical titles and content
def process_hierarchical_data_with_splitting(documents, chunk_size=200, chunk_overlap=50):
    """
    Process JSON data with hierarchical titles and content and split into chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, 
        separators=["\n\n", ". ", " "]
    )
    
    sentences = []
    stack = []  # To keep track of the current hierarchy

    for doc in documents:
        current_title = doc.get('title', '').strip()

        if not current_title:  # Skip if there's no title
            continue

        # Add the current title to the hierarchy stack
        while stack and stack[-1]['content'] is not None:
            stack.pop()  # Move up in the hierarchy if the stack is outdated
        stack.append({'title': current_title, 'content': doc.get('content')})

        # Combine titles in the stack to create a hierarchical prefix
        title_context = " > ".join(item['title'] for item in stack if item['title'])

        # Process the content
        content = doc.get('content', [])
        if not content:  # If content is empty, continue processing the stack
            continue

        # Handle textual content
        for paragraph in content:
            if isinstance(paragraph, str):  # If plain text
                combined_text = f"{title_context} - {paragraph.strip()}"
                chunks = splitter.split_text(combined_text)
                sentences.extend(chunks)
            elif isinstance(paragraph, dict):  # If nested structure
                nested_sentences = convert_nested_to_sentence(paragraph, f"{title_context} -")
                for nested in nested_sentences:
                    chunks = splitter.split_text(nested)
                    sentences.extend(chunks)

    return sentences

# Step 3: Function to handle nested dictionaries and lists
def convert_nested_to_sentence(data, prefix=""):
    """
    Recursively convert nested dictionaries and lists into meaningful sentences.
    """
    sentences = []
    if isinstance(data, dict):  # If the data is a dictionary
        for key, value in data.items():
            new_prefix = f"{prefix} {key}".strip()
            sentences.extend(convert_nested_to_sentence(value, new_prefix))
    elif isinstance(data, list):  # If the data is a list
        for item in data:
            sentences.extend(convert_nested_to_sentence(item, prefix))
    else:  # Base case: primitive type
        sentences.append(f"{prefix} is {data}".strip())
    return sentences

# Step 4: Process JSON files in the specified directory
directory_path = r'D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\data'
output_directory = r'D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\output'

# Create the output directory if it does not exist
os.makedirs(output_directory, exist_ok=True)

for file_name in os.listdir(directory_path):
    if file_name.endswith('.json'):
        # Step 5: Load the JSON data
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            try:
                documents = json.load(file)
            except json.JSONDecodeError as e:
                print(f"Error loading {file_name}: {e}")
                continue  # Skip the problematic file

        # Step 6: Process the documents
        sentences = process_hierarchical_data_with_splitting(documents)

        embeddings = []
        metadata = []

        for sentence in sentences:
            # Generate embedding for each sentence
            embedding = model.encode(sentence)
            if embedding is not None and len(embedding) > 0:
                embeddings.append(embedding)
                # Store metadata
                metadata.append({'content': sentence})
            else:
                print(f"Warning: Empty embedding for sentence: {sentence}")

        # Step 7: Convert embeddings to numpy array and normalize
        if embeddings:  # Only process if there are embeddings
            embeddings_np = np.array(embeddings, dtype=np.float32)
            print(f"Shape of embeddings for {file_name}: {embeddings_np.shape}")
            
            if embeddings_np.shape[0] > 0:  # Ensure there are embeddings to normalize
                faiss.normalize_L2(embeddings_np)  # Normalize for cosine similarity
            else:
                print(f"Warning: No valid embeddings found in '{file_name}'")
        else:
            print(f"Warning: No embeddings found for '{file_name}'")

        # Step 8: Create FAISS index and save
        if embeddings_np.shape[0] > 0:  # Only create index if there are embeddings
            index = faiss.IndexFlatL2(embeddings_np.shape[1])
            index.add(embeddings_np)

            # Save the index
            index_file_name = f"{os.path.splitext(file_name)[0]}_embeddings.index"
            faiss.write_index(index, os.path.join(output_directory, index_file_name))

            # Save the metadata
            metadata_file_name = f"{os.path.splitext(file_name)[0]}_metadata.json"
            with open(os.path.join(output_directory, metadata_file_name), 'w') as f:
                json.dump(metadata, f)

            print(f"Embeddings and metadata for '{file_name}' created successfully!")
        else:
            print(f"Skipping index creation for '{file_name}' due to no valid embeddings.")