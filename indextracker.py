import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# def convert_nested_to_sentence(data, prefix=""):
#     """
#     Recursively convert nested dictionaries and lists into meaningful sentences.
#     """
#     sentences = []
#     if isinstance(data, dict):  # If the data is a dictionary
#         for key, value in data.items():
#             new_prefix = f"{prefix} {key}".strip()
#             sentences.extend(convert_nested_to_sentence(value, new_prefix))
#     elif isinstance(data, list):  # If the data is a list
#         for item in data:
#             sentences.extend(convert_nested_to_sentence(item, prefix))
#     else:  # Base case: primitive type
#         sentences.append(f"{prefix} is {data}".strip())
#     return sentences

# def process_hierarchical_data_with_splitting(documents, chunk_size=200, chunk_overlap=50):
#     """
#     Process JSON data with hierarchical titles and content and split into chunks.
#     """
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size, chunk_overlap=chunk_overlap, 
#         separators=["\n\n", ". ", " "]
#     )
    
#     sentences = []
#     stack = []  # To keep track of the current hierarchy

#     for doc in documents:
#         current_title = doc.get('title', '').strip()

#         if not current_title:  # Skip if there's no title
#             continue

#         # Add the current title to the hierarchy stack
#         while stack and stack[-1]['content'] is not None:
#             stack.pop()  # Move up in the hierarchy if the stack is outdated
#         stack.append({'title': current_title, 'content': doc.get('content')})

#         # Combine titles in the stack to create a hierarchical prefix
#         title_context = " > ".join(item['title'] for item in stack if item['title'])

#         # Process the content
#         content = doc.get('content', [])
#         if not content:  # If content is empty, continue processing the stack
#             continue

#         # Handle textual content
#         for paragraph in content:
#             if isinstance(paragraph, str):  # If plain text
#                 combined_text = f"{title_context} - {paragraph.strip()}"
#                 chunks = splitter.split_text(combined_text)
#                 sentences.extend(chunks)
#             elif isinstance(paragraph, dict):  # If nested structure
#                 nested_sentences = convert_nested_to_sentence(paragraph, f"{title_context} -")
#                 for nested in nested_sentences:
#                     chunks = splitter.split_text(nested)
#                     sentences.extend(chunks)

#     return sentences

# Directory paths
directory_path = r'D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\data'
output_directory = r'D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\output'
index_tracker_path = os.path.join(output_directory, "index_tracker.json")

# Initialize the index tracker
index_tracker = []

# Create output directory if not exist
os.makedirs(output_directory, exist_ok=True)

# Processing each JSON file in the directory
for file_name in os.listdir(directory_path):
    if file_name.endswith('.json') and not file_name.endswith('_metadata.json'):  # Skip metadata files
        # Load the JSON content
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            try:
                documents = json.load(file)
            except json.JSONDecodeError as e:
                print(f"Error loading {file_name}: {e}")
                continue  # Skip problematic files

        # Process the documents (embedding generation)
        sentences = process_hierarchical_data_with_splitting(documents)

        embeddings = []
        metadata = []

        # Generate embeddings for each sentence
        for sentence in sentences:
            embedding = model.encode(sentence)
            if embedding is not None and len(embedding) > 0:
                embeddings.append(embedding)
                metadata.append({'content': sentence})
            else:
                print(f"Warning: Empty embedding for sentence: {sentence}")

        # Convert embeddings to numpy array
        if embeddings:
            embeddings_np = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_np)  # Normalize for cosine similarity

            # Create FAISS index
            index = faiss.IndexFlatL2(embeddings_np.shape[1])
            index.add(embeddings_np)

            # Save the index
            index_file_name = f"{os.path.splitext(file_name)[0]}_embeddings.index"
            faiss.write_index(index, os.path.join(output_directory, index_file_name))

            # Save metadata
            metadata_file_name = f"{os.path.splitext(file_name)[0]}_metadata.json"
            with open(os.path.join(output_directory, metadata_file_name), 'w') as f:
                json.dump(metadata, f)

            # Add entry to the index tracker
            index_tracker.append({
                "json_file": file_name,
                "index_file": index_file_name,
                "metadata_file": metadata_file_name,
                "num_sentences": len(sentences)  # You can store additional info
            })

            print(f"Embeddings and metadata for '{file_name}' created successfully!")

# Save the index tracker to a JSON file
with open(index_tracker_path, 'w') as f:
    json.dump(index_tracker, f, indent=4)

print(f"Index tracker saved to {index_tracker_path}")
