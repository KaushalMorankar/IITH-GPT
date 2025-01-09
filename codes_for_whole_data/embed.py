import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to process JSON data with hierarchical titles and content
def process_hierarchical_data_with_splitting(documents, chunk_size=200, chunk_overlap=50, min_words=20):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", ". ", " "]
    )

    sentences = []
    current_main_heading = None  # To track the main heading

    for doc in documents:
        current_title = doc.get('title', '').strip()
        content = doc.get('content', [])

        if not current_title:
            continue

        # If content is empty, set the current title as the main heading
        if not content:
            current_main_heading = current_title
            continue

        # If content exists, process it under the current main heading
        if current_main_heading:
            title_context = f"{current_main_heading} > {current_title}"
        else:
            title_context = current_title

        combined_paragraph = ""
        for i, paragraph in enumerate(content):
            if isinstance(paragraph, str):
                paragraph = paragraph.strip()

                # Append short sentences within the same content
                if len(paragraph.split()) < min_words:
                    combined_paragraph += (", " if combined_paragraph else "") + paragraph
                else:
                    # If combined_paragraph exists, finalize it
                    if combined_paragraph:
                        combined_text = f"{title_context} - {combined_paragraph.strip()}"
                        sentences.append(combined_text)
                        combined_paragraph = ""

                    # Process the current long paragraph
                    combined_paragraph += paragraph

                # Split sentences intelligently
                while len(combined_paragraph.split()) >= min_words:
                    split_index = find_split_index(combined_paragraph, min_words)
                    if split_index == -1:
                        break

                    # Extract the chunk
                    chunk = combined_paragraph[:split_index].strip()
                    combined_text = f"{title_context} - {chunk}"
                    sentences.append(combined_text)

                    # Update the combined_paragraph with remaining content
                    combined_paragraph = combined_paragraph[split_index:].strip()

            elif isinstance(paragraph, dict):
                # Handle nested content
                nested_sentences = convert_nested_to_sentence(paragraph, f"{title_context} -")
                for nested in nested_sentences:
                    chunks = splitter.split_text(nested)
                    sentences.extend(chunks)

        # Handle remaining combined paragraph for the current content
        if combined_paragraph.strip():
            combined_text = f"{title_context} - {combined_paragraph.strip()}"
            sentences.append(combined_text)

    return sentences


# Function to find the split index for sentences
def find_split_index(text, min_words):
    words = text.split()

    # Return -1 if the sentence has fewer words than the minimum threshold
    if len(words) < min_words:
        return -1

    # Find the split point at the nearest punctuation
    for i in range(min_words, len(words)):
        if words[i].endswith(('.', '!', '?')):
            return len(' '.join(words[:i + 1]))

    # If no punctuation is found, do not split
    return -1

# Function to recursively convert nested structures to sentences
def convert_nested_to_sentence(data, prefix=""):
    sentences = []
    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix} {key}".strip()
            sentences.extend(convert_nested_to_sentence(value, new_prefix))
    elif isinstance(data, list):
        for item in data:
            sentences.extend(convert_nested_to_sentence(item, prefix))
    else:
        sentences.append(f"{prefix} is {data}".strip())
    return sentences

# Process JSON files in the directory
directory_path = r'D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\data'
output_directory = r'D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\output'
os.makedirs(output_directory, exist_ok=True)

for file_name in os.listdir(directory_path):
    if file_name.endswith('.json'):
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            try:
                documents = json.load(file)
            except json.JSONDecodeError as e:
                print(f"Error loading {file_name}: {e}")
                continue

        sentences = process_hierarchical_data_with_splitting(documents)

        embeddings = []
        metadata = []

        for sentence in sentences:
            embedding = model.encode(sentence)
            if embedding is not None and len(embedding) > 0:
                embeddings.append(embedding)
                metadata.append({'content': sentence})
            else:
                print(f"Warning: Empty embedding for sentence: {sentence}")

        if embeddings:
            embeddings_np = np.array(embeddings, dtype=np.float32)
            print(f"Shape of embeddings for {file_name}: {embeddings_np.shape}")

            if embeddings_np.shape[0] > 0:
                faiss.normalize_L2(embeddings_np)
            else:
                print(f"Warning: No valid embeddings in '{file_name}'")
        else:
            print(f"Warning: No embeddings for '{file_name}'")

        if embeddings_np.shape[0] > 0:
            index = faiss.IndexFlatL2(embeddings_np.shape[1])
            index.add(embeddings_np)

            index_file_name = f"{os.path.splitext(file_name)[0]}_embeddings.index"
            faiss.write_index(index, os.path.join(output_directory, index_file_name))

            metadata_file_name = f"{os.path.splitext(file_name)[0]}_metadata.json"
            with open(os.path.join(output_directory, metadata_file_name), 'w') as f:
                json.dump(metadata, f)

            print(f"Embeddings and metadata for '{file_name}' created successfully!")
        else:
            print(f"Skipping index creation for '{file_name}' due to no valid embeddings.")