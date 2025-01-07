# # import json
# # from sentence_transformers import SentenceTransformer

# # # Load the pre-trained model (SBERT)
# # model = SentenceTransformer('all-MiniLM-L6-v2')

# # # Load your JSON data from a file
# # with open('D:\\pytorch_projects+tensorflow_projects_3.12\\IITH_GPT\\About_section\\about_iith_data.json', 'r') as f:
# #     data = json.load(f)

# # # Extract text from the JSON data
# # text_data = []

# # # Function to recursively extract text content from JSON
# # def extract_text_from_json(data):
# #     if isinstance(data, dict):
# #         for key, value in data.items():
# #             if isinstance(value, str):
# #                 text_data.append(value)
# #             else:
# #                 extract_text_from_json(value)  # Recursive call to handle nested structures
# #     elif isinstance(data, list):
# #         for item in data:
# #             extract_text_from_json(item)  # Recursive call for list items

# # # Extracting text data from your JSON structure
# # extract_text_from_json(data)

# # # Function to split long texts into sentences based on full stops
# # def split_into_sentences(text, max_length=300):
# #     """
# #     Split text into smaller chunks based on full stops if it exceeds a certain length.
# #     """
# #     if len(text) > max_length:
# #         sentences = text.split('. ')  # Split by '. ' for sentence boundary
# #         return [sentence.strip() for sentence in sentences if sentence.strip()]
# #     else:
# #         return [text]  # If text is short, no need to split

# # # Generate embeddings for each extracted text
# # chunked_texts = []
# # embeddings = []

# # for text in text_data:
# #     # Split long text into smaller sentences if necessary
# #     chunks = split_into_sentences(text)
# #     chunked_texts.extend(chunks)
# #     chunk_embeddings = model.encode(chunks)  # Generate embeddings for each chunk
# #     embeddings.extend(chunk_embeddings)

# # # Print the first few chunks and their embeddings
# # for idx, embedding in enumerate(embeddings[:10]):  # Display the first 10 embeddings for demonstration
# #     print(f"Chunked Text: {chunked_texts[idx]}")
# #     print(f"Embedding: {embedding[:5]}...")  # Print the first 5 elements of the embedding
# #     print()



# # import json
# # import numpy as np
# # from sentence_transformers import SentenceTransformer

# # # Load the pre-trained model (SBERT)
# # model = SentenceTransformer('all-MiniLM-L6-v2')

# # # Load the JSON file (replace the path with the actual location of your JSON file)
# # file_path = 'D:\\pytorch_projects+tensorflow_projects_3.12\\IITH_GPT\\About_section\\about_iith_data.json'

# # # Read the JSON data
# # with open(file_path, 'r') as f:
# #     data = json.load(f)

# # # Function to extract text from JSON
# # def extract_text_from_json(data):
# #     text_data = []

# #     # Check if data is a dictionary
# #     if isinstance(data, dict):
# #         for key, value in data.items():
# #             if isinstance(value, str):  # If the value is a string, add it to the list
# #                 text_data.append(value)
# #             elif isinstance(value, (dict, list)):  # If the value is a dictionary or list, recurse
# #                 text_data.extend(extract_text_from_json(value))
    
# #     # Check if data is a list
# #     elif isinstance(data, list):
# #         for item in data:
# #             text_data.extend(extract_text_from_json(item))
    
# #     return text_data

# # # Extracting all text content from the JSON
# # text_data = extract_text_from_json(data)

# # # Generate embeddings for each text segment using Sentence-BERT
# # embeddings = model.encode(text_data)

# # # Save the embeddings in a .npy file
# # np.save('about.npy', embeddings)

# # # Print the shape of the embeddings and a preview of the first embedding
# # print(f"Embeddings shape: {embeddings.shape}")
# # print(f"First embedding (first 5 values): {embeddings[0][:5]}")





# import json
# import re
# import numpy as np
# from sentence_transformers import SentenceTransformer

# # Load the pre-trained model (SBERT)
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Function to split text into sentences
# def split_into_sentences(text):
#     # Split by '.', '!', or '?' followed by a space
#     sentences = re.split(r'(?<=[.!?])\s+', text.strip())
#     return [sentence for sentence in sentences if sentence]  # Remove empty sentences

# # Function to extract sentences from JSON
# def extract_sentences_from_json(data):
#     sentences = []

#     # If the data is a dictionary
#     if isinstance(data, dict):
#         for key, value in data.items():
#             # If value is a string, split it into sentences
#             if isinstance(value, str):
#                 sentences.extend(split_into_sentences(value))
#             elif isinstance(value, (dict, list)):  # If value is a dictionary or list, recurse
#                 sentences.extend(extract_sentences_from_json(value))
    
#     # If the data is a list
#     elif isinstance(data, list):
#         for item in data:
#             sentences.extend(extract_sentences_from_json(item))
    
#     return sentences

# # Main Script
# if __name__ == "__main__":
#     # Load the JSON file
#     file_path = 'D:\\pytorch_projects+tensorflow_projects_3.12\\IITH_GPT\\About_section\\about_iith_data.json'
#     with open(file_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     # Extract sentences from the entire JSON data
#     sentences = extract_sentences_from_json(data)

#     # Ensure no sentences are missed by checking the length and some samples
#     print(f"Extracted {len(sentences)} sentences.")
#     print(f"Sample sentences: {sentences[:5]}")

#     # Generate embeddings for all sentences
#     embeddings = model.encode(sentences)

#     # Save embeddings to a .npy file
#     np.save('iith_data_embeddings.npy', embeddings)

#     # Save the extracted sentences to a text file
#     with open('flattened_data.txt', 'w', encoding='utf-8') as txt_file:
#         txt_file.write('\n'.join(sentences))
    
#     # Output the shape of embeddings and preview the first embedding
#     print(f"Embeddings shape: {embeddings.shape}")
#     print(f"First embedding (first 5 values): {embeddings[0][:5]}")





# import json
# from sentence_transformers import SentenceTransformer

# # Path to the JSON file
# file_path = 'data.json'

# # Read the JSON data from the file
# with open(file_path, 'r') as file:
#     data = json.load(file)

# # Initialize the Sentence Transformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Function to split text into sentences after each full stop
# def split_text(text):
#     return [sentence.strip() for sentence in text.split('.') if sentence]

# # Prepare a list to store all sentences along with titles
# sentences = []

# # Extract titles and content and split content into sentences
# for entry in data:
#     title = entry.get('title', '')
#     for paragraph in entry.get('content', []):
#         # Split the content into sentences
#         split_sentences = split_text(paragraph)
#         # Add title and content sentences
#         for sentence in split_sentences:
#             sentences.append(f"Title: {title}, Sentence: {sentence}")

# # Create embeddings for each sentence
# embeddings = model.encode(sentences)

# # Print the embeddings for each sentence
# for sentence, embedding in zip(sentences, embeddings):
#     print(f"Sentence: {sentence}\nEmbedding: {embedding[:5]}...")  # Print first 5 values of embedding for brevity






import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Step 1: Load the JSON data
with open(r'D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\data\about-iith.json', 'r') as file:
    documents = json.load(file)

# Step 2: Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 3: Function to handle hierarchical titles and content
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

# Step 4: Function to handle nested dictionaries and lists
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

# Step 5: Generate embeddings for the processed sentences
sentences = process_hierarchical_data_with_splitting(documents)

embeddings = []
metadata = []

for sentence in sentences:
    # Generate embedding for each sentence
    embedding = model.encode(sentence)
    embeddings.append(embedding)

    # Store metadata
    metadata.append({'content': sentence})

# Step 6: Convert embeddings to numpy array and normalize
embeddings_np = np.array(embeddings, dtype=np.float32)
faiss.normalize_L2(embeddings_np)  # Normalize for cosine similarity

# Step 7: Create FAISS index and save
index = faiss.IndexFlatL2(embeddings_np.shape[1])
index.add(embeddings_np)
faiss.write_index(index, 'document_embeddings.index')

# Step 8: Save metadata
with open('document_metadata.json', 'w') as f:
    json.dump(metadata, f)

print("Embeddings and metadata created successfully!")

