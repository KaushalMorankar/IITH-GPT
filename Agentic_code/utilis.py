from llama_index.core import PromptTemplate
from datetime import datetime
from llama_index.core.llms import ChatMessage, MessageRole
from prompts import (
    query_classification_prompt,
    query_classification_prompt_no_doc,
)
import pdfplumber
import json
import os
import pandas as pd
import streamlit as st
from typing import Tuple
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-large")  # You can choose a different T5 model
model = T5ForConditionalGeneration.from_pretrained("t5-large")

def classify_query_utility_t5(user_query, query_type="nope"):
    if 'direct' in query_type:
        prompt = query_classification_prompt.format(user_query=user_query)
    else:
        prompt = query_classification_prompt_no_doc.format(user_query=user_query)
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    try:
        outputs = model.generate(input_ids, max_length=50, num_return_sequences=1, num_beams=3)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error in T5 generation: {e}")
        return "error"
    result = result.strip().lower() 
    print(result)
    if result in ['code_execution', 'summary', 'search', 'comparison']:
        return result
    else:
        return "other"  



model_name = "t5-large"  # You can choose larger models like t5-base or t5-large
tokenizer = T5Tokenizer.from_pretrained(model_name)
model_T5 = T5ForConditionalGeneration.from_pretrained(model_name)
# Prepare the input text for classification
def classify_query_with_t5(query):
    # Updated prompt to instruct T5 to output only 'summarization' or 'question_answering'
    input_text = """Classify the following query as either 'summarization' or 'question_answering': {query}
                #Example 1: What is the capital of India?
                Output : Question-answering

                #Example 2: Summarize the given paragraph.
                Output : Summarization

                #Example 3: What is the weather today?
                Output : Question-answering

                #Example 4: Brief about NSS?
                Output : Summarization
                """

    # Tokenize and encode the input
    input_ids = tokenizer.encode(input_text, return_tensors="pt")  # Move to GPU if available

    # Generate the prediction (adjust max_length and num_beams)
    output = model_T5.generate(input_ids, max_length=5, num_beams=3, early_stopping=True)

    # Decode the output and return the classification
    predicted_class = tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()
    print(f"Predicted class: {predicted_class}")
    return predicted_class  

# prom=classify_query_with_t5("Brief about India")
# def maninmiddle(ans):
#     # Initialize the LLM model
#     if ans=="yes":
#         # llm = Gemini()



# # For language models and agent functionality
# from langchain.agents import Agent
# from langchain.llms import OpenAI

# # For routing logic
# from langchain.chains.router import MultiRouteChain

# # For vector store integration
# from langchain.vectorstores import FAISS  # or your preferred vector store

# # For query evaluation and filtering
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain

# # Optional: For advanced grading and hallucination detection
# from langchain.evaluation import Evaluator
# from langchain_ollama import OllamaLLM
# import torch
# # # For structured logging and debugging
# # import logging
# from langgraph.checkpoint.memory import MemorySaver

# memory = MemorySaver()








# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def load_llama_model(model_name="llama3.1",device=device):
#     """Initialize Ollama and load LLaMA model locally."""
#     return OllamaLLM(model=model_name, device="cuda" if torch.cuda.is_available() else "cpu")

# ollama_model = load_llama_model(model_name="llama3.1",device=device)  # Replace with LLaMA 3.1 model if different



# router_agent=Agent(
#     role='router',
#     goal='route user questions to vector store or google search',
#     backstory=(
#         "You are a router agent that routes user questions to the vector store or Google search. ",
#         "You are responsible for determining the type of question and routing it to the appropriate system. ",
#         "Use vector store for questions on concept related to Retrival augmented generation",
#         "Use Google search for general knowledge questions"

#     ),
#     verbose=True,
#     allow_delegation=False,
#     llm=ollama_model,
# )

# retriver_agent=Agent(
#     role='retriver',
#     goal='retrive the relevant information from the vector store to answer the user question',
#     backstory=(
#         "You are assistant for question answering task",
#         "You are a retriver agent that retrives the relevant information from the vector store. ",
#         "You are responsible for retriving the relevant information from the vector store. ",
#         "Use vector store for questions on concept related to Retrival augmented generation",
#     ),
#     verbose=True,
#     allow_delegation=False,
#     llm=ollama_model,
# )

# Grader_agent=Agent(
#     role='Answer grader',
#     goal='Filter out erroneous retrievals',
#     backstory=(
#         "You are a grader agent assesing relevance of a retrived document to a user query. ",
#         "If a document contains keywords related to the user queries , grade it as relevant. ",
#         "You have to make sure that the answer is relevant to the user query. ",
#     ),
#     verbose=True,
#     allow_delegation=False,
#     llm=ollama_model,
# )

# hallucination_agent=Agent(
#     role='hallucination grader',
#     goal='Filter out hallucination',
#     backstory=(
#         "You are a hallucination grader assessing whether an answer to the query is grounded in/ supported by a set of facts",
#         "Make sure you meticulously review the answer and check if the response provided is in alignment with the question asked"
#     ),
#     verbose=True,
#     allow_delegation=False,
#     llm=ollama_model,
# )

# answer_grader=Agent(
#     role='Answer grader',
#     goal='Filter out erroneous retrievals and hallucinations',
#     backstory=(
#         "You are a grader agent assesing whether an answer is useful to resolve the question ",
#         "Make sure you meticulously review the answer and check if the response provided is in alignment with the question asked ",
#         "If the answer is clear generate a clear and consise answer",
#         "If the answer is not clear ask the user to rephrase the question"
#     ),
#     verbose=True,
#     allow_delegation=False,
#     llm=ollama_model,
# )


