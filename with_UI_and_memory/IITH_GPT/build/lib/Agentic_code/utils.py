import requests
from llama_index.core.llms import ChatMessage, MessageRole

# Define Gemini API endpoint and API key
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
GEMINI_API_KEY = "AIzaSyAE9uat1s7n2zquCWe5k_vwe_pC_oGNVP8"

def classify_query_with_gemini(query):
    """
    Classify a user query as 'summarization' or 'question_answering' using Google Gemini API.
    """
    # Construct the API URL with the key
    api_url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"

    # Create the request payload
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"""
                        Classify the following query into one of these types:
                        - 'summarization'
                        - 'question_answering'
                        - 'search'
                        - 'fact_verification'
                        - 'exploration'
                        - 'math'

                        Query: {query}

                        Examples:
                        1. What is the capital of India?
                           Output: question_answering
                        2. Summarize the given paragraph.
                           Output: summarization
                        3. Find documents on climate change policies.
                           Output: search
                        4. Verify if the claim 'Earth is flat' is true.
                           Output: fact_verification
                        5. Explore the history of space exploration.
                           Output: exploration
                        6. 3+5*2
                           Output: math
                        7. add 3 and 5
                           Output: math
                        """
                    }
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Make a POST request to the Gemini API
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        # Extract the classification result
        classification = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip().lower()
        # print(f"Predicted class: {classification}")
        return classification

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Gemini API: {e}")
        return "error"
    

def chunk_text(text, chunk_size):
    """Split text into smaller chunks for parallel processing."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def summarize_history(summarizer, messages):
    history_text = " ".join([msg.content for msg in messages if msg.role != MessageRole.SYSTEM])

    # No need to split if text is short
    if len(history_text) <= 1000:
        return summarizer(history_text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']

    # Handle longer texts by splitting
    chunks = chunk_text(history_text, chunk_size=1000)  # Use an optimized chunk size
    summaries = [summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return " ".join(summaries)


import re

def do_math(expression: str) -> str:
    """
    Evaluates a mathematical expression following BODMAS rule.
    
    Parameters:
        expression (str): A string representing the math expression (e.g., '3 + 5 * 2')
    
    Returns:
        str: The result of the evaluation or an error message.
    """
    try:
        # Remove any unwanted characters (allow only numbers, operators, and spaces)
        sanitized_expr = re.sub(r'[^0-9+\-*/(). ]', '', expression)
        
        # Evaluate the expression following BODMAS rules using Python's eval
        result = eval(sanitized_expr)
        
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error: Invalid expression ({str(e)})"


# JSON Schema for Ollama Integration
function_schema = {
    "type": "function",
    "function": {
        "name": "do_math",
        "description": "Evaluates mathematical expressions following BODMAS rules",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '3 + 5 * 2')"
                }
            },
            "required": ["expression"]
        }
    }
}