import requests

# Define Gemini API endpoint and API key
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
GEMINI_API_KEY = "API_KEY"

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
                        Classify the following query as either 'summarization' or 'question_answering':
                        
                        Query: {query}
                        
                        Examples:
                        1. What is the capital of India?
                           Output: question_answering
                        2. Summarize the given paragraph.
                           Output: summarization
                        3. What is the weather today?
                           Output: question_answering
                        4. Brief about NSS.
                           Output: summarization
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
        classification = result.get("contents", [{}])[0].get("parts", [{}])[0].get("text", "").strip().lower()
        print(f"Predicted class: {classification}")
        return classification

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Gemini API: {e}")
        return "error"