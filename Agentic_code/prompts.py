# Classify the user query into one of these categories: code_execution, summary, search, analysis, comparison
query_classification_prompt = """You are an intelligent AI assistant. You will be provided with a user query. You will have to classify the query into one of the following categories:
- **code_execution**: The query specifically requires a code to be **executed**, such as plotting, performing numerical calculations, or verifying the output of a code. **Note:** If the user only asks to write the code and not execute it, do not classify it as code_execution.
- **summary**: The query is either asking for a summary or it requires a large amount of information to be retrieved and summarized.
- **search**: The query is asking for a specific information which can be answered with a single piece of information.
- **analysis**: The query is asking for a thorough analysis of every part of some document or text, which may require reasoning and understanding of the text.
- **comparison**: The query is asking for a comparison between two or more entities, which may require multiple sources of information.

**Note:**
- If you are not confident about the classification, respond with 'other'.
- Only provide your answer in lowercase. Do not provide any other explanation or response.

User Query: {user_query}
Answer:"""

# Don't classify into analysis if document is not given
query_classification_prompt_no_doc = """
You are an intelligent AI assistant. Classify the user query into one of the following categories:
- code_execution
- summary
- search
- comparison

Your response must be one of these categories and nothing else. Do not include any other text or explanation.

User Query: {user_query}
Answer:"""