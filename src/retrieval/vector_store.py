import sys
import os
# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.retrieval.retrieve import retrieve_documents
from together import Together
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("TOGETHERAI_API_KEY")


def generate_response(query, top_k=3):
    # Retrieve relevant documents
    retrieved_docs = retrieve_documents(query, top_k)
    
    # Debug: Print retrieved documents to check their content
    print("Retrieved Documents:")
    for doc in retrieved_docs:
        print(doc)

    # Prepare context from retrieved documents
    if not retrieved_docs:
        return "No relevant documents found.", "No sources available."
    
    context = "\n\n".join([f"Source {i+1}: {doc['text']}\n(Metadata: {doc['metadata']})" for i, doc in enumerate(retrieved_docs)])
    
    # Construct the prompt
    prompt = f"""Use the following context to answer the question:
    {context}
    \nQuestion: {query}
    \nAnswer:"""
    
    client = Together(api_key=api_key)
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role": "system", "content": "You are an AI assistant."}, 
                  {"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>", "<|eom_id|>"],
        stream=True,
    )
    
    return response, retrieved_docs 

# Example usage
query = "How do young adults perceive parental sharing of their children's content?"
response_stream, source_context = generate_response(query)

# Print the streaming response properly
print("Answer:")
for chunk in response_stream:
    if chunk.choices[0].delta.content:  # Access content safely
        print(chunk.choices[0].delta.content, end="", flush=True)

print("\n\nSources:\n", source_context)
