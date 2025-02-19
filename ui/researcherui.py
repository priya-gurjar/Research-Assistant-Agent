import streamlit as st
import os
import sys
# Dynamically add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.ingestion.pdf_ingestion import ingest_pdfs
from src.retrieval.retrieve import retrieve_documents  # Import retrieval function
from src.retrieval.vector_store import generate_response  # Import LLM response generator

TEMP_DIR="/Users/priyagurjar/Desktop/Machine learning/Research Assistant Agent /data/raw_papers"
os.makedirs(TEMP_DIR, exist_ok=True)
# Streamlit UI
st.title("AI Research Assistant")
st.write("Upload research papers and ask questions!")

# --- New Chat Button ---
if st.button("🆕 New Chat"):
    # Delete all files in TEMP_DIR
    for file in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    st.session_state.clear()  # Reset Streamlit session state
    st.success("🔄 Chat reset! Upload new files to start again.")
    st.rerun()  # Stop execution to refresh UI
# File Upload
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} files uploaded successfully!")
    
    # Save uploaded files to a temp folder
    file_paths=[]
    for file in uploaded_files:
        file_path = os.path.join(TEMP_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        file_paths.append(file_path)
    
    st.write("Files saved. You can now ask questions!")

    #pdf_ingestion
    ingest_pdfs(TEMP_DIR)

    # Query Input
    user_query = st.text_input("Enter your question about the papers:")
    
    if st.button("Get Answer") and user_query:
        with st.spinner("Processing..."):
            response_stream, source_documents = generate_response(user_query)
            
            # Collect full response from streaming output
            response_text = ""
            for chunk in response_stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            
        # Display Answer
        st.subheader("Answer:")
        st.write(response_text)

        # Display Retrieved Sources Correctly
        st.subheader("Sources:")
        if source_documents and isinstance(source_documents, list):
            for i, doc in enumerate(source_documents):
                st.markdown(f"**Source {i+1}:** {doc['metadata']}")  # Display metadata properly
        else:
            st.write("No sources found.")

