import os
import pymupdf
import pytesseract
from PIL import Image
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from pinecone import Pinecone,ServerlessSpec
import dotenv
from groq import Groq

# Set Tesseract command path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize Pinecone
# Load environment variables
dotenv.load_dotenv()
PINECONE_API_KEY = os.getenv('PINE_API')

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists, if not, create it
index_name = "rag-index"  # Set your index name
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=1536, metric='euclidean', 
                    spec=ServerlessSpec(cloud='aws', region='us-east-1'))

# Initialize Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = pymupdf.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from {pdf_path}: {e}")
        return None

# Process the PDF folder
def process_pdf(pdf_file):
    st.write(f"Processing file: {pdf_file.name}")
    text = extract_text_from_pdf(pdf_file)
    if text is None:
        st.error("No text found, applying OCR...")
        # Here you would apply OCR if needed, currently only text extraction is shown
    return text

# Load embeddings
def load_embeddings(text_data):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_data, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings

# Generate a response using Groq
def generate_response(prompt):
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama3-70b-8192",  # Specify the model you want to use
    )
    return chat_completion.choices[0].message.content

# Main function to create the app
def main():
    st.title("PDF Text Extraction and Querying App")
    
    # Upload PDF files
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if st.button("Process PDFs"):
        if uploaded_files:
            text_data = []
            for pdf_file in uploaded_files:
                text = process_pdf(pdf_file)
                if text:
                    text_data.append(text)

            if text_data:
                embeddings = load_embeddings(text_data)

                # Store embeddings in FAISS index
                dim = embeddings.shape[1]
                index = faiss.IndexFlatL2(dim)
                index.add(embeddings)

                st.success(f"Indexed {len(embeddings)} text files with {dim} dimensions.")

                # Query input
                query = st.text_input("Enter your query:")
                if query:
                    # Generate embedding for the query
                    query_embedding = embeddings.embed_query(query)

                    # Perform similarity search in FAISS
                    search_results = index.search(query_embedding, k=5)  # Adjust k as needed
                    retrieved_texts = [text_data[result] for result in search_results[1]]

                    # Prepare prompt for Groq
                    prompt = f"Based on the following text, answer the query: '{query}'\n\n" + "\n".join(retrieved_texts)

                    # Generate response using Groq
                    response = generate_response(prompt)
                    st.write("Generated Response:")
                    st.write(response)

if __name__ == "__main__":
    main()
