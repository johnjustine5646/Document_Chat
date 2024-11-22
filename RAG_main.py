#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pymupdf
import pytesseract
from pdfminer.high_level import extract_text
from PIL import Image, ImageOps, ImageFilter
import cv2  # For multilingual OCR
import os
import numpy as np
import fitz
import genai


# In[3]:


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# In[4]:


doc = pymupdf.open(r"D:\SEM 5\RAG\sample_pdfs\en\Blue_Ocean_Strategy,_Expanded_Edition_How_to_Create_Uncontested-2.pdf") 
out = open("output.txt", "wb")
for page in doc: 
    text = page.get_text().encode("utf8") 
    out.write(text) 
    out.write(bytes((12,)))
out.close()


# In[5]:


def extract_text_from_pdf(pdf_path, language = 'eng'):
    try:
        doc = pymupdf.open(pdf_path)
        text = ""
        for page in doc:
            page_text = page.get_text()
            text += page_text
            if page_text:
                text += page_text
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None
    


# In[6]:


# Process the PDFs

def process_pdf_folder(input_folder, output_folder, language = 'eng'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(input_folder, filename)
            print(f"Found PDF: {pdf_path}...")

            text = extract_text_from_pdf(pdf_path)

            if text is None:
                print(f'No text found in {filename}, Applying OCR using pytesseract...')
                text = ocr_from_pdf(pdf_path, language)

            output_filename = os.path.splitext(filename)[0] + '.txt'
            output_path = os.path.join(output_folder, output_filename)

            with open(output_path, 'w', encoding='utf-8') as file_out:
                file_out.write(text)

            print(f"Saved text to {output_path}")


# In[7]:


# English PDFs

eng_pdfs= r"D:\SEM 5\RAG\sample_pdfs\en"
output_folder_en = r"D:\SEM 5\RAG\converted_files\en"

process_pdf_folder(eng_pdfs, output_folder_en, language = 'eng')


# In[8]:


# Bengali PDFs

bengali_pdfs= r"D:\SEM 5\RAG\sample_pdfs\bn"
output_folder_bn = r"D:\SEM 5\RAG\converted_files\bn"

process_pdf_folder(bengali_pdfs, output_folder_bn, language = 'ben')


# In[9]:


# Urdu PDFs

urdu_pdfs= r"D:\SEM 5\RAG\sample_pdfs\ur"
output_folder_ur = r"D:\SEM 5\RAG\converted_files\ur"

process_pdf_folder(urdu_pdfs, output_folder_ur, language = 'urd')


# In[10]:


# Chinese PDFs

chinese_pdfs= r"D:\SEM 5\RAG\sample_pdfs\zh"
output_folder_zh = r"D:\SEM 5\RAG\converted_files\zh"

process_pdf_folder(chinese_pdfs, output_folder_zh, language = 'chi_sim')


# ---

# ## Vector Embeddings

# In[11]:


from sentence_transformers import SentenceTransformer
import faiss


# In[12]:


model = SentenceTransformer('all-MiniLM-L6-v2')
text_file_folder = r"D:\SEM 5\RAG\converted_files"
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Iterate through text files and generate embeddings
text_data = []
file_names = []

for root, dirs, files in os.walk(text_file_folder):
    for file_name in files:
        if file_name.endswith('.txt'):  # Check for text files
            file_path = os.path.join(root, file_name)
            text = read_text_file(file_path)
            text_data.append(text)  # Store the text content
            file_names.append(file_name)
 


# In[13]:


embeddings = model.encode(text_data, convert_to_numpy= True)
print(embeddings.shape)

embeddings = np.array(embeddings)
embeddings = embeddings/np.linalg.norm(embeddings, axis=1, keepdims=True)


# In[14]:


dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

print(f"Indexed {len(embeddings)} text files with {dim} dimensions")


# In[15]:


faiss.write_index(index, 'faiss_index.bin')


# ## Retrieval

# In[16]:


from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.chat_models import ChatOllama
import pinecone
pinecone.init
# Load the FAISS index
index = faiss.read_index('faiss_index.bin')

# Load the text files
text_files = [os.path.join(text_file_folder, file_name) for file_name in file_names]


# In[17]:


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# In[18]:


import os
from pinecone import Pinecone, ServerlessSpec

PINECONE_API_KEY = os.getenv('PINE_API')
 
index_name="pdf-rag-system"

# Initialize Pinecone using the Pinecone class
pc = Pinecone(
    api_key='7cfea3c1-2d79-4c22-8ec7-9e5fb82897e3'
)

# Check if the index already exists
if index_name not in pc.list_indexes().names():
    # Create an index with the desired name and dimension
    pc.create_index(
        name=index_name,
        dimension=384 ,  # Adjust dimension based on your embeddings
        spec=ServerlessSpec(
            cloud='aws',  # Specify cloud provider, e.g., 'aws'
            region='us-east-1'  # Try using a region like 'us-east-1'
        )
    )

# Connect to the created index
index = pc.Index(index_name)
print(f"Index '{index_name}' created and connected.")


# In[19]:


index = pc.Index(index_name)
print(f"Index '{index_name}' connected.")


# In[20]:


import pinecone

# Assume embeddings and index are initialized
query = input("Enter your query: ")

# Generate embedding for the query
query_embedding = embeddings.embed_query(query)

# Perform similarity search on Pinecone
search_results = index.query(
    namespace="ns1",  # Replace with your actual namespace
    vector=query_embedding,
    top_k=5,  # Get more results for training
    include_values=True,
    include_metadata=True
)

# Extract text from the search results
retrieved_texts = []
for match in search_results['matches']:
    retrieved_texts.append(match['metadata']['text'])  # Assuming metadata has the text field

# Combine retrieved documents into one dataset
training_data = "\n".join(retrieved_texts)


# In[21]:


# Save the training data to a text file (simple format for fine-tuning)
with open('training_data.txt', 'w', encoding='utf-8') as f:
    f.write(training_data)


# In[ ]:


from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Load the training data
def load_dataset(file_path, tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128  # Tokenize into blocks
    )

train_dataset = load_dataset('training_data.txt', tokenizer)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Masked language modeling (if applicable, change to True)
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",  # Save the fine-tuned model here
    overwrite_output_dir=True,
    num_train_epochs=3,  # Adjust based on your needs
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

# Start fine-tuning
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")


# In[26]:


import os
import dotenv
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client with your API key
client = Groq(
    api_key='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',  # Replace with your actual Groq API key
)

# Load your training data (you may want to preprocess this for your needs)
def load_training_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

training_data = load_training_data('training_data.txt')

# Prepare a prompt for the model
prompt = f"Using the following training data, generate a response:\n{training_data}\n\nResponse:"

# Create a chat completion request
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    model="llama3-70b-8192",  # Specify the model you want to use
)

# Extract and print the generated response
response_content = chat_completion.choices[0].message.content
print("Generated Response:\n", response_content)


# In[ ]:




