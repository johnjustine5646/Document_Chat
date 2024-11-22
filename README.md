# PDF Text Extraction and Querying App  

## Overview  
This project is about making an app that lets you upload PDF files, get the text from them, and then ask questions about the text. It uses AI and something called embeddings to make smart responses to your questions. We also use a special model called TinyLlama, but we made it better by fine-tuning it.  

---

## How It Works  
The app has some main parts that do different jobs:  

```plaintext  
+--------------------------+  
|     User Interface       |  
|        (Frontend)        |  
+--------------------------+  
           |  
           v  
+--------------------------+  
|  PDF Text Extraction     |  
|  (Extracts from PDF or   |  
|    uses OCR if needed)   |  
+--------------------------+  
           |  
           v  
+--------------------------+  
|   Generate Embeddings    |  
|   (Using SentenceTransformer)  
+--------------------------+  
           |  
           v  
+--------------------------+  
|       FAISS Index        |  
|   (For finding similar   |  
|      text quickly)       |  
+--------------------------+  
           |  
           v  
+--------------------------+  
|  Groq AI and TinyLlama   |  
|   (Generates responses)  |  
+--------------------------+  

## What Each Part Does
1. User Interface:

 - This is the main page where you upload PDFs and type your questions.
 - It uses Streamlit (a library for Python).
2. PDF Text Extraction:

- Reads text from PDFs with PyMuPDF. If it doesn’t work, it tries OCR with Pytesseract.
Don’t forget to install Tesseract separately, or it won’t work!
3. Embedding Generation:

- Converts the PDF text into embeddings with SentenceTransformer. These embeddings are just numbers that represent the text.
It's hard to explain what embeddings are, but they make finding similar stuff easier.
4. FAISS Index:

 - Stores embeddings in something like a "vector database" so the app can quickly find the best match for your questions.
5. Groq AI and TinyLlama Models:

- Groq is a chatbot API (like ChatGPT but not).
- TinyLlama is a language model for responses, but we made it even better by fine-tuning it.
