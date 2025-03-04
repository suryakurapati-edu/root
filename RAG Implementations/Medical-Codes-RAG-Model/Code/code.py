import google.generativeai as genai
import faiss
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from fastapi import FastAPI
import asyncio
import uvicorn
from pydantic import BaseModel, Field
from pymongo import MongoClient
import os

# Initialize FastAPI app
app = FastAPI()

# Model related variables
dimension = 768  # Gemini Embedding size
index_standard_codes = faiss.IndexFlatL2(dimension)  # Similarity index for standard codes
texts_standard_codes = []

# LLM Gemini Models
embedding_model = "models/embedding-001"
llm_model = "gemini-1.5-pro"

# Mongo Details
connection_string = "mongodb://localhost:27017/"
database = "db_genai"
collection = "col_rag_gemini"

# Connect to MongoDB
def connect_and_fetch(connection_string, database, collection, query):
    try:
        client = MongoClient()  
        db = client[database]
        collection = db[collection]
        document = collection.find_one(query)
        return document
    except Exception as e:
        print(f"Error: {e}")
        return None

# Configure Gemini API key
gemini_doc = connect_and_fetch(connection_string, database, collection, query={"_id": "gemini"})
os.environ["GOOGLE_API_KEY"] = gemini_doc['api_key']

# Define custom class for input request body
class req_standard(BaseModel):
    pdf_path: str = Field(..., description='Pass the PDF file')

class req(BaseModel):
    query: str = Field(..., description='Pass the User Query')
    pdf_path: str = Field(..., description='Pass the PDF file')


# Read the PDF file pages
async def pdf_loader(path):
    loader = PyPDFLoader(path)
    pages = loader.load()
    return pages


# Split PDF file into chunks
async def split_pdf(pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(pages)
    chunks = [chunk.page_content for chunk in text_chunks]
    return chunks


# Convert chunks into embeddings and store in vector store FAISS
async def generate_embeddings(embedding_model, chunks, index, texts):
    vectors = []
    # texts.clear()

    for chunk in chunks:
        response = genai.embed_content(model=embedding_model, content=chunk, task_type="retrieval_document")
        response = np.array(response['embedding'])
        vectors.append(response)
        texts.append(chunk)

    vectors = np.vstack(vectors)  # Convert to FAISS format
    index.add(vectors)  # Add to FAISS index
    return index


# Perform Similarity Search
async def similarity_search(embedding_model, query, index, texts):
    query_embedding = np.array(genai.embed_content(model=embedding_model, content=query, task_type="retrieval_query")['embedding']).reshape(1, -1)
    distances, indices = index.search(query_embedding, 2)  # Fetch top 2 matches

    # Ensure indices don't exceed the range of texts
    matching_chunks = [texts[i] for i in indices[0] if i < len(texts)]  # Retrieve matching chunks

    # Debugging: Print the matching chunks
    print(f"Matching chunks: {matching_chunks}")
    return matching_chunks


# Use LLM to generate an answer based on relevant chunks
async def gen_llm_answer(llm_model, query, matching_chunks):
    context = "\n".join(matching_chunks)  # Combine all relevant chunks into context
    prompt = f"""
    Answer the following question based on the provided context:

    Context: {context}

    Question: {query}
    """
    response = await asyncio.to_thread(genai.GenerativeModel(llm_model).generate_content, prompt)
    return response.text.strip()


# Endpoint1: Read standard codes (ICD, CPT, HCPCS) from PDF and store in vector store
@app.post("/read_standard_codes/")
async def read_standard_codes(req: req_standard):
    if req.pdf_path:
        pdf_path = req.pdf_path
    else:
        raise ValueError("Error! - Missing PDF file path.")

    pages = await pdf_loader(pdf_path)
    chunks = await split_pdf(pages)
    
    # Use the index for standard codes
    global index_standard_codes
    global texts_standard_codes
    index_standard_codes = await generate_embeddings(embedding_model, chunks, index_standard_codes, texts_standard_codes)
    
    return {"message": "Standard codes stored successfully."}


# Endpoint2: Extract codes from patient PDF input and fetch from vector store
@app.post("/extract_codes/")
async def extract_codes(req: req):
    if req.query:
        query = req.query
    else:
        raise ValueError("Error! - Missing Query from the User.")

    if req.pdf_path:
        pdf_path = req.pdf_path
    else:
        raise ValueError("Error! - Missing PDF file path from the User")

    pages = await pdf_loader(pdf_path)
    chunks = await split_pdf(pages)

    # Perform similarity search on the standard codes index
    global index_standard_codes
    global texts_standard_codes
    matching_chunks = await similarity_search(embedding_model, query, index_standard_codes, texts_standard_codes)
    response = await gen_llm_answer(llm_model, query, matching_chunks)

    # Debugging: Check if the result is as expected
    print(f"Matching chunks: {matching_chunks}")
    return {"query": query, "answer": response, "context": matching_chunks}


# Endpoint3: Original endpoint for querying PDFs (RAG Model)
@app.post("/query_pdf/")
async def query_pdf(req: req):
    if req.query:
        query = req.query
    else:
        raise ValueError("Error! - Missing Query from the User.")
    
    if req.pdf_path:
        pdf_path = req.pdf_path
    else:
        raise ValueError("Error! - Missing PDF file path from the User")

    pages = await pdf_loader(pdf_path)
    chunks = await split_pdf(pages)
    
    # Use the standard codes index for general querying
    global index_standard_codes
    index_standard_codes = await generate_embeddings(embedding_model, chunks, index_standard_codes, texts_standard_codes)
    
    matching_chunks = await similarity_search(embedding_model, query, index_standard_codes, texts_standard_codes)
    response = await gen_llm_answer(llm_model, query, matching_chunks)

    return {"query": query, "answer": response, "context": matching_chunks}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
