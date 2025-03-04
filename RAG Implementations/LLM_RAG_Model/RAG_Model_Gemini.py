# Implement PDF search using RAG model
# Step1: Read PDF file
# Step2: Split into chunks using Langchain doucment loaders
# Step3: Convert chunks into Embeddings and store in vector store
# Step4: Perform similarity search against user query
# Step5: Use LLM to answer user query and fetched context/chunk of relevant PDF


# import packages
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

# Initialize FastAPI app
app = FastAPI()

# Model related variables
dimension = 768 # Gemini Embedding size
index = faiss.IndexFlatL2(dimension) # Same similarity index else will fail
texts = []

# LLM Gemini Models
embedding_model = "models/embedding-001"
llm_model = "gemini-pro"

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
genai.configure(api_key=gemini_doc['api_key'])

# Define custom class for input request body
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
async def generate_embeddings(embedding_model, chunks):
    vectors = []
    global texts
    texts.clear()

    for chunk in chunks:
        response = genai.embed_content(model=embedding_model, content=chunk, task_type="retrieval_document")
        response = np.array(response['embedding'])
        vectors.append(response)
        texts.append(chunk)

    vectors = np.vstack(vectors) # convert to FAISS format
    index.add(vectors) # Add to FAISS index
    return index


# Perform Similarity Search
async def similarity_search(embedding_model, query, index):
    query_embedding = np.array(genai.embed_content(model=embedding_model, content=query, task_type="retrieval_query")['embedding']).reshape(1, -1)
    distances, indices = index.search(query_embedding, 2)
    matching_chunks = [texts[i] for i in indices[0] if i < len(texts)]  # Retrieve matching chunks
    return matching_chunks


# Use LLM to generate answer
async def gen_llm_answer(llm_model, query, matching_chunks):
    context = "\n".join(matching_chunks)
    prompt = f"""
    Answer the question from below context only:

    Context: {context}

    Question: {query}
    """
    response = await asyncio.to_thread(genai.GenerativeModel(llm_model).generate_content, prompt)
    return response.text.strip()


@app.post("/query_pdf/")
async def query_pdf(req: req):
    if req.query:
        query = req.query
    else:
        raise "Error! - Missing Query from the User."
    
    if req.pdf_path:
        pdf_path = req.pdf_path
    else:
        raise "Error! - Missing PDF file path from the User"

    pages = await pdf_loader(pdf_path)
    chunks = await split_pdf(pages)
    index = await generate_embeddings(embedding_model, chunks)
    matching_chunks = await similarity_search(embedding_model, query, index)
    response = await gen_llm_answer(llm_model, query, matching_chunks)

    return {"query": query, "answer": response, "context": matching_chunks}


if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)