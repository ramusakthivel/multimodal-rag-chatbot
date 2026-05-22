import os
import time
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables from our .env file (gets our GOOGLE_API_KEY)
load_dotenv()

def chunk_documents(extracted_data):
    """
    Takes the dictionary of raw text from ingestion and splits it into 
    smaller chunks with semantic overlap.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,       # Number of characters per chunk
        chunk_overlap=100,    # Overlap between consecutive chunks to avoid breaking context
        length_function=len
    )
    
    chunks = []
    metadatas = []
    
    for filename, text in extracted_data.items():
        if not text.strip():
            continue
        
        # Split the text file into pieces
        file_chunks = text_splitter.split_text(text)
        
        for chunk in file_chunks:
            chunks.append(chunk)
            # Metadata keeps track of which document this chunk came from
            metadatas.append({"source": filename})
            
    return chunks, metadatas

def build_vector_store(chunks, metadatas, index_name="multimodal-rag-index"):
    """Pipes text chunks straight into cloud Pinecone indexes."""
    if not chunks: return None
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    print("Uploading elements to Cloud Pinecone index...")
    # LangChain connects over the network automatically using PINECONE_API_KEY from your environment
    vector_db = PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas,
        index_name=index_name
    )
    return vector_db

if __name__ == "__main__":
    # Test the standalone pipeline setup
    from ingestion import process_raw_documents
    
    print("--- Running Vector Store Pipeline Test ---")
    raw_data = process_raw_documents()
    
    text_chunks, metadata_list = chunk_documents(raw_data)
    print(f"Generated {len(text_chunks)} chunks from processed documents.")
    
    db = build_vector_store(text_chunks, metadata_list)