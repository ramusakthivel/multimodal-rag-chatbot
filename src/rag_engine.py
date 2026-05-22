import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore

# Load API credentials from our .env file
load_dotenv()

def format_docs(docs):
    """Formats retrieved document chunks into a single text block."""
    return "\n\n".join(doc.page_content for doc in docs)

def initialize_rag_system(mode="Strict RAG", index_name="multimodal-rag-index"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Connects instantly to your live cloud vector network cluster
    vector_db = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    """
    Connects to local ChromaDB and dynamically builds either a Strict or an Open/Guided RAG system
    based on the user's UI selection.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Vector database not found at {db_path}. Run vector_store.py first.")
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    
    # --- DYNAMIC CONFIGURATION BASED ON USER SELECTION ---
    if mode == "Strict RAG":
        # 1. Rigid factual matching
        temperature = 0.0
        system_prompt = (
            "You are a strict factual document retriever. Use ONLY the following pieces of retrieved context "
            "to answer the question. If you do not know the answer or if it's not found explicitly in the context, "
            "say 'I cannot find the answer within the uploaded documents.' Do not attempt to make up "
            "information or extrapolate outside the provided text.\n\n"
            "Context:\n{context}"
        )
    else:
        # 2. Conversational university professor mode
        temperature = 0.5
        system_prompt = (
            "You are an inspiring, expert university professor and data science tutor. Use the provided context "
            "blocks extracted from the student's actual course documents as your core syllabus foundation.\n\n"
            "Guidelines:\n"
            "1. Ground your core answer in the provided course context so the student learns their official syllabus.\n"
            "2. Use your vast data science knowledge to expand on the topic. Explain how/why it works, give real-world "
            "engineering examples, or provide code snippets if helpful.\n"
            "3. If a topic is completely absent from the context, explicitly say: 'Note: This specific topic isn't in your "
            "uploaded notes, but here is how it works conceptually...' and then explain it cleanly.\n\n"
            "Context:\n{context}"
        )

    # Initialize the LLM with the dynamic temperature
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

if __name__ == "__main__":
    print("--- Testing Modern LCEL RAG Engine ---")
    try:
        # Initialize the pipeline
        chain, retriever = initialize_rag_system()
        
        # Define a test question
        test_query = "What are the main concepts covered in Unit 1?"
        print(f"\nAsking Question: '{test_query}'")
        
        # Execute the chain
        answer = chain.invoke(test_query)
        
        print("\n🤖 Bot Response:")
        print(answer)
        
        # Explicitly print out source citations
        print("\n📄 Source Verification:")
        source_docs = retriever.invoke(test_query)
        seen_sources = set(doc.metadata.get('source', 'Unknown File') for doc in source_docs)
        for source in seen_sources:
            print(f"- {source}")
            
    except Exception as e:
        print(f"An error occurred in the RAG Engine: {e}")