import streamlit as st
import os
import time
from src.ingestion import extract_text_from_pdf, extract_text_from_docx, extract_text_from_pptx
from src.vector_store import chunk_documents
from src.rag_engine import initialize_rag_system

st.set_page_config(page_title="Educational RAG Chatbot", page_icon="📚", layout="wide")
st.title("📚 Document Intelligence Chatbot")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Bot Configuration")
    
    # 1. Dynamic Mode Selection Toggle
    rag_mode = st.radio(
        "Choose Chatbot Persona:",
        ["Strict RAG", "Open/Guided Tutor"],
        help="Strict Mode restricts answers to document facts. Tutor Mode allows the AI to expand and explain concepts deeply."
    )
    
    st.markdown("---")
    st.header("📂 Knowledge Ingestion")
    uploaded_files = st.file_uploader("Upload course files", type=["pdf", "docx", "pptx"], accept_multiple_files=True)
    process_btn = st.button("🚀 Index Uploaded Documents")

# Keep track of the current mode in session state to see if it changed
if "current_mode" not in st.session_state:
    st.session_state["current_mode"] = rag_mode

# If the user switches the radio button, clear the old chain so it rebuilds with new instructions
if st.session_state["current_mode"] != rag_mode or "rag_chain" not in st.session_state:
    st.session_state["current_mode"] = rag_mode
    with st.spinner(f"Configuring pipeline for {rag_mode}..."):
        try:
            chain, retriever = initialize_rag_system(mode=rag_mode)
            st.session_state["rag_chain"] = chain
            st.session_state["retriever"] = retriever
        except Exception as e:
            st.error(f"Failed to initialize RAG system: {e}")
            st.stop()

# --- BACKEND LOGIC: Processing New Files ---
if process_btn and uploaded_files:
    with st.sidebar:
        for uploaded_file in uploaded_files:
            st.info(f"Processing {uploaded_file.name}...")
            temp_path = os.path.join("data", "raw", uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            ext = uploaded_file.name.split('.')[-1].lower()
            raw_text = ""
            if ext == 'pdf': raw_text = extract_text_from_pdf(temp_path)
            elif ext == 'docx': raw_text = extract_text_from_docx(temp_path)
            elif ext == 'pptx': raw_text = extract_text_from_pptx(temp_path)
                
            if not raw_text.strip(): continue
                
            chunks, metadatas = chunk_documents({uploaded_file.name: raw_text})
            vector_store_instance = st.session_state["retriever"].vectorstore
            
            for i in range(0, len(chunks), BATCH_SIZE:=50):
                vector_store_instance.add_texts(texts=chunks[i:i+BATCH_SIZE], metadatas=metadatas[i:i+BATCH_SIZE])
                if i + BATCH_SIZE < len(chunks): time.sleep(5)
            st.success(f"🎉 {uploaded_file.name} successfully learned!")

# --- CHAT UI INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask something about your course materials..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("Processing..."):
            try:
                answer = st.session_state["rag_chain"].invoke(user_query)
                response_placeholder.markdown(answer)
                
                source_docs = st.session_state["retriever"].invoke(user_query)
                seen_sources = set(doc.metadata.get('source', 'Unknown File') for doc in source_docs)
                if seen_sources:
                    with st.expander("🔍 Verified Source Documents"):
                        for source in seen_sources: st.write(f"- {source}")
                            
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- FIXED STICKY FOOTER DISCLAIMER ---
footer_html = """
<style>
.fixed-footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: rgba(255, 255, 255, 0.9); /* Blends with light themes */
    color: #666666;
    text-align: center;
    padding: 8px;
    font-size: 13px;
    border-top: 1px solid #e6e6e6;
    z-index: 999;
}
/* If using Streamlit dark theme, adjust background-color in your browser settings if needed */
@media (prefers-color-scheme: dark) {
    .fixed-footer {
        background-color: rgba(14, 17, 23, 0.9);
        color: #aaaaaa;
        border-top: 1px solid #262730;
    }
}
</style>
<div class="fixed-footer">
    💡 <b>Tip for Faster Results:</b> Use specific prompts with clear topic names and unit details (e.g., <i>"Explain thresholding from Unit 1"</i>).
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)