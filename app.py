import streamlit as st
from langchain_core.documents import Document  # ✅ FIXED
from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ FIXED
from langchain_community.vectorstores import FAISS  # ✅ FIXED
from langchain_google_genai import ChatGoogleGenerativeAI  # ✅ FIXED
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from youtube_transcript_api import YouTubeTranscriptApi
import re
import os

# Page config
st.set_page_config(page_title="YouTube Video Q&A", page_icon="🎥")

# Get API key
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    pattern = r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?]*)'
    match = re.search(pattern, url)
    return match.group(1) if match else None

@st.cache_resource
def load_embeddings():
    """Load embedding model (cached)"""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_data
def get_transcript(video_id):
    """Get YouTube transcript"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([item['text'] for item in transcript_list])
        return transcript
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

def create_vectorstore(transcript, embeddings):
    """Create FAISS vectorstore from transcript"""
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(transcript)
    
    # Create documents
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    # Create vectorstore
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# UI
st.title("🎥 YouTube Video Q&A with RAG")
st.markdown("Ask questions about any YouTube video!")

# Sidebar
with st.sidebar:
    st.header("Settings")
    video_url = st.text_input("YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")
    
    if st.button("Process Video"):
        if video_url:
            video_id = extract_video_id(video_url)
            if video_id:
                with st.spinner("Fetching transcript..."):
                    transcript = get_transcript(video_id)
                    
                if transcript:
                    with st.spinner("Creating vector store..."):
                        embeddings = load_embeddings()
                        st.session_state.vectorstore = create_vectorstore(transcript, embeddings)
                        st.session_state.video_processed = True
                    st.success("✅ Video processed! Ask questions below.")
            else:
                st.error("Invalid YouTube URL")
        else:
            st.warning("Please enter a YouTube URL")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.session_state.get("video_processed"):
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if question := st.chat_input("Ask a question about the video"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Create QA chain
                llm = ChatGoogleGenerativeAI(
                    model="gemini-pro",
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.3
                )
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vectorstore.as_retriever(
                        search_kwargs={"k": 3}
                    )
                )
                
                response = qa_chain.run(question)
                st.markdown(response)
                
                # Add assistant message
                st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("👈 Enter a YouTube URL in the sidebar to get started!")
