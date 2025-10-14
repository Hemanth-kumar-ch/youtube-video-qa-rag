import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from google import genai
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Custom Embeddings class
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False)

    def embed_query(self, text):
        return self.model.encode([text])[0]

# Extract video ID from YouTube URL
def extract_video_id(url):
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'^([0-9A-Za-z_-]{11})$'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Fetch transcript
@st.cache_data
def fetch_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id,languages=['en'])
        transcript = " ".join([item.text for item in transcript_list])
        return transcript
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return None

# Create vector store
@st.cache_resource
def create_vector_store(transcript):
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([transcript])
    
    # Create embeddings
    embedding = SentenceTransformerEmbeddings()
    
    # Build FAISS vector store
    vector_store = FAISS.from_documents(docs, embedding)
    return vector_store

# Format documents
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# Call Gemini LLM
def call_gemini(prompt_text):
    try:
        api_key = os.getenv('Google_api_key')
        if not api_key:
            st.error("Google API key not found. Please set it in your .env file.")
            return "Error: API key not configured"
        
        client = genai.Client(api_key=api_key)
        model = "gemini-2.0-flash-exp"
        response = client.models.generate_content(model=model, contents=prompt_text)
        return response.text
    except Exception as e:
        st.error(f"Error calling Gemini API: {str(e)}")
        return f"Error: {str(e)}"

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="YouTube RAG Q&A",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 YouTube Video Q&A with RAG")
    st.markdown("Ask questions about any YouTube video using AI-powered analysis!")
    
    # Sidebar for video input
    with st.sidebar:
        st.header("📹 Video Settings")
        video_url = st.text_input(
            "Enter YouTube URL or Video ID:",
            placeholder="https://youtube.com/watch?v=... or video_id"
        )
        
        process_button = st.button("Process Video", type="primary")
        
        st.markdown("---")
        st.markdown("### 📝 How to use:")
        st.markdown("""
        1. Enter a YouTube video URL or ID
        2. Click 'Process Video'
        3. Wait for transcript processing
        4. Ask questions about the video
        """)
    
    # Initialize session state
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'video_id' not in st.session_state:
        st.session_state.video_id = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Process video when button is clicked
    if process_button and video_url:
        video_id = extract_video_id(video_url)
        
        if not video_id:
            st.error("Invalid YouTube URL or Video ID")
            return
        
        st.session_state.video_id = video_id
        
        with st.spinner("Fetching transcript..."):
            transcript = fetch_transcript(video_id)
        
        if transcript:
            st.success("✅ Transcript fetched successfully!")
            
            with st.spinner("Creating vector store..."):
                st.session_state.vector_store = create_vector_store(transcript)
            
            st.success("✅ Vector store created! You can now ask questions.")
            st.session_state.chat_history = []  # Reset chat history
    
    # Display video if available
    if st.session_state.video_id:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")
        
        with col2:
            st.info(f"**Video ID:** {st.session_state.video_id}")
            if st.button("Clear Video"):
                st.session_state.vector_store = None
                st.session_state.video_id = None
                st.session_state.chat_history = []
                st.rerun()
    
    # Q&A Section
    if st.session_state.vector_store:
        st.markdown("---")
        st.header("💬 Ask Questions")
        
        # Display chat history
        for i, (q, a) in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**Q{i+1}:** {q}")
                st.markdown(f"**A{i+1}:** {a}")
                st.markdown("---")
        
        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is this video about?",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            ask_button = st.button("Ask", type="primary")
        with col2:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        if ask_button and question:
            with st.spinner("Generating answer..."):
                # Create retriever
                retriever = st.session_state.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                )
                
                # Create prompt
                prompt = PromptTemplate(
                    template="""
                    You are a helpful assistant.
                    Answer ONLY from the provided transcript context.
                    If the context is insufficient, just say you don't know.

                    {context}
                    Question: {question}
                    """,
                    input_variables=['context', 'question']
                )
                
                # Create chain
                llm_runnable = RunnableLambda(call_gemini)
                
                parallel_chain = RunnableParallel({
                    'context': retriever | RunnableLambda(format_docs),
                    'question': RunnablePassthrough()
                })
                
                main_chain = parallel_chain | prompt | llm_runnable | StrOutputParser()
                
                # Get answer
                answer = main_chain.invoke({"question": question})
                
                # Add to chat history
                st.session_state.chat_history.append((question, answer))
                st.rerun()
    
    else:
        st.info("👈 Please enter a YouTube URL in the sidebar and click 'Process Video' to get started!")

if __name__ == "__main__":
    main()