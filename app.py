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
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Custom Embeddings class
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

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

# Fetch video metadata (channel name, title)
@st.cache_data
def fetch_video_metadata(video_id):
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title and channel name
        title = soup.find('meta', property='og:title')
        title = title['content'] if title else "Unknown Title"
        
        # Try to find channel name
        channel = None
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string and '"author"' in script.string:
                match = re.search(r'"author":"([^"]+)"', script.string)
                if match:
                    channel = match.group(1)
                    break
        
        return {
            'title': title,
            'channel': channel or "Unknown Channel",
            'url': url
        }
    except Exception as e:
        st.warning(f"Could not fetch video metadata: {str(e)}")
        return {
            'title': "Unknown Title",
            'channel': "Unknown Channel",
            'url': f"https://www.youtube.com/watch?v={video_id}"
        }

# Fetch transcript
@st.cache_data
def fetch_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        transcript = " ".join([item['text'] for item in transcript_list])
        return transcript
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return None

# Create vector store
@st.cache_resource
def create_vector_store(transcript, metadata):
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    # Add metadata to each chunk
    chunks = splitter.split_text(transcript)
    docs = [
        Document(
            page_content=chunk,
            metadata={
                'channel': metadata['channel'],
                'title': metadata['title'],
                'source': 'youtube_transcript'
            }
        ) for chunk in chunks
    ]
    
    # Create embeddings
    embedding = SentenceTransformerEmbeddings()
    
    # Build FAISS vector store
    vector_store = FAISS.from_documents(docs, embedding)
    return vector_store

# Format documents with metadata
def format_docs(retrieved_docs):
    formatted = []
    for doc in retrieved_docs:
        formatted.append(doc.page_content)
    return "\n\n".join(formatted)

# Call Gemini LLM with properly formatted prompt
def call_gemini(formatted_prompt):
    try:
        api_key = os.getenv('Google_api_key')
        if not api_key:
            st.error("Google API key not found. Please set it in your .env file.")
            return "Error: API key not configured"
        
        client = genai.Client(api_key=api_key)
        model = "gemini-2.0-flash-exp"
        response = client.models.generate_content(model=model, contents=formatted_prompt)
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
    if 'video_metadata' not in st.session_state:
        st.session_state.video_metadata = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    # Process video when button is clicked
    if process_button and video_url:
        video_id = extract_video_id(video_url)
        
        if not video_id:
            st.error("❌ Invalid YouTube URL or Video ID")
            return
        
        st.session_state.video_id = video_id
        st.session_state.processing = True
        
        # Fetch metadata
        with st.spinner("Fetching video metadata..."):
            metadata = fetch_video_metadata(video_id)
            st.session_state.video_metadata = metadata
        
        with st.spinner("Fetching transcript..."):
            transcript = fetch_transcript(video_id)
        
        if transcript:
            st.success("✅ Transcript fetched successfully!")
            
            with st.spinner("Creating vector store..."):
                st.session_state.vector_store = create_vector_store(transcript, metadata)
            
            st.success("✅ Vector store created! You can now ask questions.")
            st.session_state.chat_history = []  # Reset chat history
        else:
            st.error("❌ Failed to fetch transcript. The video might not have captions available.")
        
        st.session_state.processing = False
    
    # Display video if available
    if st.session_state.video_id:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")
        
        with col2:
            if st.session_state.video_metadata:
                st.info(f"**📺 Channel:** {st.session_state.video_metadata['channel']}")
                st.info(f"**🎬 Title:** {st.session_state.video_metadata['title']}")
                st.info(f"**🆔 Video ID:** {st.session_state.video_id}")
            else:
                st.info(f"**Video ID:** {st.session_state.video_id}")
            
            if st.button("🗑️ Clear Video"):
                st.session_state.vector_store = None
                st.session_state.video_id = None
                st.session_state.chat_history = []
                st.session_state.video_metadata = None
                st.rerun()
    
    # Q&A Section
    if st.session_state.vector_store:
        st.markdown("---")
        st.header("💬 Ask Questions")
        
        # Display chat history
        if st.session_state.chat_history:
            for i, (q, a) in enumerate(st.session_state.chat_history):
                with st.container():
                    st.markdown(f"**❓ Question {i+1}:** {q}")
                    with st.container():
                        st.markdown(f"**💡 Answer:**")
                        st.markdown(a)
                    st.markdown("---")
        
        # Question input form to prevent double submission
        with st.form(key='question_form', clear_on_submit=True):
            question = st.text_input(
                "Enter your question:",
                placeholder="e.g., What is this video about? Who is the YouTuber?",
                key="question_input"
            )
            
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                ask_button = st.form_submit_button("Ask", type="primary")
            with col2:
                clear_button = st.form_submit_button("Clear Chat")
        
        # Handle clear chat button
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()
        
        # Handle ask button
        if ask_button and question.strip():
            with st.spinner("🤔 Generating answer..."):
                # Create retriever
                retriever = st.session_state.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                )
                
                # Get relevant documents
                retrieved_docs = retriever.invoke(question)
                context = format_docs(retrieved_docs)
                
                # Add metadata context
                metadata_context = f"""
Video Information:
- Channel/YouTuber: {st.session_state.video_metadata['channel']}
- Video Title: {st.session_state.video_metadata['title']}

Transcript Content:
{context}
"""
                
                # Create prompt template
                prompt_template = """You are a helpful assistant analyzing a YouTube video.

Use the following video information and transcript to answer the question.
If asked about the YouTuber or channel name, use the metadata provided.
Answer based ONLY on the information provided.
If you cannot find the answer in the context, say "I don't have enough information to answer that."

{context}

Question: {question}

Answer:"""
                
                # Format the prompt
                formatted_prompt = prompt_template.format(
                    context=metadata_context,
                    question=question
                )
                
                # Get answer from Gemini
                answer = call_gemini(formatted_prompt)
                
                # Add to chat history (only once)
                st.session_state.chat_history.append((question, answer))
                st.rerun()
    
    else:
        st.info("👈 Please enter a YouTube URL in the sidebar and click 'Process Video' to get started!")

if __name__ == "__main__":
    main()
