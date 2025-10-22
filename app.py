import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from google import genai
import os
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# Create a custom embeddings class using SentenceTransformer
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        # Convert texts to embeddings for the vector store
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        # Convert a single query to an embedding
        return self.model.encode([text])[0].tolist()

# Extract the video ID from a YouTube URL
def extract_video_id(url):
    try:
        # If someone just pasted the video ID (11 characters), return it
        if len(url) == 11 and '=' not in url:
            return url
        # Otherwise, extract it from the URL after the "v=" part
        return url.split("=")[1].split("&")[0]
    except:
        return None

# Get the transcript text from a YouTube video
@st.cache_data
def fetch_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id,languages=['en'])
        transcript = " ".join([item.text for item in transcript_list])
        return transcript
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return None
# Create a vector store from the transcript so we can search through it
@st.cache_resource
def create_vector_store(transcript):
    # Break the transcript into smaller chunks (1000 characters each)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(transcript)
    docs = [Document(page_content=chunk) for chunk in chunks]
    embedding = SentenceTransformerEmbeddings()
    
    # Build the searchable vector store using FAISS
    vector_store = FAISS.from_documents(docs, embedding)
    return vector_store

# Format the retrieved documents into a single string
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# Send a prompt to Google's Gemini AI and get the response
def call_gemini(prompt_text):
    try:
            # Get the API key from environment variables
            api_key = os.getenv('Google_api_key')
            if not api_key:
                st.error("Can't find your Google API key. Add it to your .env file!")
                return "Error: API key not set up"
            
            # Connect to Gemini and generate a response
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp", 
                contents=prompt_text
            )
            return response.text
    except Exception as e:
            st.error(f"Gemini API error: {str(e)}")
            return f"Error: {str(e)}"

def main():
    st.set_page_config(
        page_title="YouTube RAG Q&A",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 YouTube Video Q&A with RAG")
    st.markdown("Ask questions about any YouTube video using AI!")
    
    # Sidebar - where users enter the video URL
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
        1. Paste a YouTube video URL or ID
        2. Click 'Process Video'
        3. Wait for the transcript to load
        4. Ask any questions about the video
        """)
    
    # Initialize session state variables (these persist across reruns)
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'video_id' not in st.session_state:
        st.session_state.video_id = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # When user clicks "Process Video"
    if process_button and video_url:
        video_id = extract_video_id(video_url)
        
        # Check if we got a valid video ID
        if not video_id:
            st.error("❌ That doesn't look like a valid YouTube URL or ID")
            return
        
        st.session_state.video_id = video_id
        
        # Fetch the video transcript
        with st.spinner("Getting the video transcript..."):
            transcript = fetch_transcript(video_id)
        
        # If we successfully got the transcript
        if transcript:
            st.success("✅ Got the transcript!")
            
            # Create the vector store for searching
            with st.spinner("Setting up the search system..."):
                st.session_state.vector_store = create_vector_store(transcript)
            
            st.success("✅ Ready! You can now ask questions.")
            st.session_state.chat_history = []  # Clear any old chat history
        else:
            st.error("❌ Couldn't get the transcript. This video might not have captions.")
    
    # Show the video player if we have a video loaded
    if st.session_state.video_id:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Embed the YouTube video
            st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")
        
        with col2:
            st.info(f"**🆔 Video ID:** {st.session_state.video_id}")
            
            # Button to clear everything and start over
            if st.button("🗑️ Clear Video"):
                st.session_state.vector_store = None
                st.session_state.video_id = None
                st.session_state.chat_history = []
                st.rerun()
    
    # Q&A Section - only show if we have a vector store ready
    if st.session_state.vector_store:
        st.markdown("---")
        st.header("💬 Ask Questions")
        
        # Display all previous questions and answers
        if st.session_state.chat_history:
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                with st.container():
                    st.markdown(f"**❓ Question {i+1}:** {question}")
                    st.markdown(f"**💡 Answer:**")
                    st.markdown(answer)
                    st.markdown("---")
        
        # Form for asking new questions (prevents double submissions)
        with st.form(key='question_form', clear_on_submit=True):
            question = st.text_input(
                "Your question:",
                placeholder="e.g., What is this video about?",
                key="question_input"
            )
            
            col1, col2 = st.columns([1, 5])
            with col1:
                ask_button = st.form_submit_button("Ask", type="primary")
            with col2:
                clear_button = st.form_submit_button("Clear Chat")
        
        # Clear the chat history if requested
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()
        
        # Process the question
        if ask_button and question.strip():
            with st.spinner("🤔 Thinking..."):
                # Search the vector store for relevant chunks
                retriever = st.session_state.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}  # Get the top 4 most relevant chunks
                )
                
                # Get the relevant document chunks
                retrieved_docs = retriever.invoke(question)
                context = format_docs(retrieved_docs)
                
                # Create the prompt for Gemini
                prompt = f"""You are a helpful assistant. Answer the question based on the video transcript below.
If you can't find the answer in the transcript, just say you don't know.

Video Transcript:
{context}

Question: {question}

Answer:"""
                
                # Get the answer from Gemini
                answer = call_gemini(prompt)
                
                # Save this Q&A to the chat history
                st.session_state.chat_history.append((question, answer))
                st.rerun()
    
    else:
        # Show this message if no video has been processed yet
        st.info("👈 Enter a YouTube URL in the sidebar and click 'Process Video' to start!")

# Run the app
if __name__ == "__main__":
    main()
