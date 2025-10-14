# 🤖 YouTube Q&A Chatbot: Gemini-Powered Video Analysis

## 🌟 Project Overview

This project is a sophisticated **Streamlit** web application that transforms any YouTube video transcript into a fully queryable knowledge base. Utilizing the power of **Google's Gemini model** for reasoning and **LangChain** for the Retrieval-Augmented Generation (RAG) pipeline, the application allows users to paste a YouTube URL and immediately start asking detailed questions about the video's content.

The core strength of this application lies in its ability to handle long, unstructured video transcripts by splitting them into manageable chunks and using a highly efficient, locally-run embedding model (`SentenceTransformer`) to find the most relevant context for your query.

### 💡 Key Features

* **YouTube Transcript Extraction:** Automatically fetches transcripts using the `youtube-transcript-api`.
* **Custom Embeddings:** Uses the fast and efficient `all-MiniLM-L6-v2` model from `sentence-transformers` for embedding, ensuring a reliable RAG process.
* **Vector Search (FAISS):** Stores and searches the embedded transcript data locally using the `FAISS` vector store for blazing-fast retrieval.
* **Gemini Integration:** Leverages the power of the Google Gemini model for accurate, context-aware question answering.
* **Streamlit UI:** Provides an intuitive, clean chat interface for a seamless user experience.

---

## 🛠️ Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

You will need the following installed:

* **Python 3.10+**
* **`uv`** (or `pip`) for dependency management.

### Step 1: Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd your-project-directory