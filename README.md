# 🎙️ Voice-Enabled RAG Chatbot (Groq, Pinecone, Streamlit)

This project is a powerful, voice-enabled chatbot that answers questions by combining information from a user-uploaded PDF and live web search results. It leverages a multi-modal RAG (Retrieval-Augmented Generation) pipeline to understand and reason over both text and images within the PDF, providing accurate, cited, and visually-supported answers.

<br>

-----

## ✨ Features

  * **🎙️ Voice Input**: Use your microphone to ask questions, powered by Deepgram's highly accurate speech-to-text API.
  * **📚 Multi-modal RAG**: The system indexes not only the text of a PDF but also analyzes and describes tables, charts, and images using a multi-modal LLM. This allows it to answer complex questions about visual data.
  * **⚡️ Lightning-Fast Responses**: The core text generation is powered by Groq's LPU, providing incredibly low-latency and fast answers.
  * **🔍 Hybrid Search**: The chatbot retrieves context from both your private document library (Pinecone vector DB) and live web search (Serper API) to provide comprehensive answers.
  * **🖼️ Visual Citations**: When an answer is derived from the uploaded PDF, the system displays a visual preview of the exact page, complete with a clickable link to view the document.
  * **💬 Streamlit UI**: A clean, interactive user interface built with Streamlit for a seamless chat experience.

<br>

-----

## 🚀 How It Works

1.  **Ingestion**: A user uploads a PDF. The application processes the document page by page. It extracts text and sends images and tables to a multi-modal LLM (e.g., Groq's `llama-4-scout-17b`) to generate detailed text descriptions.
2.  **Embedding**: All text chunks and image/table descriptions are converted into numerical vector embeddings using a local `SentenceTransformer` model.
3.  **Vector Storage**: These vectors, along with their source metadata (file name, page number), are stored in a Pinecone vector database.
4.  **Retrieval**: When a user asks a question, the query is also embedded. The system performs two parallel searches: a vector search against the Pinecone index (RAG) and a web search via the Serper API.
5.  **Generation**: The retrieved context from both sources is sent to a fast LLM on Groq's LPU (`llama-3.1-8b-instant`). The LLM synthesizes an answer, complete with citations to both the PDF and web sources.
6.  **Citation Display**: The chatbot parses the LLM's response for citations and uses the stored PDF data to render a preview of the relevant page, providing visual proof of the source.

<br>

-----

## 🛠️ Installation and Setup

### **1. Clone the repository**

```bash
git clone https://github.com/your-username/voice-rag-chatbot.git
cd voice-rag-chatbot
```

### **2. Set up a virtual environment**

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### **3. Install the dependencies**

```bash
pip install -r requirements.txt
```

> Note: Ensure you have `groq`, `pinecone-client`, `deepgram-sdk`, `serpapi`, `sentence-transformers`, `streamlit`, `streamlit-mic-recorder`, `PyMuPDF`, `Pillow`, and `python-dotenv` in your `requirements.txt` file.

### **4. Configure API Keys**

Create a `.env` file in the root directory and add your API keys.

```ini
GROQ_API_KEY="your_groq_api_key"
GEMINI_API_KEY="your_gemini_api_key"
PINECONE_API_KEY="your_pinecone_api_key"
DEEPGRAM_API_KEY="your_deepgram_api_key"
SERPER_API_KEY="your_serper_api_key"
```

### **5. Set up Pinecone**

1.  Log in to your Pinecone account and create a new index.
2.  Name the index `voicerag`.
3.  Set the dimension to `384` to match the `all-MiniLM-L6-v2` embedding model.

### **6. Run the application**

```bash
streamlit run app.py
```

<br>

-----

## 🤝 Contributing

Contributions are welcome\! Please open an issue or submit a pull request for any bugs, features, or improvements.
