import streamlit as st
import os
import io
import time
import base64
from dotenv import load_dotenv
from PIL import Image

# Groq API for LLM Generation
from groq import Groq
from pinecone import Pinecone
from deepgram import DeepgramClient
from serpapi import GoogleSearch
from sentence_transformers import SentenceTransformer

# PDF Processing Library
import fitz # PyMuPDF
from tabulate import tabulate # For formatting tables

# Streamlit Component for Microphone Input
from streamlit_mic_recorder import mic_recorder
import streamlit.components.v1 as components

# 1. CONFIGURATION AND INITIALIZATION 

load_dotenv()

# API Keys and Environment Variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
PINECONE_INDEX_NAME = "voicerag" 

# Models for each task
GROQ_MODEL = 'llama-3.1-8b-instant' 
# This model will be used for the final response generation.
# It is a highly optimized text-generation model.
GROQ_VISION_MODEL = 'meta-llama/llama-4-scout-17b-16e-instruct'
# This Groq multimodal model is used for describing images/tables in the PDF.
EMBEDDING_MODEL_LOCAL = 'all-MiniLM-L6-v2' 
EMBEDDING_DIMENSION = 384

@st.cache_resource
def load_embedding_model(model_name):
    """Loads the SentenceTransformer model only once."""
    st.info(f"Loading embedding model: {model_name}...")
    return SentenceTransformer(model_name)

def initialize_services():
    """Initializes the Groq, Pinecone, and Deepgram services."""
    if "groq_client" not in st.session_state:
        try:
            # Initialize Groq client
            st.session_state.groq_client = Groq(api_key=GROQ_API_KEY)
            
            # Groq client will also be used for vision (multimodal) queries

            # Initialize Pinecone
            st.session_state.pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
            
            # Initialize Deepgram Client
            st.session_state.deepgram_client = DeepgramClient(DEEPGRAM_API_KEY)
            
            # Load Embedding Model (cached resource)
            st.session_state.embedding_model = load_embedding_model(EMBEDDING_MODEL_LOCAL)

            # Check Pinecone Index and Dimension
            available_indexes_response = st.session_state.pinecone_client.list_indexes()
            available_indexes = available_indexes_response.names()
            
            if PINECONE_INDEX_NAME not in available_indexes:
                st.error(f"Pinecone Index '{PINECONE_INDEX_NAME}' not found. Please create it first.")
                st.stop()
            else:
                index_stats = st.session_state.pinecone_client.describe_index(PINECONE_INDEX_NAME)
                if index_stats.dimension != EMBEDDING_DIMENSION:
                    st.error(f"Index dimension mismatch! Expected {EMBEDDING_DIMENSION} (for {EMBEDDING_MODEL_LOCAL}), found {index_stats.dimension}. Please delete and re-create your index.")
                    st.stop()
                    
                st.session_state.pinecone_index = st.session_state.pinecone_client.Index(PINECONE_INDEX_NAME)
            
            # Initialize mic recorder key state
            if 'mic_recorder_key' not in st.session_state:
                st.session_state.mic_recorder_key = 0
                
            # Store uploaded PDF bytes
            if 'uploaded_pdf_bytes' not in st.session_state:
                st.session_state.uploaded_pdf_bytes = None

        except Exception as e:
            st.error(f"Initialization Error: Check API keys, index name, or network. Details: {e}")
            st.stop()

# Run initialization
if GROQ_API_KEY and PINECONE_API_KEY and DEEPGRAM_API_KEY and SERPER_API_KEY:
    initialize_services()
else:
    st.error("Please set required API keys (GROQ, PINECONE, DEEPGRAM, SERPER) in your .env file to proceed.")
    st.stop()
# 2. EMBEDDING AND RAG UTILITY FUNCTIONS (Hugging Face / Pinecone)

@st.cache_data(show_spinner=False)
def get_embedding(text):
    """Generates a vector embedding for a given text using the local SentenceTransformer model."""
    try:
        embedding = st.session_state.embedding_model.encode(text)
        return embedding.tolist()
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

def process_multi_modal_content(page, file_name, page_num):
    chunks = []
    
    # 1. Process Text
    page_text = page.get_text("text")
    text_chunks = page_text.split('\n\n')
    for i, chunk in enumerate(text_chunks):
        if len(chunk) > 20:
            vector_id = f"{file_name}_page{page_num+1}_text_chunk{i}"
            embedding = get_embedding(chunk)
            if embedding:
                chunks.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': {
                        'text': chunk,
                        'source': file_name,
                        'page': page_num + 1,
                        'type': 'text'
                    }
                })

    # 2. Process Tables and Images using Groq Vision Model
    try:
        # Render at reasonable DPI and compress to stay under Groq limits (<=4MB base64 payload)
        page_pixmap = page.get_pixmap(dpi=180)
        img = Image.frombytes("RGB", [page_pixmap.width, page_pixmap.height], page_pixmap.samples)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70, optimize=True)
        img_bytes = buf.getvalue()

        # Prepare base64 data URL for Groq multimodal input
        b64_image = base64.b64encode(img_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64_image}"

        groq_client = st.session_state.groq_client
        completion = groq_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that extracts structured descriptions "
                        "of visual content from PDF pages. Focus on tables, charts, figures, "
                        "and key visual insights. Respond concisely in plain text."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyze this image from page {page_num + 1}. Describe tables, figures, charts, and images in detail."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )
        image_description = completion.choices[0].message.content
        
        vector_id = f"{file_name}_page{page_num+1}_visual_desc"
        embedding = get_embedding(image_description)
        if embedding:
            chunks.append({
                'id': vector_id,
                'values': embedding,
                'metadata': {
                    'text': image_description,
                    'source': file_name,
                    'page': page_num + 1,
                    'type': 'visual'
                }
            })
    except Exception as e:
        st.warning(f"Could not process tables/images for page {page_num + 1}: {e}")
        pass
            
    return chunks

def process_pdf_and_upsert(uploaded_file):
    """
    Processes the uploaded PDF, extracts text and images, generates embeddings, 
    and upserts them to Pinecone.
    """
    if not uploaded_file:
        return []

    status_message = st.status("Processing Document...", expanded=True)
    status_message.write(f"Reading file: {uploaded_file.name}")
    
    doc = fitz.open(stream=st.session_state.uploaded_pdf_bytes, filetype="pdf")
    all_chunks = []
    
    try:
        for page_num, page in enumerate(doc):
            all_chunks.extend(process_multi_modal_content(page, uploaded_file.name, page_num))
            time.sleep(0.05) # Small delay to prevent network strain

        status_message.write(f"Generated {len(all_chunks)} fragments. Upserting to Pinecone...")
        for i in range(0, len(all_chunks), 100):
            batch = all_chunks[i:i + 100]
            st.session_state.pinecone_index.upsert(vectors=batch)
            status_message.write(f"Upserted batch {int(i/100) + 1}...")

        status_message.success(f"Successfully processed and indexed {len(all_chunks)} chunks for RAG.")
        return all_chunks

    except Exception as e:
        status_message.error(f"Error during PDF processing or Pinecone upsert: {e}")
        return []
# 3. SEARCH AND ORCHESTRATION FUNCTIONS (Groq API) ---

def get_rag_context(query, index):
    """Retrieves context from Pinecone (RAG)."""
    
    query_embedding = get_embedding(query)
    
    if query_embedding is None:
        return "No RAG context available due to embedding generation failure.", []

    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )
    
    def _safe_page(value):
        try:
            return int(round(float(value)))
        except Exception:
            return value

    rag_context = "\n---\n".join([
        f"Source: {match.metadata['source']} (Page {_safe_page(match.metadata['page'])})\nContent: {match.metadata['text']}" 
        for match in results.matches
    ])

    sources = [(m.metadata['source'], _safe_page(m.metadata['page'])) for m in results.matches]
    
    return rag_context, sources

def get_web_context(query):
    """Retrieves context from Web Search (Serper API)."""
    if not SERPER_API_KEY:
        return "", []
        
    search = GoogleSearch({"q": query, "api_key": SERPER_API_KEY})
    results = search.get_json()
    
    search_context = ""
    web_sources = []
    
    if "organic_results" in results:
        for result in results["organic_results"][:3]: 
            search_context += f"Source: {result['title']} ({result['link']})\nContent: {result['snippet']}\n---\n"
            web_sources.append({
                'title': result.get('title'),
                'link': result.get('link'),
                'snippet': result.get('snippet')
            })
            
    return search_context, web_sources

def get_llm_response(query, rag_context, web_context):
    """Generates the final response using the Groq LLM."""
    full_context = f"--- DOCUMENT CONTEXT (RAG) ---\n{rag_context}\n\n--- WEB SEARCH CONTEXT ---\n{web_context}"

    system_instruction = (
        "You are an intelligent RAG + Web chatbot. Use BOTH sources when available. "
        "Answer concisely and accurately using the provided DOCUMENT CONTEXT (RAG) and WEB SEARCH CONTEXT. "
        "Always integrate relevant points from both contexts when present; do not ignore either unless it is clearly irrelevant. "
        "Citations: Use [D1], [D2], ... for document chunks; and [W1], [W2], ... for web results. "
        "At the end, output two distinct sections: 'Document Citations' and 'Web Citations'. "
        "In 'Document Citations', list each as '[-] [Dx] [File Name] (Page [Number])'. "
        "In 'Web Citations', list each as '[-] [Wx] [Title] ([Link])'. "
        "Do not duplicate the same citation. Only cite items you used. "
        "If neither context contains the answer, say so. "
        "Format in markdown. Use integer PDF page numbers only (e.g., Page 4)."
    )
    
    try:
        chat_completion = st.session_state.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Context: {full_context}\n\nQuestion: {query}"}
            ],
            model=GROQ_MODEL,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred during LLM generation (Groq API): {e}"

def render_pdf_page_image(pdf_file_bytes, page_number):
    """
    Renders a specific page of a PDF as a PNG image in a bytes buffer.
    
    Args:
        pdf_file_bytes (bytes): The raw bytes of the PDF file.
        page_number (int): The 1-based index of the page to render.

    Returns:
        io.BytesIO: A bytes buffer containing the PNG image data, or None on error.
    """
    try:
        pdf_stream = io.BytesIO(pdf_file_bytes)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        if page_number < 1 or page_number > len(doc):
            st.error(f"Page number {page_number} is out of range.")
            return None
        
        page = doc.load_page(page_number - 1)
        pix = page.get_pixmap(dpi=200)
        
        img_buffer = io.BytesIO()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        
        return img_buffer
    except Exception as e:
        st.error(f"Error rendering PDF page: {e}")
        return None

def render_pdf_viewer(pdf_bytes, page_number, height=600):
    """
    Embeds a PDF viewer focused on a given page using an inline iframe.
    Note: This uses a base64 data URL; large PDFs can be heavy to embed.
    """
    try:
        b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        # Use #page= to jump to the specific page
        html = f"""
        <iframe src='data:application/pdf;base64,{b64_pdf}#page={page_number}'
                width='100%' height='{height}' style='border:1px solid #444'></iframe>
        """
        components.html(html, height=height)
    except Exception as e:
        st.warning(f"Could not embed PDF viewer: {e}")
# 4. DEEPGRAM VOICE TRANSCRIPTION 

def transcribe_audio(audio_data):
    """Processes recorded audio chunk using Deepgram's REST transcription API."""
    if not DEEPGRAM_API_KEY:
        return "Deepgram API key not set."
    
    audio_bytes = audio_data.get('bytes')
    if not audio_bytes:
        return "Could not transcribe audio."

    source = {
        "buffer": audio_bytes,
        "mimetype": "audio/wav"
    }

    try:
        if 'deepgram_client' not in st.session_state:
            st.error("Deepgram client not initialized.")
            return "Could not transcribe audio."

        response = st.session_state.deepgram_client.listen.rest.v("1").transcribe_file(
            source,
            {"smart_format": True, "punctuate": True}
        )

        transcript = response.results['channels'][0]['alternatives'][0]['transcript']
        return transcript
            
    except Exception as e:
        st.error(f"Deepgram Transcription Error: {e}")
        return "Could not transcribe audio."
# 5. STREAMLIT APPLICATION LAYOUT ---

# Configure Streamlit page
st.set_page_config(
    page_title="Voice RAG Chatbot (Groq + Pinecone)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎙️ Voice-Enabled RAG Chatbot")
st.caption("Powered by Groq (Vision + Text), Local Embedding, Deepgram, and Serper API")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello! Please upload a document to begin RAG, or ask a general question."})

# Sidebar for controls
with st.sidebar:
    st.header("1. Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a PDF Report (Max 50MB)",
        type="pdf",
        accept_multiple_files=False,
        key="pdf_uploader"
    )
    
    if uploaded_file and st.button("Index Document"):
        with st.spinner("Indexing document..."):
            st.session_state.uploaded_pdf_bytes = uploaded_file.getvalue()
            process_pdf_and_upsert(uploaded_file)
            st.session_state.pdf_indexed_name = uploaded_file.name
        
        st.session_state.mic_recorder_key += 1
        
    st.markdown("---")
    st.header("2. Voice Input (Deepgram)")
    
    audio_data = mic_recorder(
        start_prompt="Start Recording",
        stop_prompt="Stop Recording",
        just_once=True,
        key=f"recorder_{st.session_state.mic_recorder_key}"
    )

    if audio_data and audio_data.get('bytes') and len(audio_data.get('bytes')) > 100:
        st.info("Transcribing audio...")
        with st.spinner("Transcribing..."):
            transcript = transcribe_audio(audio_data)
        st.session_state.voice_transcript = transcript

    st.markdown("---")
    st.header("3. Status")
    
    if "pinecone_index" in st.session_state:
        try:
            stats = st.session_state.pinecone_index.describe_index_stats()
            st.metric("Total Vectors", stats['total_vector_count'])
            st.metric("Indexed Doc", st.session_state.get('pdf_indexed_name', 'None'))
            st.metric("Embedding Dim.", EMBEDDING_DIMENSION)
        except Exception as e:
            st.error(f"Could not load index stats: {e}")

    st.markdown("---")
    st.header("4. Citations")
    # Display latest RAG sources gathered for the last answer
    pdf_sources = st.session_state.get('last_pdf_sources', [])
    if pdf_sources and st.session_state.get('uploaded_pdf_bytes'):
        for i, (src_name, page_no) in enumerate(pdf_sources, start=1):
            cols = st.columns([1, 3])
            with cols[0]:
                if st.button(f"#{i}", key=f"cite_btn_{i}"):
                    try:
                        st.session_state.selected_pdf_page = int(round(float(page_no)))
                    except Exception:
                        st.session_state.selected_pdf_page = page_no
            with cols[1]:
                st.caption(f"{src_name} (Page {page_no})")
    else:
        st.caption("No PDF citations yet.")

# Main chat display loop
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("image_buffer"):
            st.image(message["image_buffer"], caption=f"Citation Source: Page {message['page_number']}", width=400)

# Always respond to a selected citation from the sidebar
if st.session_state.get('selected_pdf_page') and st.session_state.get('uploaded_pdf_bytes'):
    try:
        sel_page = int(round(float(st.session_state.selected_pdf_page)))
    except Exception:
        sel_page = st.session_state.selected_pdf_page
    with st.expander(f"Citation Preview: Page {sel_page}", expanded=True):
        img_buf = render_pdf_page_image(st.session_state.uploaded_pdf_bytes, sel_page)
        if img_buf:
            st.image(img_buf, caption=f"Page {sel_page}")
        render_pdf_viewer(st.session_state.uploaded_pdf_bytes, sel_page, height=500)
            
# Handle transcribed voice input
if "voice_transcript" in st.session_state and st.session_state.voice_transcript:
    user_prompt = st.session_state.voice_transcript
    del st.session_state.voice_transcript
    
    with st.chat_message("user"):
        st.markdown(f"**🗣️ Transcribed:** *{user_prompt}*")
    st.session_state.messages.append({"role": "user", "content": f"**🗣️ Transcribed:** *{user_prompt}*"})
    
    with st.chat_message("assistant"):
        with st.spinner(f"Thinking... (Combining RAG and Web Search for: {user_prompt})"):
            rag_context, rag_sources = get_rag_context(user_prompt, st.session_state.pinecone_index)
            web_context, web_sources = get_web_context(user_prompt)
            final_answer = get_llm_response(user_prompt, rag_context, web_context)
            
            st.markdown(final_answer)

            # Store sources for sidebar citations
            st.session_state.last_pdf_sources = rag_sources

            import re
            pdf_citation_match = re.search(r"\(Page (\d+)\)", final_answer)
            if pdf_citation_match and st.session_state.uploaded_pdf_bytes:
                page_number = int(pdf_citation_match.group(1))
                image_buffer = render_pdf_page_image(st.session_state.uploaded_pdf_bytes, page_number)
                if image_buffer:
                    st.image(image_buffer, caption=f"Citation Source: Page {page_number}", width=400)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": final_answer,
                        "image_buffer": image_buffer,
                        "page_number": page_number
                    })
                else:
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
            else:
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

    # If user clicked a citation in the sidebar, show page image and viewer
    if st.session_state.get('selected_pdf_page') and st.session_state.get('uploaded_pdf_bytes'):
        sel_page = st.session_state.selected_pdf_page
        with st.expander(f"Citation Preview: Page {sel_page}", expanded=True):
            img_buf = render_pdf_page_image(st.session_state.uploaded_pdf_bytes, sel_page)
            if img_buf:
                st.image(img_buf, caption=f"Page {sel_page}")
            render_pdf_viewer(st.session_state.uploaded_pdf_bytes, sel_page, height=500)

# Handle regular text input
if prompt := st.chat_input("Type your question or use the mic in the sidebar"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner(f"Thinking... (Combining RAG and Web Search for: {prompt})"):
            rag_context, rag_sources = get_rag_context(prompt, st.session_state.pinecone_index)
            web_context, web_sources = get_web_context(prompt)
            final_answer = get_llm_response(prompt, rag_context, web_context)
            
            st.markdown(final_answer)

            # Store sources for sidebar citations
            st.session_state.last_pdf_sources = rag_sources

            import re
            pdf_citation_match = re.search(r"\(Page (\d+)\)", final_answer)
            if pdf_citation_match and st.session_state.uploaded_pdf_bytes:
                page_number = int(pdf_citation_match.group(1))
                image_buffer = render_pdf_page_image(st.session_state.uploaded_pdf_bytes, page_number)
                if image_buffer:
                    st.image(image_buffer, caption=f"Citation Source: Page {page_number}", width=400)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": final_answer,
                        "image_buffer": image_buffer,
                        "page_number": page_number
                    })
                else:
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
            else:
                st.session_state.messages.append({"role": "assistant", "content": final_answer})