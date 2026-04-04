import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredMarkdownLoader, Docx2txtLoader,
    UnstructuredPowerPointLoader, UnstructuredExcelLoader,
    TextLoader, CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="◈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stApp"] {
    background: #0f1015 !important;
    color: #f0f0f5 !important;
    font-family: 'Outfit', sans-serif !important;
}

/* Drifting background orbs */
[data-testid="stApp"]::before {
    content: '';
    position: fixed;
    top: -20%; left: -10%;
    width: 60vw; height: 60vw;
    background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
    animation: drift 20s ease-in-out infinite alternate;
}
[data-testid="stApp"]::after {
    content: '';
    position: fixed;
    bottom: -20%; right: -10%;
    width: 50vw; height: 50vw;
    background: radial-gradient(circle, rgba(236,72,153,0.08) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
    animation: drift 25s ease-in-out infinite alternate-reverse;
}
@keyframes drift {
    0% { transform: translate(0, 0) scale(1); }
    100% { transform: translate(5%, 10%) scale(1.1); }
}

[data-testid="stVerticalBlock"] { position: relative; z-index: 1; }

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container {
    max-width: 800px !important;
    padding: 2.5rem 1.5rem 8rem !important;
}

/* Header */
.rag-header {
    text-align: center;
    padding: 2rem 0 2.5rem;
    position: relative;
}
.rag-header .badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    color: #818cf8;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.25);
    padding: 0.4rem 1.2rem;
    border-radius: 100px;
    margin-bottom: 1.5rem;
    text-transform: uppercase;
    box-shadow: 0 0 20px rgba(99,102,241,0.1);
}
.rag-header h1 {
    font-size: clamp(2.5rem, 6vw, 3.8rem);
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin: 0 0 0.8rem;
    background: linear-gradient(135deg, #ffffff 0%, #c7d2fe 50%, #fbcfe8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.rag-header p {
    font-size: 1.1rem;
    color: #94a3b8;
    font-weight: 300;
    margin: 0;
}

/* Status pill */
.status-pill {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    background: rgba(15,23,42,0.6);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 100px;
    padding: 0.5rem 1.2rem;
    width: fit-content;
    margin: 0 auto 2.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #cbd5e1;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
}
.status-dot {
    width: 8px; height: 8px;
    background: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 10px rgba(16,185,129,0.5);
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(0.85); }
}

/* Glassmorphism Chat messages */
@keyframes slideUp {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.3rem 0 !important;
    animation: slideUp 0.4s ease-out forwards;
}

.user-bubble {
    background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(79,70,229,0.12));
    backdrop-filter: blur(12px);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px 20px 4px 20px;
    padding: 1rem 1.4rem;
    margin: 0.5rem 0 0.5rem 3rem;
    font-size: 1.05rem;
    line-height: 1.6;
    color: #ffffff;
    box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
}

.assistant-bubble {
    background: rgba(30,41,59,0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px 20px 20px 4px;
    padding: 1rem 1.4rem;
    margin: 0.5rem 3rem 0.5rem 0;
    font-size: 1.05rem;
    line-height: 1.6;
    color: #f1f5f9;
    box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
}

.source-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    color: #818cf8;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 6px;
    padding: 0.25rem 0.6rem;
    margin-top: 0.8rem;
    margin-right: 0.4rem;
    transition: all 0.2s;
}
.source-tag:hover {
    background: rgba(99,102,241,0.25);
    transform: translateY(-2px);
}

/* Avatar icons */
[data-testid="stChatMessage"] [data-testid="stImage"],
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"],
[data-testid="stIcon"] { display: none !important; }

/* Glowing Chat input */
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 0 !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: min(800px, 100vw) !important;
    padding: 1.5rem !important;
    background: linear-gradient(to top, rgba(15,16,21,0.95) 60%, transparent) !important;
    backdrop-filter: blur(8px) !important;
    z-index: 100 !important;
}

[data-testid="stChatInput"] textarea {
    background: rgba(30,41,59,0.85) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 16px !important;
    color: #ffffff !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.05rem !important;
    padding: 0.85rem 1.2rem !important;
    overflow-y: hidden !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
    transition: all 0.3s ease !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(129,140,248,0.7) !important;
    background: rgba(30,41,59,1) !important;
    box-shadow: 0 0 0 4px rgba(129,140,248,0.15), 0 4px 20px rgba(0,0,0,0.3) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #64748b !important;
}

/* Welcome cards with 3D hover */
.welcome-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.25rem;
    margin: 2rem 0;
}
.welcome-card {
    background: rgba(30,41,59,0.4);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.5rem;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.welcome-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
    opacity: 0;
    transition: opacity 0.3s;
}
.welcome-card:hover {
    background: rgba(30,41,59,0.8);
    border-color: rgba(129,140,248,0.4);
    transform: translateY(-5px);
    box-shadow: 0 15px 30px -10px rgba(99,102,241,0.25);
}
.welcome-card:hover::before { opacity: 1; }

.welcome-card .wc-icon {
    font-size: 1.8rem;
    margin-bottom: 1rem;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));
}
.welcome-card .wc-text {
    font-size: 0.95rem;
    color: #e2e8f0;
    font-weight: 400;
    line-height: 1.5;
}

/* File Uploader styling */
[data-testid="stExpander"] {
    background: rgba(30,41,59,0.3) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(10px);
    transition: border-color 0.3s;
}
[data-testid="stExpander"]:hover {
    border-color: rgba(129,140,248,0.3) !important;
}
[data-testid="stExpander"] summary {
    font-weight: 500 !important;
    color: #f8fafc !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.1rem !important;
}

/* Success/info alerts */
[data-testid="stAlert"] {
    background: rgba(16,185,129,0.1) !important;
    border: 1px solid rgba(16,185,129,0.2) !important;
    border-radius: 12px !important;
    color: #34d399 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* Spinner & Progress bar */
[data-testid="stSpinner"] { color: #818cf8 !important; }
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #6366f1, #ec4899) !important;
    box-shadow: 0 0 10px rgba(236,72,153,0.5) !important;
}
/* User Custom Tweak */
.st-emotion-cache-1f3w014 {
    vertical-align: middle;
    overflow: hidden;
    color: inherit;
    fill: currentcolor;
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    font-size: 1.5rem;
    width: 1.5rem;
    height: 2.0rem !important; /* height adjusted */
    flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
    <div class="badge">◈ RAG · Retrieval Augmented Generation</div>
    <h1>Knowledge<br>Assistant</h1>
    <p>Ask anything about your documents</p>
</div>
""", unsafe_allow_html=True)

# ── Loaders ───────────────────────────────────────────────────
LOADERS = {
    ".pdf":  PyPDFLoader,
    ".md":   UnstructuredMarkdownLoader,
    ".docx": Docx2txtLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".txt":  TextLoader,
    ".csv":  CSVLoader,
}

# ── Functions ─────────────────────────────────────────────────
def load_documents(docs_dir="docs"):
    documents = []
    if not os.path.exists(docs_dir):
        return documents
    for filename in os.listdir(docs_dir):
        ext = os.path.splitext(filename)[1].lower()
        path = os.path.join(docs_dir, filename)
        if ext in LOADERS:
            try:
                loader = LOADERS[ext](path)
                documents.extend(loader.load())
            except Exception as e:
                st.warning(f"Could not load {filename}: {e}")
    return documents

def build_vectorstore(docs_dir="docs", store_path="vectorstore"):
    documents = load_documents(docs_dir)
    if not documents:
        # Create dummy vectorstore if there are no documents
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(["No documents have been uploaded yet. Please upload documents."], embeddings)
        vectorstore.save_local(store_path)
        return vectorstore

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    batch_size = 20
    vectorstore = None
    total_batches = (len(chunks) - 1) // batch_size + 1
    progress = st.progress(0, text="Embedding documents...")
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        success = False
        retries = 0
        while not success and retries < 3:
            try:
                progress.progress(batch_num / total_batches, text=f"Embedding batch {batch_num} of {total_batches}...")
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    batch_store = FAISS.from_documents(batch, embeddings)
                    vectorstore.merge_from(batch_store)
                success = True
                time.sleep(1) # Small pause to help with rate limits
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "quota" in err_msg or "exhausted" in err_msg:
                    retries += 1
                    if retries >= 3:
                        st.error(f"API Rate Limit/Quota Exhausted after 3 retries. Wait a minute and try again. Error: {e}")
                        raise e
                    progress.progress(batch_num / total_batches, text=f"API rate limit hit. Waiting 40s before resuming... ({batch_num}/{total_batches}, Retry {retries}/3)")
                    time.sleep(40)
                else:
                    st.error(f"Unexpected Error: {e}")
                    raise e
                    
    vectorstore.save_local(store_path)
    progress.progress(1.0, text="Knowledge base ready!")
    return vectorstore

def load_vectorstore(store_path="vectorstore"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)

def build_qa_chain(vectorstore):
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True
    )

# ── Session state ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = set()

store_path = "vectorstore"

# ── Document Management Section ─────────────────────────────────────
with st.expander("📁 Manage Documents", expanded=False):
    st.write("Upload PDF, MD, DOCX, PPTX, XLSX, TXT, or CSV files")
    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        type=[k.lstrip('.') for k in LOADERS.keys()],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        os.makedirs("docs", exist_ok=True)
        for uploaded_file in uploaded_files:
            file_path = os.path.join("docs", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved: {uploaded_file.name}")

    if os.path.exists("docs") and os.listdir("docs"):
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("#### Indexed Files")
        for file in os.listdir("docs"):
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                st.write(f"📄 {file}")
            with col2:
                if st.button("🗑️", key=f"del_{file}", help=f"Delete {file}"):
                    os.remove(os.path.join("docs", file))
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Rebuild Knowledge Base", type="primary", use_container_width=True):
        with st.spinner("Rebuilding knowledge base with your current documents..."):
            vectorstore = build_vectorstore(store_path=store_path)
            st.session_state.qa_chain = build_qa_chain(vectorstore)
            if os.path.exists("docs"):
                st.session_state.docs_loaded = set(os.listdir("docs"))
            else:
                st.session_state.docs_loaded = set()
        st.success("Knowledge base completely updated!")

# ── Load knowledge base ────────────────────────────────────────
st.session_state.docs_loaded = set(os.listdir("docs")) if os.path.exists("docs") else set()
if st.session_state.qa_chain is None:
    if os.path.exists(os.path.join(store_path, "index.faiss")):
        with st.spinner("Loading knowledge base..."):
            vectorstore = load_vectorstore(store_path)
        st.session_state.qa_chain = build_qa_chain(vectorstore)
    else:
        with st.spinner("Building knowledge base from docs/ folder..."):
            vectorstore = build_vectorstore(store_path=store_path)
        st.session_state.qa_chain = build_qa_chain(vectorstore)

# ── Status pill ───────────────────────────────────────────────
doc_count = len([f for f in os.listdir("docs") if os.path.splitext(f)[1].lower() in LOADERS]) if os.path.exists("docs") else 0
st.markdown(f"""
<div class="status-pill">
    <div class="status-dot"></div>
    {doc_count} document{"s" if doc_count != 1 else ""} indexed · ready
</div>
""", unsafe_allow_html=True)

# ── Welcome suggestions (show only if no messages) ────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-grid">
        <div class="welcome-card">
            <div class="wc-icon">📌</div>
            <div class="wc-text">Summarize the main points of the document</div>
        </div>
        <div class="welcome-card">
            <div class="wc-icon">🔍</div>
            <div class="wc-text">What are the key findings or conclusions?</div>
        </div>
        <div class="welcome-card">
            <div class="wc-icon">💡</div>
            <div class="wc-text">What technologies or methods are mentioned?</div>
        </div>
        <div class="welcome-card">
            <div class="wc-icon">📊</div>
            <div class="wc-text">What data or results are presented?</div>
        </div>
    </div>
    <hr class="section-divider">
    """, unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="user-bubble">{prompt}</div>', unsafe_allow_html=True)

    with st.spinner(""):
        result = st.session_state.qa_chain.invoke({"question": prompt})
        answer = result["answer"]
        sources = {os.path.basename(doc.metadata.get("source", "unknown")) for doc in result["source_documents"]}
        source_html = "".join(f'<span class="source-tag">📄 {s}</span> ' for s in sources)
        full_html = f'<div class="assistant-bubble">{answer}<br><br>{source_html}</div>'

    st.markdown(full_html, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": f"{answer}\n\n📄 Sources: {', '.join(sources)}"})