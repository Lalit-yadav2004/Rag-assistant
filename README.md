# ◈ Knowledge Assistant (RAG Pipeline)

A highly polished, dark-mode Retrieval-Augmented Generation (RAG) assistant. This application allows users to seamlessly upload documents and instantly chat with them using local, unlimited embeddings and lightning-fast cloud LLMs.

![RAG Assistant](https://img.shields.io/badge/Status-Complete-success) ![License](https://img.shields.io/badge/License-MIT-blue)

## ✨ Features
- **Premium User Interface:** A stunning Glassmorphism UI built on Streamlit with smooth micro-animations, glowing inputs, and custom `Outfit` & `JetBrains Mono` fonts.
- **Unlimited Document Indexing:** Completely free, open-source local embeddings using `HuggingFace` (`all-MiniLM-L6-v2`) integrated with `sentence-transformers`. No more API rate limits!
- **Multi-Format Support:** Easily ingest PDF, Markdown, DOCX, PPTX, XLSX, TXT, and CSV files directly from the UI.
- **Persistent Local Vector Database:** Efficient semantic search utilizing local `FAISS` storage.
- **Full Document Management:** Delete indexed files or re-build your entire knowledge base securely from the dashboard.
- **Rapid Inference:** Powered by `ChatGroq` (`llama-3.1-8b-instant`) for incredibly fast chat completions with chat memory.

## 🛠️ Technology Stack
*   **Frontend:** [Streamlit](https://streamlit.io/)
*   **Orchestration:** [LangChain](https://python.langchain.com/)
*   **Vector Database:** [FAISS](https://faiss.ai/)
*   **Embeddings:** [HuggingFace](https://huggingface.co/) via `sentence-transformers`
*   **LLM:** [Groq](https://groq.com/) (Llama 3.1)

---

## 🚀 How to Run Locally

### 1. Requirements
Ensure you have Python 3.9+ installed.

### 2. Setup Virtual Environment
Clone the repository and set up your Python virtual environment:
```bash
git clone https://github.com/Lalit-yadav2004/Rag-assitant.git
cd Rag-assitant
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setting up Environment Variables
Create a `.env` file in the root directory and securely add your ChatGroq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Launch the Application
Start the beautiful Streamlit web platform:
```bash
streamlit run app.py
```
*(Alternatively, if you prefer a pure terminal session, run `python main.py`)*

---

## ☁️ Deployment (Streamlit Community Cloud)

This application is perfectly optimized for immediate deployment via **Streamlit Cloud** rather than Vercel (since it requires a persistent Python webserver).

1. Log into [Streamlit Community Cloud](https://share.streamlit.io).
2. Click **New app** and select this repository.
3. Set the **Main file path** to `app.py`.
4. Open the **Advanced settings** and paste your `GROQ_API_KEY` into the Secrets box.
5. Hit **Deploy**!
