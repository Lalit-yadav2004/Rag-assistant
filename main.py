import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

def load_documents(docs_dir="docs"):
    from langchain_community.document_loaders import (
        PyPDFLoader,
        UnstructuredMarkdownLoader,
        Docx2txtLoader,
        UnstructuredPowerPointLoader,
        UnstructuredExcelLoader,
        TextLoader,
        CSVLoader
    )

    loaders = {
        ".pdf":  PyPDFLoader,
        ".md":   UnstructuredMarkdownLoader,
        ".docx": Docx2txtLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".txt":  TextLoader,
        ".csv":  CSVLoader,
    }

    documents = []
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir, exist_ok=True)
        print(f"Created '{docs_dir}' directory. Please add documents and run again.")
        return documents
        
    for filename in os.listdir(docs_dir):
        ext = os.path.splitext(filename)[1].lower()
        path = os.path.join(docs_dir, filename)
        if ext in loaders:
            print(f"Loading: {filename}")
            try:
                loader = loaders[ext](path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Could not load {filename}: {e}")
        else:
            print(f"Skipping unsupported file: {filename}")
    
    print(f"Loaded {len(documents)} document pages total.")
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks

def build_vectorstore(chunks, store_path="vectorstore"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    batch_size = 20
    vectorstore = None
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(chunks) - 1) // batch_size + 1
        
        success = False
        retries = 0
        while not success and retries < 3:
            try:
                print(f"Embedding batch {batch_num} of {total_batches}...")
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
                        print(f"Failed after 3 retries due to rate limit/quota: {e}")
                        raise e
                    print(f"API rate limit hit. Waiting 40s before resuming... ({batch_num}/{total_batches}, Retry {retries}/3)")
                    time.sleep(40)
                else:
                    raise e
                    
    vectorstore.save_local(store_path)
    print("Vector store saved.")
    return vectorstore

def load_vectorstore(store_path="vectorstore"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)

def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant. Answer the question using ONLY
the information in the context below. If the answer is not in the context,
say "I don't know based on the provided documents."

Context:
{context}

Question: {question}
Answer:"""
    )
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )

def main():
    store_path = "vectorstore"
    if not os.path.exists(os.path.join(store_path, "index.faiss")):
        docs = load_documents("docs")
        if not docs:
            print("No documents found in 'docs/' folder.")
            return
        chunks = chunk_documents(docs)
        vectorstore = build_vectorstore(chunks, store_path)
    else:
        print("Loading existing vector store...")
        vectorstore = load_vectorstore(store_path)

    qa_chain = build_qa_chain(vectorstore)
    print("\n=== RAG Assistant Ready ===")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() == "quit":
            break
        result = qa_chain.invoke({"query": query})
        print(f"\nAssistant: {result['result']}")
        sources = {doc.metadata.get("source", "unknown") for doc in result["source_documents"]}
        print(f"Sources: {', '.join(sources)}\n")

if __name__ == "__main__":
    main()
