import os
import sys
import warnings
import json
import contextlib
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.llms import GPT4All
from langchain.chains import ConversationalRetrievalChain

# --- Read config from properties file ---
def read_properties(filepath):
    props = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                props[key.strip()] = value.strip()
    return props

CFG_PATH = os.path.join(os.path.dirname(__file__), 'pdf_chat_cfg.properties')
cfg = read_properties(CFG_PATH)
PDF_DIR = cfg.get('PDF_DIR')
MODEL_DIR = cfg.get('MODEL_PATH')
MODEL_NAME = cfg.get('MODEL_NAME')
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Check for --in_mem_db flag
USE_INMEMORY = '--in_mem_db' in sys.argv

# Helper: get a manifest of all PDFs (filenames, sizes, mtimes)
def get_pdf_manifest(pdf_dir):
    manifest = {}
    for fname in sorted(os.listdir(pdf_dir)):
        if fname.lower().endswith('.pdf'):
            path = os.path.join(pdf_dir, fname)
            stat = os.stat(path)
            manifest[fname] = {'size': stat.st_size, 'mtime': stat.st_mtime}
    return manifest

# Helper: check if manifest has changed
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), 'chroma_db', 'pdf_manifest.json')
def manifest_changed():
    if not os.path.exists(MANIFEST_PATH):
        return True
    try:
        with open(MANIFEST_PATH, 'r') as f:
            old = json.load(f)
        new = get_pdf_manifest(PDF_DIR)
        return old != new
    except Exception:
        return True

def save_manifest():
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(get_pdf_manifest(PDF_DIR), f)

# --- OS-level stderr suppression context manager ---
@contextlib.contextmanager
def suppress_stderr_fully():
    """Suppresses all output to stderr, including from C/C++ extensions."""
    import sys, os
    stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)

# --- MAIN LOGIC ---
if USE_INMEMORY:
    # Always process PDFs and create FAISS in-memory DB
    print("[INFO] Scanning PDF directory:", PDF_DIR)
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    print(f"[INFO] Found {len(pdf_files)} PDF files.")
    all_docs = []
    for i, fname in enumerate(pdf_files):
        print(f"[INFO] Loading PDF {i+1}/{len(pdf_files)}: {fname}")
        loader = PyPDFLoader(os.path.join(PDF_DIR, fname))
        loaded = loader.load()
        print(f"[INFO] Loaded {len(loaded)} documents from {fname}")
        all_docs.extend(loaded)
    print(f"[INFO] Total loaded documents: {len(all_docs)}")
    print("[INFO] Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(all_docs)
    print(f"[INFO] Total chunks after splitting: {len(docs)}")
    print("[INFO] Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("[INFO] Embeddings created.")
    print("[INFO] Creating vector database (FAISS)...")
    vectordb = FAISS.from_documents(docs, embeddings)
    print("[INFO] Vector database (FAISS) ready.")
else:
    chroma_db_exists = os.path.exists(os.path.join(os.path.dirname(__file__), 'chroma_db'))
    if chroma_db_exists and not manifest_changed():
        print("[INFO] Loading cached Chroma vector database...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        print("[INFO] Cached vector database loaded.")
    else:
        print("[INFO] Scanning PDF directory:", PDF_DIR)
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
        print(f"[INFO] Found {len(pdf_files)} PDF files.")
        all_docs = []
        for i, fname in enumerate(pdf_files):
            print(f"[INFO] Loading PDF {i+1}/{len(pdf_files)}: {fname}")
            loader = PyPDFLoader(os.path.join(PDF_DIR, fname))
            loaded = loader.load()
            print(f"[INFO] Loaded {len(loaded)} documents from {fname}")
            all_docs.extend(loaded)
        print(f"[INFO] Total loaded documents: {len(all_docs)}")
        print("[INFO] Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(all_docs)
        print(f"[INFO] Total chunks after splitting: {len(docs)}")
        print("[INFO] Creating embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("[INFO] Embeddings created.")
        print("[INFO] Creating vector database (Chroma)...")
        vectordb = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
        save_manifest()
        print("[INFO] Vector database (Chroma) created and cached.")

# Suppress CUDA-related GPT4All errors at OS-level (native and Python)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Failed to load libllamamodel-mainline-cuda.*")
    warnings.filterwarnings("ignore", message="Failed to load libllamamodel-mainline-cuda-avxonly.*")
    print("[INFO] Loading GPT4All model...")
    with suppress_stderr_fully():
        llm = GPT4All(model=MODEL_PATH, verbose=True)
print("[INFO] LLM loaded.")

print("[INFO] Setting up ConversationalRetrievalChain...")
qa_chain = ConversationalRetrievalChain.from_llm(llm, vectordb.as_retriever())
print("[INFO] Ready for chat!")

# Warn about missing CUDA libraries if present in logs
try:
    import ctypes
    try:
        ctypes.CDLL("libcudart.so.11.0")
    except OSError:
        print("[WARNING] CUDA libraries not found. Model will run on CPU. If you have a GPU and CUDA installed, ensure your library paths are set correctly.")
except Exception as e:
    print(f"[WARNING] CUDA check failed: {e}")

# Simple chat loop
chat_history = []
print("Ask questions about your PDFs! Type 'exit' to quit.")
while True:
    query = input("You: ")
    if query.strip().lower() == "exit":
        break
    # Use invoke() instead of __call__ (deprecation fix)
    result = qa_chain.invoke({"question": query, "chat_history": chat_history})
    print("Bot:", result["answer"])
    chat_history.append((query, result["answer"]))
