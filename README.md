# PDF Chat with LangChain and GPT4All

This project enables you to chat with the contents of your local PDF files using a local LLM (such as GPT4All-compatible models) and semantic search via embeddings.

## How It Works
- **PDF Loading:** The script loads all PDF files from a directory specified in `pdf_chat_cfg.properties`.
- **Text Splitting:** Each PDF is split into manageable text chunks for better retrieval.
- **Embeddings:** Chunks are embedded using a HuggingFace sentence transformer.
- **Vector Store:** Embeddings are stored in a local Chroma vector database for fast similarity search.
- **LLM:** A local LLM (e.g., DeepSeek, GPT4All) is loaded from a path and model name specified in the config file.
- **Conversational Retrieval:** User questions are answered by retrieving relevant PDF chunks and passing them to the LLM.
- **Interactive Chat:** The script provides a terminal-based chat interface. Type questions about your PDFs; type `exit` to quit.

## Configuration
Edit `pdf_chat_cfg.properties` to set:
- `PDF_DIR`: Directory containing your PDF files
- `MODEL_PATH`: Directory where your LLM model is stored
- `MODEL_NAME`: Name of the LLM model file (e.g., `DeepSeek-R1-Distill-Llama-8B-Q4_0.gguf`)

Example:
```
PDF_DIR=/path/to/pdfs
MODEL_PATH=/home/youruser/.local/share/nomic.ai/GPT4All/
MODEL_NAME=DeepSeek-R1-Distill-Llama-8B-Q4_0.gguf
```

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -U langchain-huggingface
   pip install -U langchain-chroma
   ```
2. Edit `pdf_chat_cfg.properties` as above.
3. Run the script:
   ```bash
   python3 pdf_chat.py
   ```
   - Use `--in_mem_db` for in-memory FAISS vector DB (no caching).
4. Ask questions in the terminal about your PDFs!

## Notes
- The script prints progress at every major step.
- If CUDA libraries are missing, the LLM will run on CPU (with a warning).
- All configuration is handled in the `.properties` file for easy portability.
- For best results, use high-quality LLMs and ensure your PDFs are text-based (not just scanned images).

## Recent Enhancements
- **Chroma caching:** Chroma vector DB is only rebuilt if PDFs change; otherwise, a cached DB is loaded instantly.
- **Manifest-based invalidation:** Uses PDF filenames, sizes, and mtimes to detect changes.
- **FAISS option:** Use `--in_mem_db` flag for in-memory (no-persistence) vector DB.
- **Noisy CUDA errors suppressed:** All native backend CUDA errors are now fully hidden during LLM load; only a clear Python warning is shown if CUDA is missing.
- **Chroma import updated:** Now uses `langchain_chroma` to avoid deprecation warnings.
