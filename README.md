## Notes and Recent Enhancements
- The script prints progress at every major step.
- If CUDA libraries are missing, the LLM will run on CPU (with a warning).
- All configuration is handled in the `.properties` file for easy portability.
- For best results, use high-quality LLMs and ensure your PDFs are text-based (not just scanned images).
- **Chroma caching:** Chroma vector DB is only rebuilt if PDFs change; otherwise, a cached DB is loaded instantly.
- **FAISS option:** Use `--in_mem_db` flag for in-memory (no-persistence) vector DB.
- **Noisy CUDA errors suppressed:** All native backend CUDA errors are now fully hidden during LLM load; only a clear Python warning is shown if CUDA is missing.
