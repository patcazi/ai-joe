## Local Vector DB

This project uses ChromaDB to store embeddings for retrieval.  
The `.chroma/` directory is ignored in git — it will be created automatically when you run the ingestion pipeline:

```bash
python ingest.py

If .chroma/ doesn’t exist, regenerate it locally before asking questions in the UI.
