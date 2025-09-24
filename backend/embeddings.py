# backend/embeddings.py
import os
from typing import List
USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() in ("1", "true", "yes")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if USE_OPENAI:
    import openai
    openai.api_key = OPENAI_API_KEY
else:
    from sentence_transformers import SentenceTransformer
    _st_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(texts: List[str]) -> List[List[float]]:
    """
    Return list of embeddings for a list of texts.
    """
    if USE_OPENAI:
        # OpenAI embeddings (example; choose model per your account)
        resp = openai.Embedding.create(input=texts, model="text-embedding-3-small")
        return [r["embedding"] for r in resp["data"]]
    else:
        return _st_model.encode(texts, show_progress_bar=False).tolist()
