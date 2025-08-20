import openai

# modules/embedder_openai.py
import os
from openai import OpenAI
from typing import List   # <-- ADD THIS

class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-large", dim: int = 3072):
        self.model = model
        self.dim = dim
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        # v1 client
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        out = []
        for text in chunks:
            r = self.client.embeddings.create(model=self.model, input=text)
            out.append(r.data[0].embedding)
        return out
