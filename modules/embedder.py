# modules/embedder.py
from modules.embedder_openai import OpenAIEmbedder
from modules.embedder_pubmedbert import PubMedBERTEmbedder

def get_embedder(model_name):
    if model_name == "openai":
        return OpenAIEmbedder()
    elif model_name == "pubmedbert":
        return PubMedBERTEmbedder()
    else:
        raise ValueError(f"Unknown embedding model: {model_name}")

# OpenAIEmbedder and PubMedBERTEmbedder will each have an .embed_chunks(list_of_strings) method
