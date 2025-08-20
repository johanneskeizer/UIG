from transformers import AutoTokenizer, AutoModel
import torch

class PubMedBERTEmbedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        self.model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

    def embed_chunks(self, chunks):
        embeddings = []
        for text in chunks:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                embeddings.append(vector.tolist())
        return embeddings
