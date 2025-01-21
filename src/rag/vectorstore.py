from typing import List
import faiss
import numpy as np
from .embeddings import get_embeddings

class VectorStore:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.index = None
        self.texts = []
    
    def add_texts(self, texts: List[str]):
        self.texts.extend(texts)
        embeddings = self.embeddings.encode(texts)
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))
    
    def similarity_search(self, query: str, k: int = 3):
        query_embedding = self.embeddings.encode([query])
        D, I = self.index.search(query_embedding, k)
        return [self.texts[i] for i in I[0]]