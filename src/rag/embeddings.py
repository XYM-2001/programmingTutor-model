from sentence_transformers import SentenceTransformer
from ..core.config import settings

def get_embeddings():
    return SentenceTransformer(settings.EMBEDDINGS_MODEL)