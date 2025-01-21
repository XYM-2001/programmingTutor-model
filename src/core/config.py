from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Programming Tutor"
    MODEL_PATH: str = "models/llama-2-7b-chat.gguf"
    EMBEDDINGS_MODEL: str = "all-MiniLM-L6-v2"
    VECTOR_STORE_PATH: str = "vectorstore"
    
    class Config:
        env_file = ".env"

settings = Settings()