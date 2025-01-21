from llama_cpp import Llama
from ..core.config import settings

class LlamaModel:
    def __init__(self):
        self.model = Llama(
            model_path=settings.MODEL_PATH,
            n_ctx=2048,
            n_threads=4
        )
    
    def generate(self, prompt: str) -> str:
        output = self.model(
            prompt,
            max_tokens=512,
            temperature=0.7
        )
        return output["choices"][0]["text"]