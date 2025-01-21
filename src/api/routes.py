from fastapi import APIRouter, HTTPException
from .schemas import ChatRequest, ChatResponse
from ..llm.model import LlamaModel
from ..rag.vectorstore import VectorStore

router = APIRouter()
llm = LlamaModel()
vectorstore = VectorStore()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        context = vectorstore.similarity_search(request.query)
        prompt = f"Context: {context}\n\nQuestion: {request.query}\nAnswer:"
        response = llm.generate(prompt)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))