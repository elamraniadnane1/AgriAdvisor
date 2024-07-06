from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordBearer
from services.qdrant_service import generate_response
from pydantic import BaseModel

qdrant_router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

class QueryRequest(BaseModel):
    question: str

@qdrant_router.post("/generate_response")
def api_generate_response(request: QueryRequest, token: str = Depends(oauth2_scheme)):
    response_text = generate_response(request.question)
    return {"response": response_text}
