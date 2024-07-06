 
from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordBearer
from services.embedding_service import translate_text, translate_to_darija, format_rtl_text
from pydantic import BaseModel

text_ops_router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class TranslateRequest(BaseModel):
    text: str
    target_language: str

@text_ops_router.post("/translate")
def api_translate_text(request: TranslateRequest, token: str = Depends(oauth2_scheme)):
    translated_text = translate_text(request.text, request.target_language)
    return {"translated_text": translated_text}

@text_ops_router.post("/translate_to_darija")
def api_translate_to_darija(text: str, token: str = Depends(oauth2_scheme)):
    translated_text = translate_to_darija(text)
    return {"translated_text": translated_text}

@text_ops_router.post("/format_rtl")
def api_format_rtl(text: str, token: str = Depends(oauth2_scheme)):
    formatted_text = format_rtl_text(text)
    return {"formatted_text": formatted_text}
