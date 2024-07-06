 
from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordBearer
from services.speech_service import recognize_speech_from_microphone, text_to_speech
from pydantic import BaseModel

speech_router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class TextToSpeechRequest(BaseModel):
    text: str
    language: str

@speech_router.post("/recognize_speech")
def api_recognize_speech(language: str = "ar", token: str = Depends(oauth2_scheme)):
    speech_text = recognize_speech_from_microphone(language=language)
    return {"speech_text": speech_text}

@speech_router.post("/text_to_speech")
def api_text_to_speech(request: TextToSpeechRequest, token: str = Depends(oauth2_scheme)):
    text_to_speech(request.text, request.language)
    return {"message": "Text to speech conversion completed successfully"}
