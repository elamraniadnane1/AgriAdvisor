import os
import uuid
import hashlib
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi_login import LoginManager
from fastapi_login.exceptions import InvalidCredentialsException
from pydantic import BaseModel
from typing import Optional, List
import fitz  # PyMuPDF
import pandas as pd
from langdetect import detect
import openai
from qdrant_client import QdrantClient
from qdrant_client.http import models
import speech_recognition as sr
from gtts import gTTS
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Directory containing the PDF files
pdf_directory = r"C:\Users\LENOVO\OneDrive\Bureau\Dataset"

# Output CSV files
output_csv_ar = r"C:\Users\LENOVO\OneDrive\Bureau\Dataset\agriculture_data_ar_.csv"
output_csv_fr = r"C:\Users\LENOVO\OneDrive\Bureau\Dataset\agriculture_data_fr_.csv"

# OpenAI API key
openai.api_key = 'sk-proj-IoxzbELHRwIIhrlZVwrtT3BlbkFJvyxGl7jRv3fEzURZJt6g'
# Qdrant client configuration
qdrant_client = QdrantClient("localhost", port=6333)

# Secret key for authentication
SECRET = "supersecretkey"
manager = LoginManager(SECRET, token_url="/auth/login", use_cookie=True)
manager.cookie_name = "auth_token"

# In-memory user storage
users_db = {}

# Pydantic models for request/response data validation
class User(BaseModel):
    username: str
    password: str

class UserInDB(User):
    hashed_password: str

class QueryRequest(BaseModel):
    question: str
    collection: str

class RegisterRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

# Hashing password
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

@manager.user_loader
def load_user(username: str):
    user = users_db.get(username)
    return user

@app.post('/auth/register', status_code=status.HTTP_201_CREATED)
async def register(user: RegisterRequest):
    if user.username in users_db:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")
    
    hashed_password = hash_password(user.password)
    users_db[user.username] = UserInDB(username=user.username, hashed_password=hashed_password)
    return {"message": "User registered successfully"}

@app.post('/auth/login')
async def login(data: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(data.username)
    if not user or not user.hashed_password == hash_password(data.password):
        raise InvalidCredentialsException
    
    access_token = manager.create_access_token(data={"sub": user.username})
    manager.set_cookie(response=None, token=access_token)
    return {"access_token": access_token, "token_type": "bearer"}

@app.post('/auth/logout')
async def logout():
    response = RedirectResponse(url="/")
    manager.set_cookie(response, "")
    return {"message": "Logged out successfully"}

@app.post('/query', dependencies=[Depends(manager)])
async def query(query_request: QueryRequest):
    question = query_request.question
    collection_name = query_request.collection
    response_text = generate_response(question, collection_name)
    return {"response": response_text}

# Remaining functions unchanged

@app.post('/process_pdfs', dependencies=[Depends(manager)])
async def api_process_pdfs():
    process_pdfs(pdf_directory)
    return {"message": "PDF processing and data extraction completed successfully."}

@app.post('/vectorize_and_store', dependencies=[Depends(manager)])
async def api_vectorize_and_store(data: dict):
    collection_name = data['collection_name']
    csv_path = data['csv_path']
    vectorize_and_store(csv_path, collection_name)
    return {"message": f"Vectorization and storage for {collection_name} completed successfully."}

@app.post('/text_to_speech', dependencies=[Depends(manager)])
async def api_text_to_speech(data: dict):
    text = data['text']
    language = data['language']
    text_to_speech(text, language)
    return {"message": "Text to speech conversion completed successfully."}

@app.post('/generate_response', dependencies=[Depends(manager)])
async def api_generate_response(data: dict):
    question = data['question']
    collection_name = data['collection_name']
    response_text = generate_response(question, collection_name)
    return {"response": response_text}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
