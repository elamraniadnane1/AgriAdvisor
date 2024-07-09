import os
import uuid
import hashlib
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi_login import LoginManager
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
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse


# Initialize FastAPI app
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    
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

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def create_csv(data_list: List[dict], output_csv: str):
    records = []
    for data in data_list:
        record = {"filename": data["filename"], "content": data["content"]}
        records.append(record)
    
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

def process_pdfs(pdf_directory: str):
    data_list_ar = []
    data_list_fr = []

    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            text = extract_text_from_pdf(pdf_path)
            try:
                language = detect(text)
            except:
                language = "unknown"
            
            data = {"filename": filename, "content": text}
            
            if language == "ar":
                data_list_ar.append(data)
            elif language == "fr":
                data_list_fr.append(data)
    
    if data_list_ar:
        create_csv(data_list_ar, output_csv_ar)
    
    if data_list_fr:
        create_csv(data_list_fr, output_csv_fr)

def chunk_text(text: str, max_tokens: int = 800) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word)
        if current_length + word_length > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def get_embedding(content: str) -> List[float]:
    response = openai.Embedding.create(model="text-embedding-ada-002", input=content)
    return response['data'][0]['embedding']

def vectorize_and_store(csv_path: str, collection_name: str):
    if collection_name in [col.name for col in qdrant_client.get_collections().collections]:
        print(f"Collection '{collection_name}' already exists. Skipping vectorization.")
        return
    
    df = pd.read_csv(csv_path)
    
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
    )
    
    points = []
    for index, row in df.iterrows():
        chunks = chunk_text(row['content'])
        for chunk in chunks:
            embedding = get_embedding(chunk)
            point_id = str(uuid.uuid4())  # Generate a unique UUID for each point
            points.append(models.PointStruct(
                id=point_id,
                vector=embedding,
                payload={"filename": row['filename'], "content": chunk, "language": detect(chunk)}
            ))
    
    qdrant_client.upsert(collection_name=collection_name, points=points)

def recognize_speech_from_microphone(language: str = "ar") -> str:
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        return recognizer.recognize_google(audio, language=language)
    except sr.UnknownValueError:
        return "Speech was unintelligible"
    except sr.RequestError:
        return "Could not request results from Google Speech Recognition service"

def text_to_speech(text: str, language: str = "ar"):
    tts = gTTS(text=text, lang=language)
    tts.save("output.mp3")
    os.system("mpg321 output.mp3")

def query_qdrant(question: str, collection_name: str) -> List[str]:
    question_embedding = get_embedding(question)
    
    search_result = qdrant_client.search(collection_name=collection_name, query_vector=question_embedding, limit=3)
    return [hit.payload["content"] for hit in search_result]

def generate_response(question: str, collection_name: str) -> str:
    relevant_chunks = query_qdrant(question, collection_name)
    
    # Integrate the retrieved chunks into the prompt
    prompt = (
        "You are an AI assistant specialized in agricultural advice. Here are some relevant information chunks:\n"
        + "\n".join(f"- {chunk}" for chunk in relevant_chunks)
        + f"\nNow answer the following question: {question}"
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant specialized in agricultural advice."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response['choices'][0]['message']['content']

def translate_text(text: str, target_language: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Translate the following text to {target_language}."},
            {"role": "user", "content": text}
        ]
    )
    
    return response['choices'][0]['message']['content']

def translate_to_darija(text: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Translate the following text to Moroccan Darija using Arabic letters."},
            {"role": "user", "content": text}
        ]
    )
    
    return response['choices'][0]['message']['content']

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

@app.get("/")
async def read_root():
    return {"message": "Welcome to the agricultural advice API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
