import os
import uuid
import hashlib
import re
import time
import json
import tempfile
import wave
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import pyaudio
import pygame
import speech_recognition as sr
from gtts import gTTS
from langdetect import detect
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
import openai
import mlflow
from threading import Thread, Lock
from functools import lru_cache

# Directory and output file paths
PDF_DIRECTORY = r"C:\Users\LENOVO\OneDrive\Bureau\Dataset"
OUTPUT_CSV_AR = r"C:\Users\LENOVO\OneDrive\Bureau\Dataset\agriculture_data_ar_.csv"
OUTPUT_CSV_FR = r"C:\Users\LENOVO\OneDrive\Bureau\Dataset\agriculture_data_fr_.csv"

# OpenAI API key
openai.api_key = 'sk-proj-IoxzbELHRwIIhrlZVwrtT3BlbkFJvyxGl7jRv3fEzURZJt6g'

# Initialize Qdrant client
qdrant_client = QdrantClient("localhost", port=6333)

# FastAPI app setup
app = FastAPI()

# In-memory user store
users = {}
USERS_FILE = 'users.json'

class User(BaseModel):
    id: str
    username: str
    password: str

class UserInDB(User):
    hashed_password: str

class UserCreate(BaseModel):
    username: str
    password: str

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r', encoding='utf-8') as file:
            user_data = json.load(file)
            for user_id, data in user_data.items():
                users[user_id] = UserInDB(id=user_id, username=data['username'], hashed_password=data['password'])
load_users()

def save_users():
    with open(USERS_FILE, 'w', encoding='utf-8') as file:
        json.dump({user_id: {'username': user.username, 'password': user.hashed_password} for user_id, user in users.items()}, file, ensure_ascii=False, indent=4)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username: str, password: str):
    hashed_password = hash_password(password)
    for user in users.values():
        if user.username == username and user.hashed_password == hashed_password:
            return user
    return None

def create_access_token(data: dict, expires_delta: int = 3600):
    return data["sub"]  # Simplified token creation

@app.post("/register", response_model=User)
def register(user: UserCreate):
    user_id = str(uuid.uuid4())
    hashed_password = hash_password(user.password)
    user_in_db = UserInDB(id=user_id, username=user.username, hashed_password=hashed_password)
    users[user_id] = user_in_db
    save_users()
    return user_in_db

@app.post("/login")
def login(form_data: UserCreate):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": user.id})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/logout")
def logout(token: str = Depends(oauth2_scheme)):
    return JSONResponse(content={"message": "Logged out successfully"})

@app.post("/query")
def query(request: Request, token: str = Depends(oauth2_scheme)):
    user = users[token]
    data = request.json()
    question = data['question']
    collection_name = data['collection']
    response_text = generate_response(question, collection_name)
    log_interaction(user.username, question, response_text, collection_name)
    return {"response": response_text}

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def create_csv(data_list, output_csv):
    """Create a CSV file from the extracted data."""
    records = [{"filename": data["filename"], "content": data["content"]} for data in data_list]
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

CACHE_FILE = 'cache.json'

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as file:
            return json.load(file)
    return {}

def truncate_text(text, max_tokens):
    words = text.split()
    truncated_text = ' '.join(words[:max_tokens])
    return truncated_text

def save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as file:
        json.dump(cache, file, ensure_ascii=False, indent=4)

def process_pdfs(pdf_directory):
    """Process all PDFs in the directory and extract data."""
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
        create_csv(data_list_ar, OUTPUT_CSV_AR)
    
    if data_list_fr:
        create_csv(data_list_fr, OUTPUT_CSV_FR)

def chunk_text(text, max_tokens=800):
    """Chunk text into smaller pieces based on a maximum token limit."""
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

def get_embedding(content):
    """Get embedding from OpenAI API."""
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=content
    )
    return response['data'][0]['embedding']

def vectorize_and_store(csv_path, collection_name):
    """Vectorize text and store in Qdrant."""
    if collection_name in [col.name for col in qdrant_client.get_collections().collections]:
        print(f"Collection '{collection_name}' already exists. Skipping vectorization.")
        return
    
    df = pd.read_csv(csv_path)
    
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=1536,  # Dimension of the embeddings
            distance=models.Distance.COSINE
        )
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
                payload={
                    "filename": row['filename'], 
                    "content": chunk,
                    "language": detect(chunk)
                }
            ))
    
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )

recognizer = sr.Recognizer()
mic_lock = Lock()

def recognize_speech_from_microphone(language="ar", callback=None):
    """Recognize speech from the microphone."""
    with mic_lock:
        mic = sr.Microphone()
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = recognizer.listen(source)
        try:
            transcription = recognizer.recognize_google(audio, language=language)
            if callback:
                callback(transcription)
            return transcription
        except sr.UnknownValueError:
            return "Speech was unintelligible"
        except sr.RequestError:
            return "Could not request results from Google Speech Recognition service"

def clean_text_for_speech(text):
    """Clean text for speech by removing unnecessary characters."""
    text = re.sub(r'\*\*\*|\.{2,}', '', text)
    return text.strip()

def text_to_speech(text, language="ar"):
    """Convert text to speech."""
    # Handle unsupported language "dar" by mapping it to "ar"
    if language == "dar":
        language = "ar"
    
    clean_text = clean_text_for_speech(text)
    tts = gTTS(text=clean_text, lang=language)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        temp_filename = fp.name
        tts.save(temp_filename)
    pygame.mixer.init()
    pygame.mixer.music.load(temp_filename)
    pygame.mixer.music.play()

def play_audio():
    """Play the audio."""
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    pygame.mixer.music.unpause()

def pause_audio():
    """Pause the audio."""
    pygame.mixer.music.pause()

def replay_audio():
    """Replay the audio."""
    pygame.mixer.music.stop()
    pygame.mixer.music.play()

@lru_cache(maxsize=100)
def cached_query_qdrant(question, collection_name):
    """Cached version of querying Qdrant to find the closest matching chunks for the given question."""
    question_embedding = get_embedding(question)
    
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=question_embedding,
        limit=3
    )
    
    return [hit.payload["content"] for hit in search_result]

def generate_response(question, collection_name, quality_mode="good", input_token_limit=800, output_token_limit=500):
    # Define model and max tokens based on quality mode
    model = "gpt-3.5-turbo"
    max_tokens = 500
    
    if quality_mode == "premium":
        model = "gpt-4o"
        max_tokens = 2000
    elif quality_mode == "economy":
        model = "gpt-4"
        max_tokens = 100

    # Truncate input question based on token limit
    truncated_question = truncate_text(question, input_token_limit)
    
    relevant_chunks = cached_query_qdrant(truncated_question, collection_name)
    
    prompt = (
        "You are an AI assistant specialized in agricultural advice. Here are some relevant information chunks:\n"
        + "\n".join(f"- {chunk}" for chunk in relevant_chunks)
        + f"\nNow answer the following question: {truncated_question}"
    )
    
    # Ensure the prompt is within the input token limit
    prompt = truncate_text(prompt, input_token_limit)
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an AI assistant specialized in agricultural advice."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=output_token_limit
    )
    
    return response['choices'][0]['message']['content']

def translate_text(text, target_language):
    """Translate text to the target language using GPT-4."""
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Translate the following text to {target_language}."},
            {"role": "user", "content": text}
        ]
    )
    
    return response['choices'][0]['message']['content']

def translate_to_darija(text):
    """Translate text to Moroccan Darija using GPT-4."""
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Translate the following text to Moroccan Darija using Arabic letters."},
            {"role": "user", "content": text}
        ]
    )
    
    return response['choices'][0]['message']['content']

def user_input_choice(input_lang, output_lang, user_input, input_type="text", quality_mode="good", input_token_limit=800, output_token_limit=500):
    """Process user input and generate a response."""
    if input_type == "voice":
        print("Please speak into the microphone...")
        speech_text = recognize_speech_from_microphone(language=input_lang)
        print(f"Recognized Speech: {speech_text}")
        response_text = generate_response(speech_text, "agriculture_ar" if input_lang == "ar" else "agriculture_fr", quality_mode, input_token_limit, output_token_limit)
        if output_lang == "dar":
            response_text = translate_to_darija(response_text)
        elif input_lang != output_lang:
            response_text = translate_text(response_text, output_lang)
        print(f"Response: {response_text}")
        return response_text
    elif input_type == "text":
        print(f"Received Text: {user_input}")
        response_text = generate_response(user_input, "agriculture_ar" if input_lang == "ar" else "agriculture_fr", quality_mode, input_token_limit, output_token_limit)
        if output_lang == "dar":
            response_text = translate_to_darija(response_text)
        elif input_lang != output_lang:
            response_text = translate_text(response_text, output_lang)
        print(f"Response: {response_text}")
        return response_text
    else:
        print("Invalid choice. Please enter 'voice' or 'text'.")
        return "Invalid choice."

def format_rtl_text(text):
    """Format text for right-to-left languages."""
    return f"\u202B{text}\u202C"

def log_interaction(user, question, response, collection_name):
    """Log user interaction for LLMOps."""
    log_data = {
        "timestamp": time.time(),
        "user": user,
        "question": question,
        "response": response,
        "collection_name": collection_name
    }
    with open("interaction_logs.json", "a") as log_file:
        log_file.write(json.dumps(log_data) + "\n")
    
    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_param("user", user)
        mlflow.log_param("question", question)
        mlflow.log_param("response", response)
        mlflow.log_param("collection_name", collection_name)
        mlflow.log_metric("timestamp", log_data["timestamp"])

def generate_report():
    """Generate a report of user interactions."""
    interactions = []
    try:
        with open("interaction_logs.json", "r") as log_file:
            for line in log_file:
                interactions.append(json.loads(line))
    except FileNotFoundError:
        pass

    return pd.DataFrame(interactions)

if __name__ == "__main__":
    def run_fastapi_app():
        process_pdfs(PDF_DIRECTORY)
        vectorize_and_store(OUTPUT_CSV_AR, "agriculture_ar")
        vectorize_and_store(OUTPUT_CSV_FR, "agriculture_fr")
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8000)

    Thread(target=run_fastapi_app).start()
