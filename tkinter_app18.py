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
from flask import Flask, request, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from qdrant_client import QdrantClient
from qdrant_client.http import models
import openai
import mlflow
import mlflow.pyfunc
import customtkinter as ctk
from tkinter import scrolledtext, messagebox
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

# Flask app setup
flask_app = Flask(__name__)
flask_app.secret_key = 'supersecretkey'
login_manager = LoginManager()
login_manager.init_app(flask_app)

# In-memory user store
users = {}
USERS_FILE = 'users.json'

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r', encoding='utf-8') as file:
            user_data = json.load(file)
            for user_id, data in user_data.items():
                users[user_id] = User(user_id, data['username'], data['password'])
load_users()

def save_users():
    with open(USERS_FILE, 'w', encoding='utf-8') as file:
        json.dump({user_id: {'username': user.username, 'password': user.password} for user_id, user in users.items()}, file, ensure_ascii=False, indent=4)

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@flask_app.route('/register', methods=['POST'])
def register():
    username = request.json['username']
    password = hash_password(request.json['password'])
    user_id = str(uuid.uuid4())
    users[user_id] = User(user_id, username, password)
    save_users()
    return jsonify({'message': 'User registered successfully'}), 201

@flask_app.route('/login', methods=['POST'])
def login():
    username = request.json['username']
    password = hash_password(request.json['password'])
    for user in users.values():
        if user.username == username and user.password == password:
            login_user(user)
            return jsonify({'message': 'Login successful'}), 200
    return jsonify({'message': 'Invalid credentials'}), 401

@flask_app.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logged out successfully'}), 200

@flask_app.route('/query', methods=['POST'])
@login_required
def query():
    question = request.json['question']
    collection_name = request.json['collection']
    response_text = generate_response(question, collection_name)
    log_interaction(current_user.username, question, response_text, collection_name)
    return jsonify({'response': response_text}), 200

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

class Application(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI AgriAdvisor")
        self.geometry("1200x800")
        self.iconbitmap("icon.ico")

        self.username = None
        self.recording = False
        self.frames = []

        self.show_login_window()

        # Cache dictionary for storing prompts and their responses
        self.cache = {}

    def show_login_window(self):
        """Display the login window."""
        self.login_window = ctk.CTkToplevel(self)
        self.login_window.title("Login")
        self.login_window.geometry("400x300")
        self.login_window.iconbitmap("icon.ico")

        ctk.CTkLabel(self.login_window, text="Username", font=("Times New Roman", 12)).grid(row=0, column=0, padx=10, pady=10)
        ctk.CTkLabel(self.login_window, text="Password", font=("Times New Roman", 12)).grid(row=1, column=0, padx=10, pady=10)

        self.username_entry = ctk.CTkEntry(self.login_window, font=("Times New Roman", 12))
        self.password_entry = ctk.CTkEntry(self.login_window, show='*', font=("Times New Roman", 12))

        self.username_entry.grid(row=0, column=1, padx=10, pady=10)
        self.password_entry.grid(row=1, column=1, padx=10, pady=10)

        ctk.CTkLabel(self.login_window, text="Language", font=("Times New Roman", 12)).grid(row=2, column=0, padx=10, pady=10)
        self.language_var = ctk.StringVar()
        self.language_combobox = ctk.CTkComboBox(self.login_window, variable=self.language_var, values=["AR", "FR", "DAR"], font=("Times New Roman", 12))
        self.language_combobox.grid(row=2, column=1, padx=10, pady=10)
        self.language_combobox.set("AR")

        ctk.CTkButton(self.login_window, text="Login", command=self.login, font=("Times New Roman", 12)).grid(row=3, column=0, columnspan=2, pady=10)
        ctk.CTkButton(self.login_window, text="Register", command=self.register, font=("Times New Roman", 12)).grid(row=4, column=0, columnspan=2, pady=10)

    def login(self):
        """Handle user login."""
        username = self.username_entry.get()
        password = self.password_entry.get()
        language = self.language_var.get()

        if self.authenticate(username, password):
            self.username = username
            self.language = language
            self.login_window.destroy()
            self.show_main_window()
            self.translate_app()
        else:
            messagebox.showerror("Error", "Invalid credentials")

    def authenticate(self, username, password):
        """Authenticate the user."""
        hashed_password = hash_password(password)
        for user in users.values():
            if user.username == username and user.password == hashed_password:
                return True
        return False

    def register(self):
        """Handle user registration."""
        username = self.username_entry.get()
        password = self.password_entry.get()

        if username in [user.username for user in users.values()]:
            messagebox.showerror("Error", "Username already exists")
        else:
            hashed_password = hash_password(password)
            user_id = str(uuid.uuid4())
            users[user_id] = User(user_id, username, hashed_password)
            save_users()
            messagebox.showinfo("Success", "User registered successfully")

    def translate_app(self):
        """Translate the application based on the selected language."""
        if self.language == "AR":
            self.translate_to_arabic()
        elif self.language == "FR":
            self.translate_to_french()
        elif self.language == "DAR":
            self.translate_to_darija()

    def translate_to_arabic(self):
        self.input_label.configure(text="نص المدخلات:")
        self.lang_label.configure(text="اختر لغة الإدخال:")
        self.output_lang_label.configure(text="اختر لغة الإخراج:")
        self.quality_label.configure(text="اختر وضع الجودة:")
        self.input_token_label.configure(text="حد المدخلات:")
        self.output_token_label.configure(text="حد المخرجات:")
        self.output_label.configure(text="نص الإخراج:")
        self.submit_button.configure(text="إرسال")
        self.submit_text_button.configure(text="إرسال")
        self.record_button.configure(text="تسجيل")
        self.stop_button.configure(text="إيقاف التسجيل")
        self.speak_button.configure(text="قراءة الإخراج بصوت عال")
        self.play_button.configure(text="▶ تشغيل")
        self.pause_button.configure(text="⏸ إيقاف مؤقت")
        self.replay_button.configure(text="⏪ إعادة التشغيل")
        self.report_button.configure(text="توليد التقرير")
        self.user_info_label.configure(text=f"مسجل الدخول: {self.username}")
        self.logout_button.configure(text="تسجيل الخروج")
        self.recording_label.configure(text="تسجيل... يرجى التحدث في الميكروفون")
        self.feedback_label.configure(text="ملاحظات (اختياري):")
        self.submit_feedback_button.configure(text="إرسال الملاحظات")

    def translate_to_french(self):
        self.input_label.configure(text="Texte d'entrée :")
        self.lang_label.configure(text="Sélectionnez la langue d'entrée :")
        self.output_lang_label.configure(text="Sélectionnez la langue de sortie :")
        self.quality_label.configure(text="Sélectionnez le mode de qualité :")
        self.input_token_label.configure(text="Limite de tokens d'entrée :")
        self.output_token_label.configure(text="Limite de tokens de sortie :")
        self.output_label.configure(text="Texte de sortie :")
        self.submit_button.configure(text="Soumettre")
        self.submit_text_button.configure(text="Soumettre")
        self.record_button.configure(text="Enregistrer")
        self.stop_button.configure(text="Arrêter l'enregistrement")
        self.speak_button.configure(text="Lire le texte de sortie à haute voix")
        self.play_button.configure(text="▶ Jouer")
        self.pause_button.configure(text="⏸ Pause")
        self.replay_button.configure(text="⏪ Rejouer")
        self.report_button.configure(text="Générer le rapport")
        self.user_info_label.configure(text=f"Connecté en tant que : {self.username}")
        self.logout_button.configure(text="Se déconnecter")
        self.recording_label.configure(text="Enregistrement... Veuillez parler dans le microphone")
        self.feedback_label.configure(text="Commentaires (Optionnel) :")
        self.submit_feedback_button.configure(text="Soumettre les commentaires")

    def translate_to_darija(self):
        self.input_label.configure(text="النص المدخل:")
        self.lang_label.configure(text="اختار لغة الإدخال:")
        self.output_lang_label.configure(text="اختار لغة الإخراج:")
        self.quality_label.configure(text="اختار وضع الجودة:")
        self.input_token_label.configure(text="حد المدخلات:")
        self.output_token_label.configure(text="حد المخرجات:")
        self.output_label.configure(text="النص الإخراج:")
        self.submit_button.configure(text="إرسال")
        self.submit_text_button.configure(text="إرسال")
        self.record_button.configure(text="تسجيل")
        self.stop_button.configure(text="إيقاف التسجيل")
        self.speak_button.configure(text="قراءة الإخراج بصوت عالي")
        self.play_button.configure(text="▶ تشغيل")
        self.pause_button.configure(text="⏸ إيقاف مؤقت")
        self.replay_button.configure(text="⏪ إعادة التشغيل")
        self.report_button.configure(text="توليد التقرير")
        self.user_info_label.configure(text=f"مسجل الدخول: {self.username}")
        self.logout_button.configure(text="تسجيل الخروج")
        self.recording_label.configure(text="تسجيل... المرجو التحدث في الميكروفون")
        self.feedback_label.configure(text="ملاحظات (اختياري):")
        self.submit_feedback_button.configure(text="إرسال الملاحظات")

    def show_main_window(self):
        """Display the main application window."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)

        self.input_label = ctk.CTkLabel(self, text="Input Text:", font=("Times New Roman", 12))
        self.input_label.grid(row=0, column=0, pady=5, sticky='nsew')
        self.input_text = ctk.CTkTextbox(self, wrap='word', width=600, height=200, font=("Times New Roman", 12))
        self.input_text.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')

        self.lang_label = ctk.CTkLabel(self, text="Select Input Language:", font=("Times New Roman", 12))
        self.lang_label.grid(row=2, column=0, pady=5, sticky='nsew')
        self.input_lang = ctk.CTkComboBox(self, values=["ar", "fr", "dar"], font=("Times New Roman", 12))
        self.input_lang.grid(row=3, column=0, pady=5, sticky='nsew')
        
        self.output_lang_label = ctk.CTkLabel(self, text="Select Output Language:", font=("Times New Roman", 12))
        self.output_lang_label.grid(row=4, column=0, pady=5, sticky='nsew')
        self.output_lang = ctk.CTkComboBox(self, values=["ar", "fr", "dar"], font=("Times New Roman", 12))
        self.output_lang.grid(row=5, column=0, pady=5, sticky='nsew')
        
        self.quality_label = ctk.CTkLabel(self, text="Select Quality Mode:", font=("Times New Roman", 12))
        self.quality_label.grid(row=6, column=0, pady=5, sticky='nsew')
        self.quality_mode = ctk.CTkComboBox(self, values=["economy", "good", "premium"], font=("Times New Roman", 12))
        self.quality_mode.grid(row=7, column=0, pady=5, sticky='nsew')

        self.input_token_label = ctk.CTkLabel(self, text="Input Token Limit:", font=("Times New Roman", 12))
        self.input_token_label.grid(row=8, column=0, pady=5, sticky='nsew')
        self.input_token_limit = ctk.CTkEntry(self, font=("Times New Roman", 12))
        self.input_token_limit.grid(row=9, column=0, pady=5, sticky='nsew')
        self.input_token_limit.insert(0, "800")

        self.output_token_label = ctk.CTkLabel(self, text="Output Token Limit:", font=("Times New Roman", 12))
        self.output_token_label.grid(row=10, column=0, pady=5, sticky='nsew')
        self.output_token_limit = ctk.CTkEntry(self, font=("Times New Roman", 12))
        self.output_token_limit.grid(row=11, column=0, pady=5, sticky='nsew')
        self.output_token_limit.insert(0, "500")

        self.submit_button = ctk.CTkButton(self, text="Submit", command=self.process_input, font=("Times New Roman", 12))
        self.submit_button.grid(row=12, column=0, pady=5)

        self.output_label = ctk.CTkLabel(self, text="Output Text:", font=("Times New Roman", 12))
        self.output_label.grid(row=0, column=2, pady=5, sticky='nsew')
        self.output_text = ctk.CTkTextbox(self, wrap='word', width=800, height=300, font=("Times New Roman", 12))
        self.output_text.grid(row=1, column=2, padx=5, pady=5, sticky='nsew')

        self.record_button = ctk.CTkButton(self, text="Record", command=self.start_recording, font=("Times New Roman", 12))
        self.record_button.grid(row=2, column=1, pady=5)

        self.stop_button = ctk.CTkButton(self, text="Stop Recording", command=self.stop_recording, font=("Times New Roman", 12))
        self.stop_button.grid(row=3, column=1, pady=5)
        self.stop_button.configure(state="disabled")

        self.speak_button = ctk.CTkButton(self, text="Read Aloud Output", command=self.read_aloud_output, font=("Times New Roman", 12))
        self.speak_button.grid(row=4, column=1, pady=5)

        self.play_button = ctk.CTkButton(self, text="▶ Play", command=play_audio, font=("Times New Roman", 12))
        self.play_button.grid(row=5, column=1, pady=5)

        self.pause_button = ctk.CTkButton(self, text="⏸ Pause", command=pause_audio, font=("Times New Roman", 12))
        self.pause_button.grid(row=6, column=1, pady=5)

        self.replay_button = ctk.CTkButton(self, text="⏪ Replay", command=replay_audio, font=("Times New Roman", 12))
        self.replay_button.grid(row=7, column=1, pady=5)

        self.report_button = ctk.CTkButton(self, text="Generate Report", command=self.display_report, font=("Times New Roman", 12))
        self.report_button.grid(row=8, column=1, pady=5)

        self.user_info_label = ctk.CTkLabel(self, text=f"Logged in as: {self.username}", font=("Times New Roman", 12))
        self.user_info_label.grid(row=9, column=1, pady=5, sticky='w')

        self.logout_button = ctk.CTkButton(self, text="Logout", command=self.logout, font=("Times New Roman", 12))
        self.logout_button.grid(row=9, column=2, pady=5, sticky='e')

        self.recording_label = ctk.CTkLabel(self, text="Recording... Please speak into the microphone", text_color="red", font=("Times New Roman", 12))
        self.recording_label.grid(row=10, column=0, columnspan=3, pady=5)
        self.recording_label.grid_remove()

        self.transcription_label = ctk.CTkLabel(self, text="", text_color="blue", font=("Times New Roman", 12))
        self.transcription_label.grid(row=11, column=0, columnspan=3, pady=5)
        self.transcription_label.grid_remove()

        self.feedback_label = ctk.CTkLabel(self, text="Feedback (Optional):", font=("Times New Roman", 12))
        self.feedback_label.grid(row=12, column=0, pady=5, sticky='nsew')
        self.feedback_text = ctk.CTkTextbox(self, wrap='word', width=40, height=10, font=("Times New Roman", 12))
        self.feedback_text.grid(row=13, column=0, padx=5, pady=5, sticky='nsew')

        self.submit_feedback_button = ctk.CTkButton(self, text="Submit Feedback", command=self.submit_feedback, font=("Times New Roman", 12))
        self.submit_feedback_button.grid(row=14, column=0, pady=5)

        self.submit_text_button = ctk.CTkButton(self, text="Submit Text", command=self.process_input, font=("Times New Roman", 12))
        self.submit_text_button.grid(row=12, column=1, pady=5)

    def logout(self):
        """Handle user logout."""
        self.username = None
        for widget in self.winfo_children():
            widget.destroy()
        self.show_login_window()

    def process_input(self):
        """Process the text input."""
        user_input = self.input_text.get("1.0", 'end').strip()
        input_lang = self.input_lang.get()
        output_lang = self.output_lang.get()
        quality_mode = self.quality_mode.get()
        input_token_limit = self.validate_token_limit(self.input_token_limit.get())
        output_token_limit = self.validate_token_limit(self.output_token_limit.get())

        if not user_input or not input_lang or not output_lang or not quality_mode:
            messagebox.showerror("Error", "All fields must be filled")
            return
        
        if input_token_limit is None or output_token_limit is None:
            messagebox.showerror("Error", "Token limits must be valid integers")
            return

        # Check if the prompt is already in the cache
        cache_key = f"{input_lang}:{output_lang}:{user_input}"
        if cache_key in self.cache:
            response_text = self.cache[cache_key]
            self.update_output_text(response_text)
        else:
            Thread(target=self.run_user_input_choice, args=(input_lang, output_lang, user_input, "text", cache_key, quality_mode, input_token_limit, output_token_limit)).start()

    def validate_token_limit(self, token_limit):
        try:
            return int(token_limit)
        except ValueError:
            return None

    def start_recording(self):
        """Start recording audio."""
        self.recording = True
        self.record_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.recording_label.grid()

        Thread(target=self.record_audio).start()

    def stop_recording(self):
        """Stop recording audio."""
        self.recording = False
        self.record_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.recording_label.grid_remove()

    def record_audio(self):
        """Record audio from the microphone."""
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        self.frames = []

        while self.recording:
            data = stream.read(1024)
            self.frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        self.save_audio()

    def save_audio(self):
        """Save recorded audio to a file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            self.audio_filename = fp.name

        wf = wave.open(self.audio_filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        self.transcribe_audio()

    def transcribe_audio(self):
        """Transcribe audio to text."""
        recognizer = sr.Recognizer()
        with sr.AudioFile(self.audio_filename) as source:
            audio = recognizer.record(source)
            try:
                transcription = recognizer.recognize_google(audio)
                self.update_transcription_label(transcription)
                cache_key = f"{self.input_lang.get()}:{self.output_lang.get()}:{transcription}"
                self.run_user_input_choice(self.input_lang.get(), self.output_lang.get(), transcription, "voice", cache_key, self.quality_mode.get(), int(self.input_token_limit.get()), int(self.output_token_limit.get()))
            except sr.UnknownValueError:
                self.update_transcription_label("Speech was unintelligible")
            except sr.RequestError:
                self.update_transcription_label("Could not request results from Google Speech Recognition service")

    def run_user_input_choice(self, input_lang, output_lang, user_input, input_type, cache_key, quality_mode, input_token_limit, output_token_limit):
        """Run the user input choice function in a separate thread."""
        if input_type == "voice":
            response_text = generate_response(user_input, "agriculture_ar" if input_lang == "ar" else "agriculture_fr", quality_mode, input_token_limit, output_token_limit)
            if output_lang == "dar":
                response_text = translate_to_darija(response_text)
            elif input_lang != output_lang:
                response_text = translate_text(response_text, output_lang)
        else:
            response_text = user_input_choice(input_lang, output_lang, user_input, input_type, quality_mode, input_token_limit, output_token_limit)
        
        if output_lang == "ar":
            response_text = format_rtl_text(response_text)
        
        self.cache[cache_key] = response_text  # Cache the response
        save_cache(self.cache)  # Save the updated cache
        self.after(0, self.update_output_text, response_text)
    
    def update_output_text(self, response_text):
        """Update the output text widget."""
        self.output_text.delete("1.0", 'end')
        self.output_text.insert('end', response_text)

    def update_transcription_label(self, transcription):
        """Update the transcription label."""
        self.transcription_label.configure(text=f"Transcription: {transcription}")

    def read_aloud_output(self):
        """Read aloud the output text."""
        output_lang = self.output_lang.get()
        response_text = self.output_text.get("1.0", 'end').strip()
        if response_text:
            Thread(target=text_to_speech, args=(response_text, output_lang)).start()

    def display_report(self):
        """Display the interaction report."""
        report = generate_report()
        report_window = ctk.CTkToplevel(self)
        report_window.title("Interaction Report")
        report_window.iconbitmap("icon.ico")
        report_text = ctk.CTkTextbox(report_window, wrap='word', width=100, height=20, font=("Times New Roman", 12))
        report_text.pack(padx=10, pady=10)
        report_text.insert('end', report.to_string())

    def submit_feedback(self):
        """Submit feedback for the generated response."""
        feedback = self.feedback_text.get("1.0", 'end').strip()
        if not feedback:
            messagebox.showerror("Error", "Feedback cannot be empty")
            return

        log_data = {
            "timestamp": time.time(),
            "user": self.username,
            "feedback": feedback
        }
        with open("feedback_logs.json", "a") as log_file:
            log_file.write(json.dumps(log_data) + "\n")
        
        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_param("user", self.username)
            mlflow.log_param("feedback", feedback)
            mlflow.log_metric("timestamp", log_data["timestamp"])

        messagebox.showinfo("Success", "Feedback submitted successfully")
        self.feedback_text.delete("1.0", 'end')

if __name__ == "__main__":
    def run_flask_app():
        process_pdfs(PDF_DIRECTORY)
        vectorize_and_store(OUTPUT_CSV_AR, "agriculture_ar")
        vectorize_and_store(OUTPUT_CSV_FR, "agriculture_fr")
        flask_app.run(debug=False)

    Thread(target=run_flask_app).start()

    app = Application()
    app.mainloop()
