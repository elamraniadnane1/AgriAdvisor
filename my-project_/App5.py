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
import customtkinter as ctk
from tkinter import Scrollbar, messagebox, Canvas, Frame
from tkinter import simpledialog  # Added this import
import requests  # Added this import
from threading import Thread, Lock
from functools import lru_cache
from PIL import Image, ImageTk
from sklearn.metrics import f1_score
from rouge_score import rouge_scorer
import sacrebleu
from prometheus_flask_exporter import PrometheusMetrics
import logging
from config import Config

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = Config()

# Initialize OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY', config.openai_api_key)

# Initialize Qdrant client
qdrant_client = QdrantClient(config.qdrant_host, port=config.qdrant_port)

# Flask app setup
flask_app = Flask(__name__)
flask_app.secret_key = config.flask_secret_key
login_manager = LoginManager()
login_manager.init_app(flask_app)
login_manager.login_view = 'login'
metrics = PrometheusMetrics(flask_app)

# In-memory user store
users = {}
USERS_FILE = config.users_file

def compute_f1_score(true_text, predicted_text):
    true_tokens = true_text.split()
    pred_tokens = predicted_text.split()
    max_len = max(len(true_tokens), len(pred_tokens))
    true_labels = [1] * len(true_tokens) + [0] * (max_len - len(true_tokens))
    pred_labels = [1 if token in true_tokens else 0 for token in pred_tokens] + [0] * (max_len - len(pred_tokens))
    return f1_score(true_labels, pred_labels, average='micro')

def compute_rouge_l(true_text, predicted_text):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(true_text, predicted_text)
    return scores['rougeL'].fmeasure

def compute_sacrebleu(true_text, predicted_text):
    bleu = sacrebleu.corpus_bleu([predicted_text], [[true_text]])
    return bleu

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

from flask import session
@flask_app.route('/user', methods=['GET'])
@login_required
def get_user_info():
    if 'user_id' not in session:
        return jsonify({'error': 'User not authenticated'}), 401

    user_info = {
        'username': current_user.username,
        'id': current_user.id
    }
    return jsonify(user_info), 200


true_text = ""

@flask_app.route('/update_true_text', methods=['POST'])
@login_required
def update_true_text():
    global true_text
    new_true_text = request.json.get('true_text')
    if not new_true_text:
        return jsonify({'error': 'True text is required'}), 400

    true_text = new_true_text
    return jsonify({'message': 'True text updated successfully'}), 200


@flask_app.route('/user', methods=['PUT'])
@login_required
def update_user_details():
    new_username = request.json.get('username')
    new_password = request.json.get('password')
    if new_username:
        current_user.username = new_username
    if new_password:
        current_user.password = hash_password(new_password)
    save_users()
    return jsonify({'message': 'User details updated successfully'}), 200

@flask_app.route('/user', methods=['DELETE'])
@login_required
def delete_user():
    user_id = current_user.id
    del users[user_id]
    save_users()
    logout_user()
    return jsonify({'message': 'User deleted successfully'}), 200

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
@metrics.counter('requests_by_user', 'Number of requests by user', labels={'username': lambda: current_user.username})
def query():
    question = request.json['question']
    collection_name = request.json['collection']
    response_text = generate_response(question, collection_name)
    log_interaction(current_user.username, question, response_text, collection_name)
    return jsonify({'response': response_text}), 200

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text("text")
    except Exception as e:
        logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
    return text.strip()

def create_csv(data_list, output_csv):
    records = [{"filename": data["filename"], "content": data["content"]} for data in data_list]
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

CACHE_FILE = 'cache.json'

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            if content:
                return json.loads(content)
    return {}

def truncate_text(text, max_tokens):
    words = text.split()
    return ' '.join(words[:max_tokens])

def save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as file:
        json.dump(cache, file, ensure_ascii=False, indent=4)

def process_pdfs(pdf_directory):
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
        create_csv(data_list_ar, config.output_csv_ar)
    
    if data_list_fr:
        create_csv(data_list_fr, config.output_csv_fr)

def chunk_text(text, max_tokens=800):
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
    for _ in range(3):
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=content
            )
            embedding = response['data'][0]['embedding']
            logger.info(f"Successfully retrieved embedding for content: {content[:30]}")
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            time.sleep(2)
    return None

def collection_exists(collection_name):
    collections = qdrant_client.get_collections().collections
    return any(col.name == collection_name for col in collections)

def check_csv_content():
    if os.path.exists(config.output_csv_ar):
        df_ar = pd.read_csv(config.output_csv_ar)
        logger.info(f"Arabic CSV contains {len(df_ar)} records.")
    else:
        logger.info("Arabic CSV does not exist.")

    if os.path.exists(config.output_csv_fr):
        df_fr = pd.read_csv(config.output_csv_fr)
        logger.info(f"French CSV contains {len(df_fr)} records.")
    else:
        logger.info("French CSV does not exist.")

check_csv_content()

def vectorize_and_store(csv_path, collection_name):
    if collection_exists(collection_name):
        logger.info(f"Collection '{collection_name}' already exists. Skipping vectorization.")
        return
    
    df = pd.read_csv(csv_path)
    logger.info(f"Read {len(df)} records from {csv_path}")
    
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=1536,
            distance=models.Distance.COSINE
        )
    )
    
    points = []
    for index, row in df.iterrows():
        chunks = chunk_text(row['content'])
        for chunk in chunks:
            embedding = get_embedding(chunk)
            if embedding is None:
                logger.error(f"Failed to get embedding for chunk: {chunk[:30]}")
                continue
            point_id = str(uuid.uuid4())
            points.append(models.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "filename": row['filename'], 
                    "content": chunk,
                    "language": detect(chunk)
                }
            ))
    
    logger.info(f"Upserting {len(points)} points to {collection_name}")
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )

recognizer = sr.Recognizer()
mic_lock = Lock()

def recognize_speech_from_microphone(language="ar", callback=None):
    with mic_lock:
        mic = sr.Microphone()
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            logger.info("Listening...")
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
    text = re.sub(r'\*\*\*|\.{2,}', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\n', ' ')
    return text.strip()

def text_to_speech(text, language="ar"):
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
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    pygame.mixer.music.unpause()

def pause_audio():
    pygame.mixer.music.pause()

def replay_audio():
    pygame.mixer.music.stop()
    pygame.mixer.music.play()

@lru_cache(maxsize=100)
def cached_query_qdrant(question, collection_name):
    question_embedding = get_embedding(question)
    if question_embedding is None:
        return []

    if not collection_exists(collection_name):
        logger.info(f"Collection '{collection_name}' does not exist.")
        return []

    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=question_embedding,
        limit=3
    )

    return [hit.payload["content"] for hit in search_result]

def generate_response(question, collection_name, quality_mode="good", input_token_limit=2000, output_token_limit=2000, feedback=None):
    model = "gpt-4"
    max_tokens = 500

    if quality_mode == "premium":
        model = "gpt-4o"
        max_tokens = 2000
    elif quality_mode == "economy":
        model = "gpt-3.5-turbo"
        max_tokens = 300

    truncated_question = truncate_text(question, input_token_limit)

    relevant_chunks = cached_query_qdrant(truncated_question, collection_name)

    prompt = (
        "You are an AI assistant specialized in agricultural advice. Here are some relevant information chunks:\n"
        + "\n".join(f"- {chunk}" for chunk in relevant_chunks)
    )

    if feedback:
        prompt += f"\n\nUser feedback:\n{feedback}"

    prompt += f"\nNow answer the following question: {truncated_question}. Please provide a detailed and accurate response."

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
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Translate the following text to {target_language}."},
            {"role": "user", "content": text}
        ]
    )
    return response['choices'][0]['message']['content']

def translate_to_darija(text):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Translate the following text to Moroccan Darija using Arabic letters."},
            {"role": "user", "content": text}
        ]
    )
    return response['choices'][0]['message']['content']

def user_input_choice(input_lang, output_lang, user_input, input_type="text", quality_mode="good", input_token_limit=800, output_token_limit=500, feedback=None):
    if input_type == "voice":
        logger.info("Please speak into the microphone...")
        speech_text = recognize_speech_from_microphone(language=input_lang)
        logger.info(f"Recognized Speech: {speech_text}")
        response_text = generate_response(speech_text, "agriculture_ar" if input_lang == "ar" else "agriculture_fr", quality_mode, input_token_limit, output_token_limit, feedback)
        if output_lang == "dar":
            response_text = translate_to_darija(response_text)
        elif input_lang != output_lang:
            response_text = translate_text(response_text, output_lang)
        logger.info(f"Response: {response_text}")
        return response_text
    elif input_type == "text":
        logger.info(f"Received Text: {user_input}")
        response_text = generate_response(user_input, "agriculture_ar" if input_lang == "ar" else "agriculture_fr", quality_mode, input_token_limit, output_token_limit, feedback)
        if output_lang == "dar":
            response_text = translate_to_darija(response_text)
        elif input_lang != output_lang:
            response_text = translate_text(response_text, output_lang)
        logger.info(f"Response: {response_text}")
        return response_text
    else:
        logger.error("Invalid choice. Please enter 'voice' or 'text'.")
        return "Invalid choice."

def format_rtl_text(text):
    return f"\u202B{text}\u202C"

def log_interaction(user, question, response, collection_name):
    log_data = {
        "timestamp": time.time(),
        "user": user,
        "question": question,
        "response": response,
        "collection_name": collection_name
    }
    with open("interaction_logs.json", "a") as log_file:
        log_file.write(json.dumps(log_data) + "\n")
    
    with mlflow.start_run():
        mlflow.log_param("user", user)
        mlflow.log_param("question", question)
        mlflow.log_param("response", response)
        mlflow.log_param("collection_name", collection_name)
        mlflow.log_metric("timestamp", log_data["timestamp"])

def generate_report():
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

        self.relevance_var = ctk.IntVar()
        self.accuracy_var = ctk.IntVar()
        self.fluency_var = ctk.IntVar()

        self.show_login_window()

        self.cache = load_cache()

    def update_true_text(self):
        new_true_text = simpledialog.askstring("Update True Text", "Enter new true text:")
        if new_true_text:
            try:
                response = requests.post('http://localhost:5005/update_true_text', json={'true_text': new_true_text})
                response.raise_for_status()
                messagebox.showinfo("Success", "True text updated successfully")
            except requests.exceptions.RequestException as e:
                messagebox.showerror("Error", f"Failed to update true text: {e}")

    def show_login_window(self):
        self.login_window = ctk.CTkToplevel(self)
        self.login_window.title("Login")
        self.login_window.geometry("500x500")
        self.login_window.iconbitmap("icon.ico")

        logo_path = "uni.png"
        logo_image = Image.open(logo_path)
        logo_image = logo_image.resize((100, 100), Image.LANCZOS)
        logo_photo = ImageTk.PhotoImage(logo_image)
        ctk.CTkLabel(self.login_window, image=logo_photo, text="").grid(row=0, column=0, columnspan=2, pady=10)
        self.login_window.logo_photo = logo_photo

        ctk.CTkLabel(self.login_window, text="Username", font=("Helvetica", 14)).grid(row=1, column=0, padx=10, pady=10)
        ctk.CTkLabel(self.login_window, text="Password", font=("Helvetica", 14)).grid(row=2, column=0, padx=10, pady=10)

        self.username_entry = ctk.CTkEntry(self.login_window, font=("Helvetica", 14))
        self.password_entry = ctk.CTkEntry(self.login_window, show='*', font=("Helvetica", 14))

        self.username_entry.grid(row=1, column=1, padx=10, pady=10)
        self.password_entry.grid(row=2, column=1, padx=10, pady=10)

        ctk.CTkLabel(self.login_window, text="Language", font=("Helvetica", 14)).grid(row=3, column=0, padx=10, pady=10)
        self.language_var = ctk.StringVar()
        self.language_combobox = ctk.CTkComboBox(self.login_window, variable=self.language_var, values=["AR", "FR", "DAR"], font=("Helvetica", 14))
        self.language_combobox.grid(row=3, column=1, padx=10, pady=10)
        self.language_combobox.set("AR")

        ctk.CTkButton(self.login_window, text="Login", command=self.login, font=("Helvetica", 14)).grid(row=4, column=0, columnspan=2, pady=10)
        ctk.CTkButton(self.login_window, text="Register", command=self.register, font=("Helvetica", 14)).grid(row=5, column=0, columnspan=2, pady=10)

    def login(self):
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
        hashed_password = hash_password(password)
        for user in users.values():
            if user.username == username and user.password == hashed_password:
                return True
        return False

    def register(self):
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
        # Create a canvas and a scrollbar
        self.canvas = Canvas(self)
        self.scrollbar = Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(1, weight=1)
        self.scrollable_frame.grid_columnconfigure(2, weight=1)

        self.input_label = ctk.CTkLabel(self.scrollable_frame, text="Input Text:", font=("Helvetica", 14))
        self.input_label.grid(row=0, column=0, pady=5, sticky='nsew')

        self.input_scrollbar = Scrollbar(self.scrollable_frame, width=10)
        self.input_text = ctk.CTkTextbox(self.scrollable_frame, wrap='word', width=400, height=600, font=("Helvetica", 14), yscrollcommand=self.input_scrollbar.set)
        self.input_text.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')
        self.input_scrollbar.grid(row=1, column=1, sticky='nsew')
        self.input_scrollbar.config(command=self.input_text.yview)

        self.lang_label = ctk.CTkLabel(self.scrollable_frame, text="Select Input Language:", font=("Helvetica", 14))
        self.lang_label.grid(row=2, column=0, pady=5, sticky='nsew')
        self.input_lang = ctk.CTkComboBox(self.scrollable_frame, values=["ar", "fr", "dar"], font=("Helvetica", 14))
        self.input_lang.grid(row=3, column=0, pady=5, sticky='nsew')
        
        self.output_lang_label = ctk.CTkLabel(self.scrollable_frame, text="Select Output Language:", font=("Helvetica", 14))
        self.output_lang_label.grid(row=4, column=0, pady=5, sticky='nsew')
        self.output_lang = ctk.CTkComboBox(self.scrollable_frame, values=["ar", "fr", "dar"], font=("Helvetica", 14))
        self.output_lang.grid(row=5, column=0, pady=5, sticky='nsew')
        
        self.quality_label = ctk.CTkLabel(self.scrollable_frame, text="Select Quality Mode:", font=("Helvetica", 14))
        self.quality_label.grid(row=6, column=0, pady=5, sticky='nsew')
        self.quality_mode = ctk.CTkComboBox(self.scrollable_frame, values=["economy", "good", "premium"], font=("Helvetica", 14))
        self.quality_mode.grid(row=7, column=0, pady=5, sticky='nsew')

        self.input_token_label = ctk.CTkLabel(self.scrollable_frame, text="Input Token Limit:", font=("Helvetica", 14))
        self.input_token_label.grid(row=8, column=0, pady=5, sticky='nsew')
        self.input_token_limit = ctk.CTkEntry(self.scrollable_frame, font=("Helvetica", 14))
        self.input_token_limit.grid(row=9, column=0, pady=5, sticky='nsew')
        self.input_token_limit.insert(0, "800")

        self.output_token_label = ctk.CTkLabel(self.scrollable_frame, text="Output Token Limit:", font=("Helvetica", 14))
        self.output_token_label.grid(row=10, column=0, pady=5, sticky='nsew')
        self.output_token_limit = ctk.CTkEntry(self.scrollable_frame, font=("Helvetica", 14))
        self.output_token_limit.grid(row=11, column=0, pady=5, sticky='nsew')
        self.output_token_limit.insert(0, "500")

        self.submit_button = ctk.CTkButton(self.scrollable_frame, text="Submit", command=self.process_input, font=("Helvetica", 14))
        self.submit_button.grid(row=12, column=0, pady=5)

        self.output_label = ctk.CTkLabel(self.scrollable_frame, text="Output Text:", font=("Helvetica", 14))
        self.output_label.grid(row=0, column=2, pady=5, sticky='nsew')

        self.output_scrollbar = Scrollbar(self.scrollable_frame, width=10)
        self.output_text = ctk.CTkTextbox(self.scrollable_frame, wrap='word', width=400, height=600, font=("Helvetica", 14), yscrollcommand=self.output_scrollbar.set)
        self.output_text.grid(row=1, column=2, padx=5, pady=5, sticky='nsew')
        self.output_scrollbar.grid(row=1, column=3, sticky='nsew')
        self.output_scrollbar.config(command=self.output_text.yview)

        self.f1_score_value = ctk.CTkLabel(self.scrollable_frame, text="F1 Score: N/A", font=("Helvetica", 14))
        self.f1_score_value.grid(row=2, column=2, pady=5, sticky='nsew')
        self.rouge_l_score_value = ctk.CTkLabel(self.scrollable_frame, text="ROUGE-L: N/A", font=("Helvetica", 14))
        self.rouge_l_score_value.grid(row=3, column=2, pady=5, sticky='nsew')
        self.sacrebleu_score_value = ctk.CTkLabel(self.scrollable_frame, text="sacreBLEU: N/A", font=("Helvetica", 14))
        self.sacrebleu_score_value.grid(row=4, column=2, pady=5, sticky='nsew')

        self.record_button = ctk.CTkButton(self.scrollable_frame, text="Record", command=self.start_recording, font=("Helvetica", 14))
        self.record_button.grid(row=5, column=3, pady=5)

        self.stop_button = ctk.CTkButton(self.scrollable_frame, text="Stop Recording", command=self.stop_recording, font=("Helvetica", 14))
        self.stop_button.grid(row=6, column=3, pady=5)
        self.stop_button.configure(state="disabled")

        self.speak_button = ctk.CTkButton(self.scrollable_frame, text="Read Aloud Output", command=self.read_aloud_output, font=("Helvetica", 14))
        self.speak_button.grid(row=7, column=3, pady=5)

        self.play_button = ctk.CTkButton(self.scrollable_frame, text="▶ Play", command=play_audio, font=("Helvetica", 14))
        self.play_button.grid(row=8, column=3, pady=5)

        self.pause_button = ctk.CTkButton(self.scrollable_frame, text="⏸ Pause", command=pause_audio, font=("Helvetica", 14))
        self.pause_button.grid(row=9, column=3, pady=5)

        self.replay_button = ctk.CTkButton(self.scrollable_frame, text="⏪ Replay", command=replay_audio, font=("Helvetica", 14))
        self.replay_button.grid(row=10, column=3, pady=5)

        self.report_button = ctk.CTkButton(self.scrollable_frame, text="Generate Report", command=self.display_report, font=("Helvetica", 14))
        self.report_button.grid(row=11, column=3, pady=5)

        self.user_info_label = ctk.CTkLabel(self.scrollable_frame, text=f"Logged in as: {self.username}", font=("Helvetica", 14))
        self.user_info_label.grid(row=12, column=4, pady=5, sticky='w')

        self.logout_button = ctk.CTkButton(self.scrollable_frame, text="Logout", command=self.logout, font=("Helvetica", 14))
        self.logout_button.grid(row=12, column=5, pady=5, sticky='e')

        self.recording_label = ctk.CTkLabel(self.scrollable_frame, text="Recording... Please speak into the microphone", text_color="red", font=("Helvetica", 14))
        self.recording_label.grid(row=13, column=0, columnspan=3, pady=5)
        self.recording_label.grid_remove()

        self.transcription_label = ctk.CTkLabel(self.scrollable_frame, text="", text_color="blue", font=("Helvetica", 14))
        self.transcription_label.grid(row=14, column=0, columnspan=3, pady=5)
        self.transcription_label.grid_remove()

        self.feedback_label = ctk.CTkLabel(self.scrollable_frame, text="Feedback (Optional):", font=("Helvetica", 14))
        self.feedback_label.grid(row=15, column=0, pady=5, sticky='nsew')
        self.feedback_scrollbar = Scrollbar(self.scrollable_frame, width=10)
        self.feedback_text = ctk.CTkTextbox(self.scrollable_frame, wrap='word', width=10, height=10, font=("Helvetica", 14), yscrollcommand=self.feedback_scrollbar.set)
        self.feedback_text.grid(row=16, column=0, padx=5, pady=5, sticky='nsew')
        self.feedback_scrollbar.grid(row=16, column=1, sticky='nsew')
        self.feedback_scrollbar.config(command=self.feedback_text.yview)

        self.submit_feedback_button = ctk.CTkButton(self.scrollable_frame, text="Submit Feedback", command=self.submit_feedback, font=("Helvetica", 14))
        self.submit_feedback_button.grid(row=17, column=0, pady=5)

        self.show_user_info_button = ctk.CTkButton(self.scrollable_frame, text="Show User Info", command=self.show_user_info, font=("Helvetica", 14))
        self.show_user_info_button.grid(row=18, column=2, pady=5)

        self.update_user_button = ctk.CTkButton(self.scrollable_frame, text="Update User", command=self.update_user_details, font=("Helvetica", 14))
        self.update_user_button.grid(row=18, column=3, pady=5)

        self.delete_user_button = ctk.CTkButton(self.scrollable_frame, text="Delete User", command=self.delete_user, font=("Helvetica", 14))
        self.delete_user_button.grid(row=18, column=4, pady=5)

        self.update_true_text_button = ctk.CTkButton(self.scrollable_frame, text="Update True Text", command=self.update_true_text, font=("Helvetica", 14))
        self.update_true_text_button.grid(row=19, column=4, pady=5)

    def logout(self):
        self.username = None
        for widget in self.winfo_children():
            widget.destroy()
        self.show_login_window()

    def process_input(self):
        user_input = self.input_text.get("1.0", 'end').strip()
        input_lang = self.input_lang.get()
        output_lang = self.output_lang.get()
        quality_mode = self.quality_mode.get()
        input_token_limit = self.validate_token_limit(self.input_token_limit.get())
        output_token_limit = self.validate_token_limit(self.output_token_limit.get())
        additional_comments = self.feedback_text.get("1.0", 'end').strip()

        if not user_input or not input_lang or not output_lang or not quality_mode:
            messagebox.showerror("Error", "All fields must be filled")
            return

        if input_token_limit is None or output_token_limit is None:
            messagebox.showerror("Error", "Token limits must be valid integers")
            return

        cache_key = f"{self.username}:{quality_mode}:{input_lang}:{output_lang}:{user_input}"

        if cache_key in self.cache and not additional_comments:
            response_text = self.cache[cache_key]['response']
            feedback = self.cache[cache_key].get('feedback', None)
            self.update_output_text(response_text, feedback)
        else:
            Thread(target=self.run_user_input_choice, args=(input_lang, output_lang, user_input, "text", cache_key, quality_mode, input_token_limit, output_token_limit, additional_comments)).start()

    def validate_token_limit(self, token_limit):
        try:
            return int(token_limit)
        except ValueError:
            return None

    def start_recording(self):
        self.recording = True
        self.record_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.recording_label.grid()
        Thread(target=self.record_audio).start()

    def stop_recording(self):
        self.recording = False
        self.record_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.recording_label.grid_remove()

    def record_audio(self):
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

    def run_user_input_choice(self, input_lang, output_lang, user_input, input_type, cache_key, quality_mode, input_token_limit, output_token_limit, feedback=None):
        collection_name = "agriculture_ar" if input_lang == "ar" else "agriculture_fr"
        
        if not collection_exists(collection_name):
            response_text = f"The collection '{collection_name}' does not exist. Please ensure the data is processed and the collection is created."
        else:
            if input_type == "voice":
                response_text = generate_response(user_input, collection_name, quality_mode, input_token_limit, output_token_limit, feedback)
                if output_lang == "dar":
                    response_text = translate_to_darija(response_text)
                elif input_lang != output_lang:
                    response_text = translate_text(response_text, output_lang)
            else:
                response_text = user_input_choice(input_lang, output_lang, user_input, input_type, quality_mode, input_token_limit, output_token_limit, feedback)
            
            if output_lang == "ar":
                response_text = format_rtl_text(response_text)
        self.cache[cache_key] = {'response': response_text, 'feedback': feedback}
        save_cache(self.cache)
        self.after(0, self.update_output_text, response_text)

    def update_output_text(self, response_text, feedback=None):
        self.output_text.delete("1.0", 'end')
        self.output_text.insert('end', response_text)
        f1 = compute_f1_score(true_text, response_text)
        rouge_l = compute_rouge_l(true_text, response_text)
        sacrebleu_score = compute_sacrebleu(true_text, response_text).score

        self.f1_score_value.configure(text=f"F1 Score: {f1:.2f}")
        self.rouge_l_score_value.configure(text=f"ROUGE-L: {rouge_l:.2f}")
        self.sacrebleu_score_value.configure(text=f"sacreBLEU: {sacrebleu_score:.2f}")

        logger.info(f"F1 Score: {f1:.2f}, ROUGE-L: {rouge_l:.2f}, sacreBLEU: {sacrebleu_score:.2f}")

        cache_key = f"{self.username}:{self.quality_mode.get()}:{self.input_lang.get()}:{self.output_lang.get()}:{self.input_text.get('1.0', 'end').strip()}"
        if cache_key in self.cache and 'feedback' in self.cache[cache_key]:
            print("Add a feedback to improve the output if you wish.")
        
        if f1 < 50 or rouge_l < 50 or sacrebleu_score < 50:
                logger.info("Prompting for feedback...")
                self.prompt_for_feedback()

    def update_transcription_label(self, transcription):
        self.transcription_label.configure(text=f"Transcription: {transcription}")
    
    def on_feedback_window_close(self):
        self.feedback_window.grab_release()
        self.feedback_window.destroy()

    def read_aloud_output(self):
        output_lang = self.output_lang.get()
        response_text = self.output_text.get("1.0", 'end').strip()
        if response_text:
            Thread(target=text_to_speech, args=(response_text, output_lang)).start()

    def display_report(self):
        report = generate_report()
        report_window = ctk.CTkToplevel(self)
        report_window.title("Interaction Report")
        report_window.iconbitmap("icon.ico")
        report_scrollbar = Scrollbar(report_window, width=10)
        report_text = ctk.CTkTextbox(report_window, wrap='word', width=100, height=20, font=("Helvetica", 14), yscrollcommand=report_scrollbar.set)
        report_text.pack(padx=10, pady=10)
        report_scrollbar.pack(side='right', fill='y')
        report_scrollbar.config(command=report_text.yview)
        report_text.insert('end', report.to_string())

    def submit_feedback(self):
        feedback = self.feedback_text.get("1.0", 'end').strip()
        feedback_details = {
            "relevance": self.relevance_var.get(),
            "accuracy": self.accuracy_var.get(),
            "fluency": self.fluency_var.get(),
            "additional_comments": feedback
        }

        if not feedback:
            messagebox.showerror("Error", "Feedback cannot be empty")
            return

        log_data = {
            "timestamp": time.time(),
            "user": self.username,
            "feedback": feedback_details
        }
        with open("feedback_logs.json", "a") as log_file:
            log_file.write(json.dumps(log_data) + "\n")

        cache_key = f"{self.username}:{self.quality_mode.get()}:{self.input_lang.get()}:{self.output_lang.get()}:{self.input_text.get('1.0', 'end').strip()}"
        if cache_key in self.cache:
            self.cache[cache_key]['feedback'] = feedback_details

        save_cache(self.cache)

        with mlflow.start_run():
            mlflow.log_param("user", self.username)
            mlflow.log_param("feedback", feedback_details)
            mlflow.log_metric("timestamp", log_data["timestamp"])

        messagebox.showinfo("Success", "Feedback submitted successfully")
        self.feedback_text.delete("1.0", 'end')

        user_input = self.input_text.get("1.0", 'end').strip()
        input_lang = self.input_lang.get()
        output_lang = self.output_lang.get()
        quality_mode = self.quality_mode.get()
        input_token_limit = self.validate_token_limit(self.input_token_limit.get())
        output_token_limit = self.validate_token_limit(self.output_token_limit.get())
        logger.info("Re-generating response with feedback...")
        Thread(target=self.run_user_input_choice, args=(input_lang, output_lang, user_input, "text", cache_key, quality_mode, input_token_limit, output_token_limit, feedback_details)).start()

    def prompt_for_feedback(self):
        response = messagebox.askyesno("Feedback Request", "The response quality seems low. Would you like to submit feedback?")
        logger.info("Prompt for feedback called.")
        if response:
            self.feedback_label.grid()
            self.feedback_text.grid()
            self.submit_feedback_button.grid()
            logger.info("Opening feedback window...")
        else:
            self.feedback_label.grid_remove()
            self.feedback_text.grid_remove()
            self.submit_feedback_button.grid_remove()


    def submit_feedback_from_window(self):
        relevance = self.relevance_var.get()
        accuracy = self.accuracy_var.get()
        fluency = self.fluency_var.get()
        additional_comments = self.additional_comments_text.get("1.0", 'end').strip()

        if not additional_comments:
            messagebox.showerror("Error", "Feedback cannot be empty")
            return

        feedback_details = {
            "relevance": relevance,
            "accuracy": accuracy,
            "fluency": fluency,
            "additional_comments": additional_comments
        }

        log_data = {
            "timestamp": time.time(),
            "user": self.username,
            "feedback": feedback_details
        }
        with open("feedback_logs.json", "a") as log_file:
            log_file.write(json.dumps(log_data) + "\n")

        cache_key = f"{self.username}:{self.quality_mode.get()}:{self.input_lang.get()}:{self.output_lang.get()}:{self.input_text.get('1.0', 'end').strip()}"
        if cache_key in self.cache:
            self.cache[cache_key]['feedback'] = feedback_details

        save_cache(self.cache)

        with mlflow.start_run():
            mlflow.log_param("user", self.username)
            mlflow.log_param("feedback", feedback_details)
            mlflow.log_metric("timestamp", log_data["timestamp"])

        messagebox.showinfo("Success", "Feedback submitted successfully")
        self.feedback_window.destroy()

        user_input = self.input_text.get("1.0", 'end').strip()
        input_lang = self.input_lang.get()
        output_lang = self.output_lang.get()
        quality_mode = self.quality_mode.get()
        input_token_limit = self.validate_token_limit(self.input_token_limit.get())
        output_token_limit = self.validate_token_limit(self.output_token_limit.get())
        Thread(target=self.run_user_input_choice, args=(input_lang, output_lang, user_input, "text", cache_key, quality_mode, input_token_limit, output_token_limit, feedback_details)).start()

    def show_user_info(self):
        try:
            response = requests.get('http://localhost:5005/user')
            response.raise_for_status()
            user_info = response.json()
            messagebox.showinfo("User Info", f"Username: {user_info['username']}\nID: {user_info['id']}")
        except requests.exceptions.RequestException as e:
            messagebox.showerror("Error", f"Failed to fetch user info: {e}")

    def update_user_details(self):
        new_username = simpledialog.askstring("Update Username", "Enter new username:")
        new_password = simpledialog.askstring("Update Password", "Enter new password:", show='*')
        data = {}
        if new_username:
            data['username'] = new_username
        if new_password:
            data['password'] = new_password
        try:
            response = requests.put('http://localhost:5005/user', json=data)
            response.raise_for_status()
            messagebox.showinfo("Success", "User details updated successfully")
        except requests.exceptions.RequestException as e:
            messagebox.showerror("Error", f"Failed to update user details: {e}")

    def delete_user(self):
        response = messagebox.askyesno("Delete User", "Are you sure you want to delete your account?")
        if response:
            try:
                response = requests.delete('http://localhost:5005/user')
                response.raise_for_status()
                messagebox.showinfo("Success", "User deleted successfully")
                self.logout()
            except requests.exceptions.RequestException as e:
                messagebox.showerror("Error", f"Failed to delete user: {e}")

if __name__ == "__main__":
    def run_flask_app():
        process_pdfs(config.pdf_directory)
        
        check_csv_content()
        
        if not collection_exists("agriculture_ar"):
            try:
                vectorize_and_store(config.output_csv_ar, "agriculture_ar")
            except Exception as e:
                logger.error(f"Error creating 'agriculture_ar' collection: {e}")

        if not collection_exists("agriculture_fr"):
            try:
                vectorize_and_store(config.output_csv_fr, "agriculture_fr")
            except Exception as e:
                logger.error(f"Error creating 'agriculture_fr' collection: {e}")
                
        flask_app.run(host="localhost", port=5005, debug=False)

    Thread(target=run_flask_app).start()

    app = Application()
    app.mainloop()
