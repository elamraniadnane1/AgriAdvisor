import os
import uuid
import hashlib
import re
import time
import json
import threading
import tempfile
import wave
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
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
from tkinter import Scrollbar, messagebox, Canvas, Frame, Scale, HORIZONTAL
from tkinter import simpledialog
import requests
from threading import Thread, Lock
from functools import lru_cache
from PIL import Image, ImageTk
from sklearn.metrics import f1_score
from rouge_score import rouge_scorer
import sacrebleu
from prometheus_flask_exporter import PrometheusMetrics
import logging
from config import Config
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Scale, HORIZONTAL  # Add this import for the Scale widget
import webbrowser
import psutil

# Directory and output file paths
PDF_DIRECTORY = r"C:\Users\LENOVO\OneDrive\Bureau\Dataset"
OUTPUT_CSV_AR = r"C:\Users\LENOVO\OneDrive\Bureau\Dataset\agriculture_data_ar_.csv"
OUTPUT_CSV_FR = r"C:\Users\LENOVO\OneDrive\Bureau\Dataset\agriculture_data_fr_.csv"

class NewFileHandler(FileSystemEventHandler):
    def __init__(self, pdf_directory, output_csv_ar, output_csv_fr):
        self.pdf_directory = pdf_directory
        self.output_csv_ar = output_csv_ar
        self.output_csv_fr = output_csv_fr

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(".pdf"):
            logger.info(f"New PDF detected: {event.src_path}")
            self.process_new_pdf(event.src_path)

    def extract_text_from_pdf(self, pdf_path):
        text = ""
        logger.info(f"Starting text extraction from PDF: {pdf_path}")
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text("text")
            logger.info(f"Text extraction completed for PDF: {pdf_path}")
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
        return text.strip()

    def get_embedding(self, content):
        for _ in range(3):
            try:
                logger.info(f"Attempting to retrieve embedding (Attempt {attempt + 1}/3) for content: {content[:30]}")
                response = openai.Embedding.create(
                    model="text-embedding-3-large",
                    input=content
                )
                embedding = response['data'][0]['embedding']
                logger.info(f"Successfully retrieved embedding for content: {content[:30]}")
                return embedding
            except Exception as e:
                logger.error(f"Error getting embedding: {e}")
                time.sleep(2)
        return None
    def chunk_text(self, text, max_tokens=4000):
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

    def translate_text_to_french(self, text):
        max_chunk_size = 4000  # Adjust based on token limits
        translated_chunks = []
        chunks = self.chunk_text(text, max_chunk_size)
        total_chunks = len(chunks)
        logger.info(f"Starting translation of text to French, total chunks: {total_chunks}")
        try:
            for i, chunk in enumerate(chunks):
                logger.info(f"Translating chunk {i + 1} of {total_chunks}")
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "Translate the following text to French."},
                        {"role": "user", "content": chunk}
                    ]
                )
                translated_chunk = response['choices'][0]['message']['content']
                translated_chunks.append(translated_chunk)

                # Vectorize the translated chunk and store it in the collection
                embedding = self.get_embedding(translated_chunk)
                if embedding:
                    point_id = str(uuid.uuid4())
                    qdrant_client.upsert(
                        collection_name="agriculture_fr",
                        points=[
                            models.PointStruct(
                                id=point_id,
                                vector=embedding,
                                payload={
                                    "filename": "translated_chunk",
                                    "content": translated_chunk,
                                    "language": "fr"
                                }
                            )
                        ]
                    )
                    logger.info(f"Inserted translated chunk {i + 1} of {total_chunks} into the collection")
            return " ".join(translated_chunks)
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text  # Return the original text if translation fails

    def process_new_pdf(self, pdf_path):
        logger.info(f"Processing new PDF: {pdf_path}")
        text = self.extract_text_from_pdf(pdf_path)
        try:
            language = detect(text)
            logger.info(f"Detected language for PDF {pdf_path}: {language}")
        except Exception as e:
            logger.error(f"Language detection failed for PDF {pdf_path}: {e}")
            language = "unknown"
        
        data = {"filename": os.path.basename(pdf_path), "content": text}
        
        if language == "ar":
            self.append_to_csv(data, self.output_csv_ar)
        elif language == "fr":
            self.append_to_csv(data, self.output_csv_fr)
        elif language == "en":
            logger.info(f"Translating English PDF to French: {pdf_path}")
            translated_text = self.translate_text_to_french(text)
            data["content"] = translated_text
            self.append_to_csv(data, self.output_csv_fr)
        else:
            logger.info(f"Detected language '{language}' is not supported for PDF {pdf_path}")


    def append_to_csv(self, data, csv_path):
        try:
            if not os.path.exists(csv_path):
                df = pd.DataFrame([data])
            else:
                df = pd.read_csv(csv_path)
                df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"Appended data to {csv_path}")
        except Exception as e:
            logger.error(f"Failed to append data to CSV {csv_path}: {e}")

def monitor_directory(pdf_directory, output_csv_ar, output_csv_fr):
    event_handler = NewFileHandler(pdf_directory, output_csv_ar, output_csv_fr)
    observer = Observer()
    observer.schedule(event_handler, path=pdf_directory, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()



def run_user_input_choice(user_info, input_lang, output_lang, user_input, input_type, cache_key, quality_mode, input_token_limit, output_token_limit, feedback=None):
        print(f"Debug: user_info = {user_info}")
        logger.info(f"Debug: user_info (type={type(user_info)}) = {user_info}")
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
                response_text = user_input_choice(user_info, input_lang, output_lang, user_input, input_type, quality_mode, input_token_limit, output_token_limit, feedback)

            if output_lang == "ar":
                response_text = format_rtl_text(response_text)
        app.cache[cache_key] = {'response': response_text, 'feedback': feedback}
        save_cache(app.cache)
        app.after(0, app.update_output_text, response_text)


# Initialize logging
log_file = 'application.log'
if not os.path.exists(log_file):
    open(log_file, 'w').close()  # Create the log file if it does not exist

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
metrics = PrometheusMetrics(flask_app)
login_manager.login_view = 'login'

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

def show_help(self):
    def open_docs():
        webbrowser.open("index.html")  # Adjust the path if necessary

    help_window = ctk.CTkToplevel(self)
    help_window.title("Help")
    help_window.geometry("800x600")
    help_window.iconbitmap("icon.ico")

    help_text = """
    Welcome to the AI AgriAdvisor application!

    Here are some features you can use:
    - **Login/Register**: Use the login screen to enter your credentials. If you don't have an account, you can register for one.
    - **Input Text**: Enter the text you want to process in the "Input Text" field.
    - **Select Input/Output Language**: Choose the input and output languages from the dropdown menus.
    - **Quality Mode**: Select the desired quality mode for the response (economy, good, premium).
    - **Token Limits**: Set the input and output token limits.
    - **Submit**: Click "Submit" to process the input text and get a response.
    - **Record Audio**: Click "Record" to start recording your voice, and "Stop Recording" to end it. The application will transcribe your speech.
    - **Read Aloud Output**: Click "Read Aloud Output" to listen to the response.
    - **Volume Control**: Adjust the volume using the slider.
    - **Seek Bar**: Use the seek bar to navigate through the audio.
    - **Generate Report**: Click "Generate Report" to view the interaction report.
    - **Feedback**: Provide feedback on the response by filling out the feedback form.

    For more detailed instructions, visit our documentation.
    """

    help_label = ctk.CTkLabel(help_window, text=help_text, wraplength=700, font=("Helvetica", 14))
    help_label.pack(padx=20, pady=20)

    docs_button = ctk.CTkButton(help_window, text="Open Documentation", command=open_docs, font=("Helvetica", 14))
    docs_button.pack(pady=10)

def show_dashboard(self):
    webbrowser.open('http://localhost:5005/dashboard')


class User(UserMixin):
    def __init__(self, id, username, password,authenticated=False):
        self.id = id
        self.username = username
        self.password = password
        self.authenticated = authenticated

    @property
    def is_authenticated(self):
        return self.authenticated

    def authenticate(self):
        self.authenticated = True

    def deauthenticate(self):
        self.authenticated = False

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            if content:
                try:
                    user_data = json.loads(content)
                    for user_id, data in user_data.items():
                        users[user_id] = User(user_id, data['username'], data['password'])
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON file {USERS_FILE}: {e}")
            else:
                logger.warning(f"{USERS_FILE} is empty.")
    else:
        logger.warning(f"{USERS_FILE} does not exist.")
load_users()
def save_users():
    with open(USERS_FILE, 'w', encoding='utf-8') as file:
        json.dump({user_id: {'username': user.username, 'password': user.password} for user_id, user in users.items()}, file, ensure_ascii=False, indent=4)

@login_manager.user_loader
def load_user(user_id):
    user = users.get(user_id)
    if user:
        logger.info(f"User {user.username} loaded successfully")
        session['user_id'] = user.id  # Ensure user_id is in the session
    else:
        logger.error(f"User ID {user_id} not found in users")
    return user




def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@flask_app.route('/feedback', methods=['POST'])
@login_required
def submit_feedback():
    feedback = request.json.get('feedback')
    # Implement feedback handling logic here
    return jsonify({'message': 'Feedback submitted successfully'}), 200

@flask_app.route('/user_activity', methods=['GET'])
@login_required
def get_user_activity():
    user_id = current_user.id
    # Implement logic to fetch and return user activity
    return jsonify({'activity': 'User activity data'}), 200

@flask_app.route('/update_password', methods=['PUT'])
@login_required
def update_password():
    new_password = request.json.get('new_password')
    # Implement password update logic here
    return jsonify({'message': 'Password updated successfully'}), 200

@flask_app.route('/admin/users', methods=['GET'])
@login_required
def get_all_users():
    # Implement logic to fetch and return all users for admin
    return jsonify({'users': 'List of all users'}), 200

@flask_app.route('/admin/user/<user_id>', methods=['DELETE'])
@login_required
def delete_user_by_admin(user_id):
    # Implement logic to delete a specific user by admin
    return jsonify({'message': f'User {user_id} deleted successfully'}), 200


from flask import render_template

@flask_app.route('/dashboard', methods=['GET'])
@login_required
def dashboard():
    interactions = generate_report()

    if interactions.empty:
        logger.error("No interactions available to generate the report.")
        return "No interactions available to generate the report.", 400

    required_columns = ['question', 'timestamp', 'user']
    missing_columns = [col for col in required_columns if col not in interactions.columns]

    if missing_columns:
        logger.error(f"Missing necessary columns in interactions data: {missing_columns}")
        return "Insufficient data to generate the report.", 400

    common_queries = interactions['question'].value_counts().head(10).to_dict()
    user_stats = interactions['user'].value_counts().to_dict()

    # Calculate performance metrics
    average_response_time = interactions['timestamp'].diff().mean()
    # Ensure average_response_time is in seconds
    if isinstance(average_response_time, pd.Timedelta):
        average_response_time = average_response_time.total_seconds()  # Convert to seconds
    elif isinstance(average_response_time, np.float64):
        average_response_time = float(average_response_time)  # Ensure it's a float

    total_interactions = len(interactions)
    unique_users = interactions['user'].nunique()
    most_active_user = interactions['user'].value_counts().idxmax()
    most_active_user_interactions = interactions['user'].value_counts().max()
    least_active_user = interactions['user'].value_counts().idxmin()
    least_active_user_interactions = interactions['user'].value_counts().min()
    time_of_first_interaction = interactions['timestamp'].min()
    time_of_last_interaction = interactions['timestamp'].max()
    interaction_rate_per_user = total_interactions / unique_users if unique_users else 0
    most_common_query = interactions['question'].value_counts().idxmax()
    most_common_query_count = interactions['question'].value_counts().max()
    total_unique_queries = interactions['question'].nunique()

    performance_metrics = {
        'average_response_time': average_response_time,
        'total_interactions': total_interactions,
        'unique_users': unique_users,
        'most_active_user': most_active_user,
        'most_active_user_interactions': most_active_user_interactions,
        'least_active_user': least_active_user,
        'least_active_user_interactions': least_active_user_interactions,
        'time_of_first_interaction': time_of_first_interaction,
        'time_of_last_interaction': time_of_last_interaction,
        'interaction_rate_per_user': interaction_rate_per_user,
        'most_common_query': most_common_query,
        'most_common_query_count': most_common_query_count,
        'total_unique_queries': total_unique_queries
    }

    return render_template('dashboard.html', common_queries=common_queries, user_stats=user_stats, performance_metrics=performance_metrics)




from flask import session
@flask_app.route('/user', methods=['GET'])
@login_required
def get_user_info():
    logger.info(f"Session data: {session}")
    logger.info(f"Current user: {current_user}")
    if 'user_id' not in session:
        return jsonify({'error': 'User not authenticated'}), 401

    user_info = {
        'username': current_user.username,
        'id': current_user.id,
        'authenticated' : True
    }
    logger.info(f"User info: {user_info}")
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

@flask_app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.json['username']
        password = hash_password(request.json['password'])
        logger.info(f"Attempting login for user: {username}")
        for user in users.values():
            if user.username == username and user.password == password:
                user.authenticate()  # Mark the user as authenticated
                login_user(user)
                next_page = request.args.get('next')
                logger.info(f"User {username} logged in successfully")
                user_info = {
                    'id': user.id,
                    'username': user.username,
                    'authenticated': user.is_authenticated
                }
                if next_page:
                    return jsonify({'message': 'Login successful', 'redirect': next_page, 'user_info': user_info}), 200
                else:
                    return jsonify({'message': 'Login successful', 'redirect': '/dashboard', 'user_info': user_info}), 200
        logger.error("Invalid credentials")
        return jsonify({'message': 'Invalid credentials'}), 401
    return render_template('login.html')  # Ensure you have a login.html template



@flask_app.route('/logout', methods=['POST'])
@login_required
def logout():
    current_user.deauthenticate() 
    logout_user()
    return jsonify({'message': 'Logged out successfully'}), 200

@flask_app.route('/query', methods=['POST'])
@login_required
@metrics.counter('requests_by_user', 'Number of requests by user', labels={'username': lambda: current_user.username})
def query():
    question = request.json['question']
    collection_name = request.json['collection']
    logger.info(f"Received query: {question} for collection: {collection_name}")
    quality_mode = request.json.get('quality_mode', 'good')
    try:
        response_text = generate_response(question, collection_name)
        #log_interaction(question, response_text, collection_name,current_user.username)
        logger.info(f"Generated response: {response_text}")
        return jsonify({'response': response_text}), 200
    except Exception as e:
        logger.error(f"Failed to process query: {e}")
        return jsonify({'error': 'Failed to process query'}), 500

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
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON file {CACHE_FILE}: {e}")
            else:
                logger.warning(f"{CACHE_FILE} is empty.")
    else:
        logger.warning(f"{CACHE_FILE} does not exist.")
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
                model="text-embedding-3-large",
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
        chunk_size = 1000  # Define your chunk size here
        chunk_list = []  # Append each chunk df here

        for chunk in pd.read_csv(config.output_csv_ar, chunksize=chunk_size):
            chunk_list.append(chunk)

        df_ar = pd.concat(chunk_list, axis=0)
        logger.info(f"Arabic CSV contains {len(df_ar)} records.")
    else:
        logger.info("Arabic CSV does not exist.")

    if os.path.exists(config.output_csv_fr):
        chunk_size = 1000  # Define your chunk size here
        chunk_list = []  # Append each chunk df here

        for chunk in pd.read_csv(config.output_csv_fr, chunksize=chunk_size):
            chunk_list.append(chunk)

        df_fr = pd.concat(chunk_list, axis=0)
        logger.info(f"French CSV contains {len(df_fr)} records.")
    else:
        logger.info("French CSV does not exist.")

check_csv_content()

def vectorize_and_store(csv_path, collection_name):
    logger.info(f"Starting vectorization for {collection_name}...")
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
    logger.info(f"Completed vectorization for {collection_name}.")

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

def text_to_speech(self, text, language="ar"):
    if language == "dar":
        language = "ar"
    
    clean_text = clean_text_for_speech(text)
    tts = gTTS(text=clean_text, lang=language)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        temp_filename = fp.name
        tts.save(temp_filename)
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    pygame.mixer.music.load(temp_filename)
    pygame.mixer.music.play()

def play_audio():
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    pygame.mixer.music.unpause()

def pause_audio():
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    pygame.mixer.music.pause()

def replay_audio():
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    pygame.mixer.music.stop()
    pygame.mixer.music.play()


def set_volume(self, volume_level):
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    volume = int(volume_level) / 100
    pygame.mixer.music.set_volume(volume)


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

    if current_user and current_user.is_authenticated:
            logger.info(f"Current user: {current_user.username}, Authenticated: {current_user.is_authenticated}")
    else:
            logger.error("Current user is None")
    
    model_mapping = {
        "premium": "gpt-4o",
        "good": "gpt-4",
        "economy": "gpt-3.5-turbo"
    }
    
    model = model_mapping.get(quality_mode, "gpt-4")
    max_tokens_mapping = {
        "premium": 2000,
        "good": 500,
        "economy": 300
    }
    
    max_tokens = max_tokens_mapping.get(quality_mode, 500)
    
    try:
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

        response_text = response['choices'][0]['message']['content']


        #log_interaction(question, response_text, collection_name,current_user.username)
        
        return response_text
    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        return "An error occurred while generating the response."
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return "An unexpected error occurred while generating the response."


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

import psutil

def print_system_usage():
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_times = psutil.cpu_times()
    print(f"CPU Usage: {cpu_percent}%")
    print(f"CPU Times: user={cpu_times.user}, system={cpu_times.system}, idle={cpu_times.idle}")

    # Memory usage
    virtual_memory = psutil.virtual_memory()
    print(f"Memory Usage: {virtual_memory.percent}%")
    print(f"Memory Details: total={virtual_memory.total / (1024 ** 3):.2f} GB, available={virtual_memory.available / (1024 ** 3):.2f} GB, used={virtual_memory.used / (1024 ** 3):.2f} GB, free={virtual_memory.free / (1024 ** 3):.2f} GB")

    # Disk usage
    disk_usage = psutil.disk_usage('/')
    print(f"Disk Usage: {disk_usage.percent}%")
    print(f"Disk Details: total={disk_usage.total / (1024 ** 3):.2f} GB, used={disk_usage.used / (1024 ** 3):.2f} GB, free={disk_usage.free / (1024 ** 3):.2f} GB")

    # Network usage
    net_io = psutil.net_io_counters()
    print(f"Network Usage: bytes_sent={net_io.bytes_sent / (1024 ** 2):.2f} MB, bytes_recv={net_io.bytes_recv / (1024 ** 2):.2f} MB")


def user_input_choice(user_info, input_lang, output_lang, user_input, input_type="text", quality_mode="good", input_token_limit=800, output_token_limit=500, feedback=None):
    logger.info(f"Debug: user_info in user_input_choice (type={type(user_info)}) = {user_info}")

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

def log_interaction(question, response, collection_name, user):
    log_data = {
        "timestamp": time.time(),
        "question": question,
        "response": response,
        "collection_name": collection_name,
        "user": user
    }
    # Log interaction to JSON file
    log_file_path = "interaction_logs.json"
    
    try:
        # Check if the log file exists
        if not os.path.exists(log_file_path):
            with open(log_file_path, "w", encoding='utf-8') as log_file:
                json.dump([log_data], log_file, ensure_ascii=False, indent=4)
                logger.info("interaction_logs.json file created and interaction logged.")
        else:
            # Load existing log data
            with open(log_file_path, "r+", encoding='utf-8') as log_file:
                try:
                    data = json.load(log_file)
                except json.JSONDecodeError:
                    data = []
                data.append(log_data)
                log_file.seek(0)
                json.dump(data, log_file, ensure_ascii=False, indent=4)
                logger.info("Logged interaction: %s", log_data)
    except Exception as e:
        logger.error("Failed to log interaction: %s", e)

    # Log interaction to MLflow
    try:
        with mlflow.start_run():
            mlflow.log_param("user", user)
            mlflow.log_param("question", question)
            mlflow.log_param("response", response)
            mlflow.log_param("collection_name", collection_name)
            mlflow.log_param("user", log_data["user"])
            mlflow.log_metric("timestamp", log_data["timestamp"])
        logger.info("Logged interaction to MLflow: %s", log_data)
    except Exception as e:
        logger.error("Failed to log interaction to MLflow: %s", e)







def generate_report():
    interactions = []
    try:
        if os.path.exists("interaction_logs.json"):
            with open("interaction_logs.json", "r", encoding='utf-8') as log_file:
                data = json.load(log_file)
                interactions.extend(data)
                logger.info(f"Interactions loaded: {interactions}")
        else:
            logger.warning("interaction_logs.json file not found.")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON file: {e}")
    except Exception as e:
        logger.error(f"Unexpected error reading interaction logs: {e}")

    df = pd.DataFrame(interactions)
    
    # Ensure required columns are present
    required_columns = ['question', 'timestamp', 'user']
    for col in required_columns:
        if col not in df.columns:
            df[col] = None  # or handle as appropriate
            logger.warning(f"Added missing column: {col}")

    logger.info(f"Generated report with {len(df)} interactions")
    return df


class Application(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI AgriAdvisor")
        self.geometry("1200x800")
        self.iconbitmap("icon.ico")

        # Update this path to your new icon file
        icon_path = "icon.ico"
        icon_image = Image.open(icon_path)
        icon_photo = ImageTk.PhotoImage(icon_image)
        
        self.iconphoto(False, icon_photo)  # Set the new icon
        
        self.user_info = None  # Initialize user_info to None

        self.username = None
        self.user_info = {}
        self.recording = False
        self.frames = []

        self.relevance_var = ctk.IntVar()
        self.accuracy_var = ctk.IntVar()
        self.fluency_var = ctk.IntVar()

        self.theme = "light"  # Default theme
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

    def text_to_speech(self, text, language="ar"):
        if language == "dar":
            language = "ar"
        
        clean_text = clean_text_for_speech(text)
        tts = gTTS(text=clean_text, lang=language)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_filename = fp.name
            tts.save(temp_filename)
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()

    def show_login_window(self):
        self.login_window = ctk.CTkToplevel(self)
        self.login_window.title("Login")
        self.login_window.geometry("700x700")
        self.login_window.iconbitmap("icon.ico")

        logo_path = "uni.png"
        logo_image = Image.open(logo_path)
        logo_image = logo_image.resize((120, 120), Image.LANCZOS)
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
    # In the Application class __init__ method
    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        logger.info(f"Attempting GUI login for user: {username}")

        response = requests.post('http://localhost:5005/login', json={'username': username, 'password': password})
        if response.status_code == 200:
            response_data = response.json()
            self.username = username
            self.user_info = response_data.get('user_info', {})
            if not self.user_info:
                self.user_info = {}
            logger.info(f"user_info set to: {self.user_info}")
            self.user_info['authenticated'] = True
            self.language = self.language_combobox.get() 
            self.login_window.destroy()
            self.show_main_window()
            self.translate_app() 
            logger.info(f"User {username} logged in successfully via GUI")
        else:
            logger.error("Invalid credentials from GUI")
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
    def show_dashboard(self):
        webbrowser.open('http://localhost:5005/dashboard')


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

    def show_help(self):
        def open_docs():
            webbrowser.open("index.html")  # Adjust the path if necessary

        help_window = ctk.CTkToplevel(self)
        help_window.title("Help")
        help_window.geometry("800x600")
        help_window.iconbitmap("icon.ico")

        help_text = """
        Welcome to the AI AgriAdvisor application!

        Here are some features you can use:
        - **Login/Register**: Use the login screen to enter your credentials. If you don't have an account, you can register for one.
        - **Input Text**: Enter the text you want to process in the "Input Text" field.
        - **Select Input/Output Language**: Choose the input and output languages from the dropdown menus.
        - **Quality Mode**: Select the desired quality mode for the response (economy, good, premium).
        - **Token Limits**: Set the input and output token limits.
        - **Submit**: Click "Submit" to process the input text and get a response.
        - **Record Audio**: Click "Record" to start recording your voice, and "Stop Recording" to end it. The application will transcribe your speech.
        - **Read Aloud Output**: Click "Read Aloud Output" to listen to the response.
        - **Volume Control**: Adjust the volume using the slider.
        - **Seek Bar**: Use the seek bar to navigate through the audio.
        - **Generate Report**: Click "Generate Report" to view the interaction report.
        - **Feedback**: Provide feedback on the response by filling out the feedback form.

        For more detailed instructions, visit our documentation.
        """

        help_label = ctk.CTkLabel(help_window, text=help_text, wraplength=700, font=("Helvetica", 14))
        help_label.pack(padx=20, pady=20)

        docs_button = ctk.CTkButton(help_window, text="Open Documentation", command=open_docs, font=("Helvetica", 14))
        docs_button.pack(pady=10)


    def show_main_window(self):
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
        self.input_text = ctk.CTkTextbox(self.scrollable_frame, wrap='word', font=("Helvetica", 14), yscrollcommand=self.input_scrollbar.set)
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
        self.output_text = ctk.CTkTextbox(self.scrollable_frame, wrap='word', font=("Helvetica", 14), yscrollcommand=self.output_scrollbar.set)
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

        # Add volume control slider
        self.volume_label = ctk.CTkLabel(self.scrollable_frame, text="Volume Control", font=("Helvetica", 14))
        self.volume_label.grid(row=12, column=3, pady=5, sticky='nsew')
        self.volume_slider = Scale(self.scrollable_frame, from_=0, to=100, orient=HORIZONTAL, command=self.set_volume)
        self.volume_slider.set(50)
        self.volume_slider.grid(row=13, column=3, pady=5, sticky='nsew')

        # Add seek bar
        self.seek_label = ctk.CTkLabel(self.scrollable_frame, text="Seek Bar", font=("Helvetica", 14))
        self.seek_label.grid(row=14, column=3, pady=5, sticky='nsew')
        self.seek_slider = Scale(self.scrollable_frame, from_=0, to=100, orient=HORIZONTAL, command=self.seek_audio)
        self.seek_slider.grid(row=15, column=3, pady=5, sticky='nsew')

        self.help_button = ctk.CTkButton(self.scrollable_frame, text="Help", command=self.show_help, font=("Helvetica", 14))
        self.help_button.grid(row=20, column=3, pady=5)

        self.dashboard_button = ctk.CTkButton(self.scrollable_frame, text="Dashboard", command=self.show_dashboard, font=("Helvetica", 14))
        self.dashboard_button.grid(row=21, column=3, pady=5)



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

        # Add theme toggle button
        self.theme_toggle_button = ctk.CTkButton(self.scrollable_frame, text="Toggle Theme", command=self.toggle_theme, font=("Helvetica", 14))
        self.theme_toggle_button.grid(row=20, column=4, pady=5)

        # Ensure the main window grid resizes with the window
        for i in range(21):
            self.scrollable_frame.grid_rowconfigure(i, weight=1)
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(2, weight=1)
        self.scrollable_frame.grid_columnconfigure(3, weight=1)
        self.scrollable_frame.grid_columnconfigure(4, weight=1)
        self.scrollable_frame.grid_columnconfigure(5, weight=1)

    def toggle_theme(self):
        if self.theme == "light":
            self.theme = "dark"
            ctk.set_appearance_mode("dark")
        else:
            self.theme = "light"
            ctk.set_appearance_mode("light")

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

        # Validate required fields
        if not user_input or not input_lang or not output_lang or not quality_mode:
            messagebox.showerror("Error", "All fields must be filled")
            return

        # Validate token limits
        if input_token_limit is None or output_token_limit is None:
            messagebox.showerror("Error", "Token limits must be valid integers")
            return

        # Check for user authentication
        user_info = self.user_info  # Use the stored user information
        logger.info(f"process_input user_info: {user_info}")
    

        # Generate cache key
        cache_key = f"{self.username}:{quality_mode}:{input_lang}:{output_lang}:{user_input}"

        # Check cache for response
        if cache_key in self.cache and not additional_comments:
            response_text = self.cache[cache_key]['response']
            feedback = self.cache[cache_key].get('feedback', None)
            self.update_output_text(response_text, feedback)
        else:
            # Add logging to ensure user_info is correctly passed
            logger.info(f"Passing user_info to run_user_input_choice: {user_info}")
            # Process user input in a separate thread to keep UI responsive
            Thread(target=run_user_input_choice, args=(
                user_info, 
                input_lang, 
                output_lang, 
                user_input, 
                "text", 
                cache_key, 
                quality_mode,
                input_token_limit, 
                output_token_limit, 
                additional_comments
            )).start()


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

    def run_user_input_choice(user_info, input_lang, output_lang, user_input, input_type, cache_key, quality_mode, input_token_limit, output_token_limit, feedback=None):
        print(f"Debug: user_info = {user_info}")
        logger.info(f"Debug: user_info (type={type(user_info)}) = {user_info}")

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
                response_text = user_input_choice(user_info, input_lang, output_lang, user_input, input_type, quality_mode, input_token_limit, output_token_limit, feedback)

            if output_lang == "ar":
                response_text = format_rtl_text(response_text)
        app.cache[cache_key] = {'response': response_text, 'feedback': feedback}
        save_cache(app.cache)
        app.after(0, app.update_output_text, response_text)






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
            Thread(target=self.text_to_speech, args=(response_text, output_lang)).start()


    def display_report(self):
        report = generate_report()
        
        report_window = ctk.CTkToplevel(self)
        report_window.title("Interaction Report")
        report_window.geometry("1200x800")  # Set higher resolution for the report window
        report_window.iconbitmap("icon.ico")
        
        report_frame = ctk.CTkFrame(report_window)
        report_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        report_scrollbar = Scrollbar(report_frame, orient="vertical")
        report_text = ctk.CTkTextbox(report_frame, wrap='word', font=("Helvetica", 14), yscrollcommand=report_scrollbar.set)
        report_scrollbar.config(command=report_text.yview)
        
        report_scrollbar.pack(side="right", fill="y")
        report_text.pack(side="left", fill="both", expand=True)
        
        # Format the DataFrame as a string with improved appearance
        report_string = report.to_string(index=False)
        report_text.insert('end', report_string)


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
        except json.JSONDecodeError as e:
            messagebox.showerror("Error", f"Failed to decode JSON response: {e}")


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

    def set_volume(self, volume_level):
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        volume = int(volume_level) / 100
        pygame.mixer.music.set_volume(volume)

    def seek_audio(self, seek_position):
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pos = int(seek_position)
        if pygame.mixer.music.get_busy():
            total_length = pygame.mixer.Sound(self.audio_filename).get_length()
            seek_time = (pos / 100) * total_length
            pygame.mixer.music.play(start=seek_time)

import logging

def print_recent_logs():
    with open('application.log', 'r') as log_file:
        logs = log_file.readlines()
        print("Recent Log Entries (last 10):")
        for log in logs[-10:]:
            print(log.strip())

print_recent_logs()

def print_performance_metrics():
    interactions = generate_report()
    if interactions.empty:
        print("No interactions available.")
        return

    total_interactions = len(interactions)
    avg_response_time = interactions['timestamp'].diff().mean()

    print(f"Total Interactions: {total_interactions}")
    print(f"Average Response Time: {avg_response_time} seconds")

print_performance_metrics()
api_request_count = 0
api_request_success = 0
api_request_failure = 0

def record_api_request(success=True):
    global api_request_count, api_request_success, api_request_failure
    api_request_count += 1
    if success:
        api_request_success += 1
    else:
        api_request_failure += 1

def print_api_request_details():
    print(f"Total API Requests: {api_request_count}")
    print(f"Successful API Requests: {api_request_success}")
    print(f"Failed API Requests: {api_request_failure}")


def print_user_session_details():
    if current_user and current_user.is_authenticated:
        print(f"Current User: {current_user.username}")
        print(f"User ID: {current_user.id}")
        print(f"Authenticated: {current_user.is_authenticated}")
    else:
        print("No user is currently authenticated.")


def print_cache_status(app):
    try:
        cache_size = len(app.cache)
        logger.info(f"Cache size: {cache_size} items")
        print(f"Cache size: {cache_size} items")
    except Exception as e:
        logger.error(f"Error accessing cache: {e}")
        
def print_collection_details():
    try:
        collections = qdrant_client.get_collections().collections
        for collection in collections:
            collection_name = collection.name
            collection_info = qdrant_client.get_collection(collection_name)
            vector_size = collection_info.config.params.vector_size
            distance = collection_info.config.params.distance
            print(f"Collection: {collection_name}")
            print(f"  Vector Size: {vector_size}")
            print(f"  Distance: {distance}")
            print(f"  Number of Points: {collection_info.status.point_count}")
    except Exception as e:
        logger.error(f"Error retrieving collection details: {e}")



if __name__ == "__main__":
    # Start directory monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor_directory, args=(PDF_DIRECTORY, OUTPUT_CSV_AR, OUTPUT_CSV_FR), daemon=True)
    monitor_thread.start()
    def run_flask_app():
        process_pdfs(config.pdf_directory)
        
        check_csv_content()
        logger.info("Processing PDFs and checking collections...")
        process_pdfs(config.pdf_directory)
        logger.info("Checking Arabic collection...")
        if not collection_exists("agriculture_ar"):
            try:
                vectorize_and_store(config.output_csv_ar, "agriculture_ar")
            except Exception as e:
                logger.error(f"Error creating 'agriculture_ar' collection: {e}")
        logger.info("Checking French collection...")
        if not collection_exists("agriculture_fr"):
            try:
                vectorize_and_store(config.output_csv_fr, "agriculture_fr")
            except Exception as e:
                logger.error(f"Error creating 'agriculture_fr' collection: {e}")
        logger.info("Starting Flask app...")      
        flask_app.run(host="localhost", port=5005, debug=False)
        # Call the function to print the system usage details

    Thread(target=run_flask_app).start()


    print_system_usage()
    print_collection_details()
    print_user_session_details()
    print_performance_metrics()
    print_recent_logs()
    app = Application()
    print_cache_status(app)
    app.mainloop()