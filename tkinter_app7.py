import fitz  # PyMuPDF
import pandas as pd
import os
import uuid
from langdetect import detect
import openai
from qdrant_client import QdrantClient
from qdrant_client.http import models
import speech_recognition as sr
from gtts import gTTS
import numpy as np
from flask import Flask, request, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import hashlib
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from threading import Thread
import re
import pygame
import tempfile
import json
import time
import mlflow
import mlflow.pyfunc

# Directory containing the PDF files
pdf_directory = r"C:\Users\LENOVO\OneDrive\Bureau\Dataset"

# Output CSV files
output_csv_ar = r"C:\Users\LENOVO\OneDrive\Bureau\Dataset\agriculture_data_ar_.csv"
output_csv_fr = r"C:\Users\LENOVO\OneDrive\Bureau\Dataset\agriculture_data_fr_.csv"

# OpenAI API key
openai.api_key = 'sk-proj-IoxzbELHRwIIhrlZVwrtT3BlbkFJvyxGl7jRv3fEzURZJt6g'

# Qdrant client configuration
qdrant_client = QdrantClient("localhost", port=6333)

flask_app = Flask(__name__)
flask_app.secret_key = 'supersecretkey'
login_manager = LoginManager()
login_manager.init_app(flask_app)

users = {}

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

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
    records = []
    for data in data_list:
        base = {"filename": data["filename"]}
        record = base.copy()
        record["content"] = data["content"]
        records.append(record)
    
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

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
        create_csv(data_list_ar, output_csv_ar)
    
    if data_list_fr:
        create_csv(data_list_fr, output_csv_fr)

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
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=content
    )
    return response['data'][0]['embedding']

def vectorize_and_store(csv_path, collection_name):
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

def recognize_speech_from_microphone(language="ar"):
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

def clean_text_for_speech(text):
    """Clean text for speech by removing unnecessary characters."""
    text = re.sub(r'\*\*\*|\.{2,}', '', text)
    return text.strip()

def text_to_speech(text, language="ar"):
    clean_text = clean_text_for_speech(text)
    tts = gTTS(text=clean_text, lang=language)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        temp_filename = fp.name
        tts.save(temp_filename)
    pygame.mixer.init()
    pygame.mixer.music.load(temp_filename)
    pygame.mixer.music.play()

def play_audio():
    pygame.mixer.music.unpause()

def pause_audio():
    pygame.mixer.music.pause()

def replay_audio():
    pygame.mixer.music.stop()
    pygame.mixer.music.play()

def query_qdrant(question, collection_name):
    """Query Qdrant to find the closest matching chunks for the given question."""
    question_embedding = get_embedding(question)
    
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=question_embedding,
        limit=3
    )
    
    return [hit.payload["content"] for hit in search_result]

def generate_response(question, collection_name):
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

def translate_text(text, target_language):
    """Translate text to the target language using GPT-4O."""
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Translate the following text to {target_language}."},
            {"role": "user", "content": text}
        ]
    )
    
    return response['choices'][0]['message']['content']

def translate_to_darija(text):
    """Translate text to Moroccan Darija using GPT-4O."""
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Translate the following text to Moroccan Darija using Arabic letters."},
            {"role": "user", "content": text}
        ]
    )
    
    return response['choices'][0]['message']['content']

def user_input_choice(input_lang, output_lang, user_input, input_type="text"):
    if input_type == "voice":
        print("Please speak into the microphone...")
        speech_text = recognize_speech_from_microphone(language=input_lang)
        print(f"Recognized Speech: {speech_text}")
        response_text = generate_response(speech_text, "agriculture_ar" if input_lang == "ar" else "agriculture_fr")
        if output_lang == "dar":
            response_text = translate_to_darija(response_text)
        elif input_lang != output_lang:
            response_text = translate_text(response_text, output_lang)
        print(f"Response: {response_text}")
        return response_text
    elif input_type == "text":
        print(f"Received Text: {user_input}")
        response_text = generate_response(user_input, "agriculture_ar" if input_lang == "ar" else "agriculture_fr")
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

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Agricultural Advisor")
        self.geometry("800x600")

        self.username = None

        self.show_login_window()

    def show_login_window(self):
        self.login_window = tk.Toplevel(self)
        self.login_window.title("Login")

        tk.Label(self.login_window, text="Username").grid(row=0, column=0, padx=10, pady=10)
        tk.Label(self.login_window, text="Password").grid(row=1, column=0, padx=10, pady=10)

        self.username_entry = tk.Entry(self.login_window)
        self.password_entry = tk.Entry(self.login_window, show='*')

        self.username_entry.grid(row=0, column=1, padx=10, pady=10)
        self.password_entry.grid(row=1, column=1, padx=10, pady=10)

        tk.Button(self.login_window, text="Login", command=self.login).grid(row=2, column=0, columnspan=2, pady=10)
        tk.Button(self.login_window, text="Register", command=self.register).grid(row=3, column=0, columnspan=2, pady=10)

    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        if self.authenticate(username, password):
            self.username = username
            self.login_window.destroy()
            self.show_main_window()
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
            messagebox.showinfo("Success", "User registered successfully")

    def show_main_window(self):
        # Set up the style
        style = ttk.Style(self)
        style.configure("TLabel", font=("Helvetica", 12))
        style.configure("TButton", font=("Helvetica", 12), padding=10)
        style.configure("TCombobox", font=("Helvetica", 12))
        style.configure("TScrolledText", font=("Helvetica", 12))

        # Configure grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)

        # Input Text Widget
        self.input_label = ttk.Label(self, text="Input Text:")
        self.input_label.grid(row=0, column=0, pady=5, sticky='nsew')
        self.input_text = scrolledtext.ScrolledText(self, wrap=tk.WORD, width=40, height=20)
        self.input_text.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')

        # Language Selection
        self.lang_label = ttk.Label(self, text="Select Input Language:")
        self.lang_label.grid(row=2, column=0, pady=5, sticky='nsew')
        self.input_lang = ttk.Combobox(self, values=["ar", "fr", "dar"])
        self.input_lang.grid(row=3, column=0, pady=5, sticky='nsew')
        
        self.output_lang_label = ttk.Label(self, text="Select Output Language:")
        self.output_lang_label.grid(row=4, column=0, pady=5, sticky='nsew')
        self.output_lang = ttk.Combobox(self, values=["ar", "fr", "dar"])
        self.output_lang.grid(row=5, column=0, pady=5, sticky='nsew')

        # Output Text Widget
        self.output_label = ttk.Label(self, text="Output Text:")
        self.output_label.grid(row=0, column=2, pady=5, sticky='nsew')
        self.output_text = scrolledtext.ScrolledText(self, wrap=tk.WORD, width=40, height=20)
        self.output_text.grid(row=1, column=2, padx=5, pady=5, rowspan=5, sticky='nsew')

        # Buttons
        self.submit_button = ttk.Button(self, text="Submit", command=self.process_input)
        self.submit_button.grid(row=1, column=1, pady=5)

        self.voice_button = ttk.Button(self, text="Voice Input", command=self.process_voice_input)
        self.voice_button.grid(row=2, column=1, pady=5)

        self.speak_button = ttk.Button(self, text="Read Aloud Output", command=self.read_aloud_output)
        self.speak_button.grid(row=3, column=1, pady=5)

        self.play_button = ttk.Button(self, text="▶ Play", command=play_audio)
        self.play_button.grid(row=4, column=1, pady=5)

        self.pause_button = ttk.Button(self, text="⏸ Pause", command=pause_audio)
        self.pause_button.grid(row=5, column=1, pady=5)

        self.replay_button = ttk.Button(self, text="⏪ Replay", command=replay_audio)
        self.replay_button.grid(row=6, column=1, pady=5)

        # Add a button to generate and display the report
        self.report_button = ttk.Button(self, text="Generate Report", command=self.display_report)
        self.report_button.grid(row=7, column=1, pady=5)

        # Display the logged-in user info and logout button
        self.user_info_label = ttk.Label(self, text=f"Logged in as: {self.username}")
        self.user_info_label.grid(row=8, column=0, pady=5, sticky='w')

        self.logout_button = ttk.Button(self, text="Logout", command=self.logout)
        self.logout_button.grid(row=8, column=2, pady=5, sticky='e')

        # Label to display recording status
        self.recording_label = ttk.Label(self, text="Recording... Please speak into the microphone", foreground="red")
        self.recording_label.grid(row=9, column=0, columnspan=3, pady=5)
        self.recording_label.grid_remove()  # Hide the label initially

    def logout(self):
        self.username = None
        for widget in self.winfo_children():
            widget.destroy()
        self.show_login_window()

    def process_input(self):
        user_input = self.input_text.get("1.0", tk.END).strip()
        input_lang = self.input_lang.get()
        output_lang = self.output_lang.get()
        if not user_input or not input_lang or not output_lang:
            messagebox.showerror("Error", "All fields must be filled")
            return
        
        # Run the processing in a separate thread
        Thread(target=self.run_user_input_choice, args=(input_lang, output_lang, user_input, "text")).start()
    
    def process_voice_input(self):
        input_lang = self.input_lang.get()
        output_lang = self.output_lang.get()
        if not input_lang or not output_lang:
            messagebox.showerror("Error", "All fields must be filled")
            return

        # Display recording status
        self.recording_label.grid()
        
        # Run the processing in a separate thread
        Thread(target=self.run_user_input_choice, args=(input_lang, output_lang, "", "voice")).start()

    def run_user_input_choice(self, input_lang, output_lang, user_input, input_type):
        response_text = user_input_choice(input_lang, output_lang, user_input, input_type)
        if output_lang == "ar":
            response_text = format_rtl_text(response_text)
        self.after(0, self.update_output_text, response_text)
    
    def update_output_text(self, response_text):
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, response_text)
        self.recording_label.grid_remove()  # Hide the recording label after processing

    def read_aloud_output(self):
        output_lang = self.output_lang.get()
        response_text = self.output_text.get("1.0", tk.END).strip()
        if response_text:
            Thread(target=text_to_speech, args=(response_text, output_lang)).start()

    def display_report(self):
        report = generate_report()
        report_window = tk.Toplevel(self)
        report_window.title("Interaction Report")
        report_text = scrolledtext.ScrolledText(report_window, wrap=tk.WORD, width=100, height=20)
        report_text.pack(padx=10, pady=10)
        report_text.insert(tk.END, report.to_string())

if __name__ == "__main__":
    # Run the Flask app in a separate thread
    def run_flask_app():
        process_pdfs(pdf_directory)
        vectorize_and_store(output_csv_ar, "agriculture_ar")
        vectorize_and_store(output_csv_fr, "agriculture_fr")
        flask_app.run(debug=False)

    Thread(target=run_flask_app).start()

    # Run the Tkinter app
    app = Application()
    app.mainloop()
