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

# Directory containing the PDF files
pdf_directory = r"C:\Users\LENOVO\OneDrive\Bureau\Dataset"

# Output CSV files
output_csv_ar = r"C:\Users\LENOVO\OneDrive\Bureau\Dataset\agriculture_data_ar_.csv"
output_csv_fr = r"C:\Users\LENOVO\OneDrive\Bureau\Dataset\agriculture_data_fr_.csv"

# OpenAI API key
openai.api_key = 'sk-proj-IoxzbELHRwIIhrlZVwrtT3BlbkFJvyxGl7jRv3fEzURZJt6g'

# Qdrant client configuration
qdrant_client = QdrantClient("localhost", port=6333)

app = Flask(__name__)
app.secret_key = 'supersecretkey'
login_manager = LoginManager()
login_manager.init_app(app)

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

@app.route('/register', methods=['POST'])
def register():
    username = request.json['username']
    password = hash_password(request.json['password'])
    user_id = str(uuid.uuid4())
    users[user_id] = User(user_id, username, password)
    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/login', methods=['POST'])
def login():
    username = request.json['username']
    password = hash_password(request.json['password'])
    for user in users.values():
        if user.username == username and user.password == password:
            login_user(user)
            return jsonify({'message': 'Login successful'}), 200
    return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/query', methods=['POST'])
@login_required
def query():
    question = request.json['question']
    collection_name = request.json['collection']
    response_text = generate_response(question, collection_name)
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

def text_to_speech(text, language="ar"):
    tts = gTTS(text=text, lang=language)
    tts.save("output.mp3")
    os.system("mpg321 output.mp3")

def query_qdrant(question, collection_name):
    """Query Qdrant to find the closest matching chunks for the given question."""
    question_embedding = get_embedding(question)
    
    search_result = qdrant_client.search(
        collection_name= collection_name,
        query_vector=question_embedding,
        limit=3
    )
    
    return [hit.payload["content"] for hit in search_result]

def generate_response(question, collection_name):
    relevant_chunks = query_qdrant(question, collection_name)
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant specialized in agricultural advice."},
            {"role": "user", "content": f"Here are some relevant information chunks: {relevant_chunks}. Now answer the following question: {question}"}
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

def user_input_choice():
    with open("output.txt", "w", encoding="utf-8") as output_file:
        choice = input("Do you want to use voice or text input? (voice/text): ").strip().lower()
        if choice == "voice":
            input_language = input("Enter the input language (ar, fr, dar): ").strip().lower()
            output_language = input("Enter the output language (ar, fr, dar): ").strip().lower()
            print("Please speak into the microphone...")
            speech_text = recognize_speech_from_microphone(language=input_language)
            print(f"Recognized Speech: {speech_text}")
            response_text = generate_response(speech_text, "agriculture_ar" if input_language == "ar" else "agriculture_fr")
            if output_language == "dar":
                response_text = translate_to_darija(response_text)
            elif input_language != output_language:
                response_text = translate_text(response_text, output_language)
            print(f"Response: {response_text}")
            output_file.write(f"Response: {response_text}\n")
            text_to_speech(response_text, language=output_language)
        elif choice == "text":
            input_language = input("Enter the input language (ar, fr, dar): ").strip().lower()
            output_language = input("Enter the output language (ar, fr, dar): ").strip().lower()
            text_input = input("Enter your text: ").strip()
            print(f"Received Text: {text_input}")
            response_text = generate_response(text_input, "agriculture_ar" if input_language == "ar" else "agriculture_fr")
            if output_language == "dar":
                response_text = translate_to_darija(response_text)
            elif input_language != output_language:
                response_text = translate_text(response_text, output_language)
            print(f"Response: {response_text}")
            output_file.write(f"Response: {response_text}\n")
            text_to_speech(response_text, language=output_language)
        else:
            print("Invalid choice. Please enter 'voice' or 'text'.")
            output_file.write("Invalid choice. Please enter 'voice' or 'text'.\n")

def format_rtl_text(text):
    """Format text for right-to-left languages."""
    return f"\u202B{text}\u202C"

if __name__ == "__main__":
    process_pdfs(pdf_directory)
    
    vectorize_and_store(output_csv_ar, "agriculture_ar")
    vectorize_and_store(output_csv_fr, "agriculture_fr")
    
    user_input_choice()

@app.route('/process_pdfs', methods=['POST'])
@login_required
def api_process_pdfs():
    process_pdfs(pdf_directory)
    return jsonify({'message': 'PDF processing and data extraction completed successfully.'}), 200

@app.route('/vectorize_and_store', methods=['POST'])
@login_required
def api_vectorize_and_store():
    data = request.json
    collection_name = data['collection_name']
    csv_path = data['csv_path']
    vectorize_and_store(csv_path, collection_name)
    return jsonify({'message': f'Vectorization and storage for {collection_name} completed successfully.'}), 200

@app.route('/text_to_speech', methods=['POST'])
@login_required
def api_text_to_speech():
    data = request.json
    text = data['text']
    language = data['language']
    text_to_speech(text, language)
    return jsonify({'message': 'Text to speech conversion completed successfully.'}), 200

@app.route('/generate_response', methods=['POST'])
@login_required
def api_generate_response():
    data = request.json
    question = data['question']
    collection_name = data['collection_name']
    response_text = generate_response(question, collection_name)
    return jsonify({'response': response_text}), 200

if __name__ == "__main__":
    app.run(debug=True)
