import os
import requests
import qdrant_client
from qdrant_client.http import models
import openai
import pytesseract
from PIL import Image
import librosa
import soundfile as sf
import moviepy.editor as mp
import numpy as np
from google.cloud import speech_v1 as speech
from google.oauth2 import service_account
from dotenv import load_dotenv

#pip install requests qdrant-client openai pytesseract pillow librosa soundfile moviepy google-cloud-speech python-dotenv


# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
GOOGLE_CLOUD_CREDENTIALS = os.getenv("GOOGLE_CLOUD_CREDENTIALS")

# Initialize Qdrant client
client = qdrant_client.QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# OpenAI setup
openai.api_key = OPENAI_API_KEY

# Function to convert images to text using OCR
def image_to_text(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

# Function to convert video to text by extracting audio and using speech-to-text
def video_to_text(video_path):
    video = mp.VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path)
    return audio_to_text(audio_path)

# Function to convert audio to text using Google Cloud Speech-to-Text
def audio_to_text(audio_path):
    credentials = service_account.Credentials.from_service_account_file(GOOGLE_CLOUD_CREDENTIALS)
    client = speech.SpeechClient(credentials=credentials)
    with open(audio_path, "rb") as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US"
    )
    response = client.recognize(config=config, audio=audio)
    text = " ".join([result.alternatives[0].transcript for result in response.results])
    return text

# Function to generate text embeddings using GPT-4
def generate_text_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"  # Use the appropriate OpenAI embedding model
    )
    return response['data'][0]['embedding']

# Function to preprocess and normalize text
def preprocess_text(text):
    # Implement your text preprocessing here (e.g., lowercasing, removing punctuation)
    normalized_text = text.lower().strip()
    return normalized_text

# Function to store vector embeddings in Qdrant
def store_in_qdrant(embedding, text):
    vector_id = hash(text)  # Generate a unique ID for the text
    client.upsert(
        collection_name="agriadvisor",
        points=[
            models.PointStruct(
                id=vector_id,
                vector=embedding,
                payload={"text": text}
            )
        ]
    )

# Main function to process data and store in Qdrant
def process_and_store_data(data_type, file_path):
    if data_type == "text":
        with open(file_path, "r") as file:
            text = file.read()
    elif data_type == "image":
        text = image_to_text(file_path)
    elif data_type == "video":
        text = video_to_text(file_path)
    elif data_type == "audio":
        text = audio_to_text(file_path)
    else:
        raise ValueError("Unsupported data type")

    normalized_text = preprocess_text(text)
    embedding = generate_text_embedding(normalized_text)
    store_in_qdrant(embedding, normalized_text)
    print(f"Processed and stored: {file_path}")

# Example usage
if __name__ == "__main__":
    # Sample file paths
    text_file_path = "sample_text.txt"
    image_file_path = "sample_image.jpg"
    video_file_path = "sample_video.mp4"
    audio_file_path = "sample_audio.wav"

    # Process and store each file type
    process_and_store_data("text", text_file_path)
    process_and_store_data("image", image_file_path)
    process_and_store_data("video", video_file_path)
    process_and_store_data("audio", audio_file_path)
