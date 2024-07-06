import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'supersecretkey')
    OPENAI_API_KEY = os.environ.get('sk-proj-IoxzbELHRwIIhrlZVwrtT3BlbkFJvyxGl7jRv3fEzURZJt6g')
    QDRANT_HOST = os.environ.get('QDRANT_HOST', 'localhost')
    QDRANT_PORT = int(os.environ.get('QDRANT_PORT', 6333))
    PDF_DIRECTORY = os.environ.get('PDF_DIRECTORY', r"C:\Users\LENOVO\OneDrive\Bureau\Dataset")
    OUTPUT_CSV_AR = os.environ.get('OUTPUT_CSV_AR', r"C:\Users\LENOVO\OneDrive\Bureau\Dataset\agriculture_data_ar_.csv")
    OUTPUT_CSV_FR = os.environ.get('OUTPUT_CSV_FR', r"C:\Users\LENOVO\OneDrive\Bureau\Dataset\agriculture_data_fr_.csv")


