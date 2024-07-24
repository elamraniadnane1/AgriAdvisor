import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

class Config:
    # Directories and output file paths
    pdf_directory = os.getenv('PDF_DIRECTORY', r"C:\Users\LENOVO\OneDrive\Bureau\Dataset")
    output_csv_ar = os.getenv('OUTPUT_CSV_AR', r"C:\Users\LENOVO\OneDrive\Bureau\Dataset\agriculture_data_ar_.csv")
    output_csv_fr = os.getenv('OUTPUT_CSV_FR', r"C:\Users\LENOVO\OneDrive\Bureau\Dataset\agriculture_data_fr_.csv")
    
    # OpenAI API key
    openai_api_key = os.getenv('OPENAI_API_KEY', 'sk-proj-IoxzbELHRwIIhrlZVwrtT3BlbkFJvyxGl7jRv3fEzURZJt6g')
    
    # Qdrant client settings
    qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
    qdrant_port = int(os.getenv('QDRANT_PORT', 6333))
    
    # Flask app settings
    flask_secret_key = os.getenv('FLASK_SECRET_KEY', 'supersecretkey')
    
    # Users file
    users_file = os.getenv('USERS_FILE', 'users.json')
