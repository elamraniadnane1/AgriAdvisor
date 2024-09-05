import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

class Config:
    # Directories and output file paths
    pdf_directory = os.getenv('PDF_DIRECTORY', r"C:\Users\Dino\OneDrive\Bureau\Dataset")
    output_csv_ar = os.getenv('OUTPUT_CSV_AR', r"C:\Users\Dino\OneDrive\Bureau\Dataset\agriculture_data_ar_.csv")
    output_csv_fr = os.getenv('OUTPUT_CSV_FR', r"C:\Users\Dino\OneDrive\Bureau\Dataset\agriculture_data_fr_.csv")
    
    # OpenAI API key
    openai_api_key = os.getenv('OPENAI_API_KEY', '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')
    
    # Qdrant client settings
    qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
    qdrant_port = int(os.getenv('QDRANT_PORT', 6333))
    
    # Flask app settings
    flask_secret_key = os.getenv('FLASK_SECRET_KEY', 'supersecretkey')
    
    # Users file
    users_file = os.getenv('USERS_FILE', 'users.json')
