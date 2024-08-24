# AgriAdvisor Project

AgriAdvisor is an AI-powered application designed to assist with agricultural advice, utilizing advanced technologies such as OpenAI's language models, Qdrant vector databases, and a Flask-based web application.

## Table of Contents

- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started with the AgriAdvisor project, follow these steps:

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/AgriAdvisor.git
    cd AgriAdvisor
    ```

2. **Install Dependencies:**

    Ensure you have Python 3.8+ installed. Then, install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up Environment Variables:**

    Create a `.env` file in the root directory with the following environment variables:

    ```env
    OPENAI_API_KEY=your_openai_api_key
    QDRANT_HOST=localhost
    QDRANT_PORT=6333
    FLASK_SECRET_KEY=your_flask_secret_key
    ```

4. **Run the Application:**

    ```bash
    python main.py
    ```

## Features

- **PDF Monitoring and Processing:** 
  - Automatically monitors a specified directory for new PDF files, extracts text, and processes the content based on the detected language.
  
- **AI-Powered Agricultural Advice:**
  - Provides detailed and accurate responses to agricultural-related queries using OpenAI's GPT models.

- **Multi-language Support:**
  - Supports Arabic, French, and Moroccan Darija languages for both input and output.

- **Speech Recognition and Text-to-Speech:**
  - Allows users to interact with the application using voice commands and listen to responses using text-to-speech.

- **Interactive User Interface:**
  - A modern GUI built using `customtkinter` and `ttkbootstrap` with features like login/register, report generation, and feedback submission.

- **Data Vectorization and Storage:**
  - Stores extracted data from PDFs in Qdrant vector databases for efficient retrieval and search.

- **User Management:**
  - Supports user registration, login, and authentication using Flask-Login.

- **Performance Monitoring:**
  - Provides metrics on response time, user activity, and API request statistics.

## Usage

### Running the Flask Application

To start the Flask application and the PDF monitoring service:

```bash
python main.py
