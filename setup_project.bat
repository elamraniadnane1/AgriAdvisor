@echo off
mkdir project
cd project

:: Create main files
echo. > main.py
echo. > config.py
echo. > models.py
echo. > utils.py

:: Create routes directory and files
mkdir routes
cd routes
echo. > __init__.py
echo. > auth.py
echo. > pdf_processing.py
echo. > qdrant_ops.py
echo. > speech.py
echo. > text_ops.py
cd ..

:: Create services directory and files
mkdir services
cd services
echo. > __init__.py
echo. > embedding_service.py
echo. > pdf_service.py
echo. > qdrant_service.py
echo. > speech_service.py
cd ..

echo Directory structure created successfully
