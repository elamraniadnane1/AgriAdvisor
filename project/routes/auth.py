from fastapi import APIRouter, Depends, HTTPException, Form
from fastapi.security import OAuth2PasswordRequestForm
from models import User
from utils import hash_password, create_access_token, verify_password
from services.pdf_service import process_pdfs
from services.qdrant_service import vectorize_and_store
from config import Config
import os

auth_router = APIRouter()

users_db = {}  # Simulated user database

@auth_router.post("/register")
async def register(username: str = Form(...), password: str = Form(...)):
    if username in users_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = hash_password(password)
    user = User(username=username, password=hashed_password)
    users_db[username] = user.dict()  # Store the user as a dictionary
    return {"message": "User registered successfully"}

@auth_router.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(form_data.username)
    if not user or not verify_password(form_data.password, user['password']):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    # Process and vectorize all PDFs after a successful login
    pdf_directory = Config.PDF_DIRECTORY
    all_pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

    process_pdfs(all_pdf_files)
    vectorize_and_store(Config.OUTPUT_CSV_AR, "agriculture_ar")
    vectorize_and_store(Config.OUTPUT_CSV_FR, "agriculture_fr")

    access_token = create_access_token(data={"sub": user['username']})
    return {"access_token": access_token, "token_type": "bearer"}
