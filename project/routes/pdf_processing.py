from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.security import OAuth2PasswordBearer
from typing import List
from services.pdf_service import process_pdfs
from services.qdrant_service import vectorize_and_store
from config import Config
import os

pdf_router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

@pdf_router.post("/process_pdfs")
async def api_process_pdfs(files: List[UploadFile] = File(...), token: str = Depends(oauth2_scheme)):
    saved_files = []
    try:
        for file in files:
            file_path = os.path.join(Config.PDF_DIRECTORY, file.filename)
            with open(file_path, "wb") as f:
                f.write(file.file.read())
            saved_files.append(file_path)

        process_pdfs(saved_files)

        vectorize_and_store(Config.OUTPUT_CSV_AR, "agriculture_ar")
        vectorize_and_store(Config.OUTPUT_CSV_FR, "agriculture_fr")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process files: {str(e)}")

    return {"message": "PDF processing and data extraction completed successfully."}
