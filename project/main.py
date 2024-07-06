from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routes.auth import auth_router
from routes.pdf_processing import pdf_router
from routes.qdrant_ops import qdrant_router
from routes.speech import speech_router
from routes.text_ops import text_ops_router
import os
from config import Config

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(pdf_router, prefix="/pdf", tags=["pdf"])
app.include_router(qdrant_router, prefix="/qdrant", tags=["qdrant"])
app.include_router(speech_router, prefix="/speech", tags=["speech"])
app.include_router(text_ops_router, prefix="/text", tags=["text"])

current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

