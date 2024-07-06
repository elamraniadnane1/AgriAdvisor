import os
import fitz
import pandas as pd
from langdetect import detect
from config import Config

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def create_csv(data_list, output_csv):
    records = [{"filename": data["filename"], "content": data["content"]} for data in data_list]
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

def process_pdfs(file_paths):
    data_list_ar = []
    data_list_fr = []

    for pdf_path in file_paths:
        text = extract_text_from_pdf(pdf_path)
        try:
            language = detect(text)
        except:
            language = "unknown"
        
        data = {"filename": os.path.basename(pdf_path), "content": text}
        
        if language == "ar":
            data_list_ar.append(data)
        elif language == "fr":
            data_list_fr.append(data)
    
    if data_list_ar:
        create_csv(data_list_ar, Config.OUTPUT_CSV_AR)
    
    if data_list_fr:
        create_csv(data_list_fr, Config.OUTPUT_CSV_FR)
