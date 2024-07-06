import fitz  # PyMuPDF
import pandas as pd
import os
from langdetect import detect

# Directory containing the PDF files
pdf_directory = r"C:\Users\LENOVO\OneDrive\Bureau\Dataset"

# Output CSV files
output_csv_ar = r"C:\Users\LENOVO\OneDrive\Bureau\Dataset\agriculture_data_ar_.csv"
output_csv_fr = r"C:\Users\LENOVO\OneDrive\Bureau\Dataset\agriculture_data_fr_.csv"

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def create_csv(data_list, output_csv):
    """Create a CSV file from the extracted data."""
    # Normalize data into a flat structure
    records = []
    for data in data_list:
        base = {"filename": data["filename"]}
        record = base.copy()
        record["content"] = data["content"]
        records.append(record)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

def process_pdfs(pdf_directory):
    """Process all PDFs in the directory and extract data."""
    data_list_ar = []
    data_list_fr = []

    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            text = extract_text_from_pdf(pdf_path)
            try:
                language = detect(text)
            except:
                language = "unknown"
            
            data = {"filename": filename, "content": text}
            
            if language == "ar":
                data_list_ar.append(data)
            elif language == "fr":
                data_list_fr.append(data)
    
    # Save Arabic documents to CSV
    if data_list_ar:
        create_csv(data_list_ar, output_csv_ar)
    
    # Save French documents to CSV
    if data_list_fr:
        create_csv(data_list_fr, output_csv_fr)

if __name__ == "__main__":
    process_pdfs(pdf_directory)
