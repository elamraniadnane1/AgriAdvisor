import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
import os

# Path to the input rasterized PDF
input_pdf_path = r"C:\Users\LENOVO\OneDrive\Bureau\Dataset\GuideConseilAgricoleAuMarocMohamedElAmrani_2.pdf"
output_pdf_path = r"C:\Users\LENOVO\OneDrive\Bureau\Dataset\GuideConseilAgricoleAuMarocMohamedElAmrani_2_OCR.pdf"

# Ensure tesseract is correctly installed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Change this path if Tesseract is installed elsewhere

# Path to poppler bin
poppler_path = r"C:\poppler-24.07.0\Library\bin"  # Change this to your actual poppler bin path

# Convert each page of the PDF to an image
images = convert_from_path(input_pdf_path, poppler_path=poppler_path)

# Create a new PDF with the OCR text
pdf_document = fitz.open(input_pdf_path)  # Open the original PDF

# Temporary directory to save images
temp_image_dir = "temp_images"
os.makedirs(temp_image_dir, exist_ok=True)

for i, image in enumerate(images):
    # Save image to a temporary file
    image_path = os.path.join(temp_image_dir, f"page_{i+1}.png")
    image.save(image_path, "PNG")

    # Perform OCR on the image
    text = pytesseract.image_to_string(image_path, lang='eng')

    # Get page dimensions
    page = pdf_document[i]
    page_width, page_height = page.rect.width, page.rect.height

    # Add the recognized text to the PDF page
    page.insert_text((0, 0), text, fontsize=12, overlay=True)

# Save the output PDF
pdf_document.save(output_pdf_path)

# Cleanup temporary images
for image_file in os.listdir(temp_image_dir):
    os.remove(os.path.join(temp_image_dir, image_file))
os.rmdir(temp_image_dir)

print(f"OCR complete. The searchable PDF is saved as {output_pdf_path}")
