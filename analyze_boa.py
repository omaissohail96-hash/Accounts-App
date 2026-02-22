import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
from pathlib import Path
import sys

# Configure tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"

def analyze_boa():
    sys.stdout.reconfigure(encoding='utf-8')
    pdf_file = Path("bank statments/BoA.pdf")
    
    print(f"--- Analyzing {pdf_file.name} ---")
    try:
        with open(pdf_file, "rb") as f:
            file_bytes = f.read()
        
        lines = []
        # Try pdfplumber first
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages[:5]:
                text = page.extract_text()
                if text:
                    lines.extend(text.split("\n"))
        
        # If empty, use OCR
        if not lines:
            print("PDF text extraction failed, trying OCR...")
            images = convert_from_bytes(file_bytes, first_page=1, last_page=5, poppler_path=POPPLER_PATH)
            for img in images:
                text = pytesseract.image_to_string(img)
                if text:
                    lines.extend(text.split("\n"))
        
        for i, line in enumerate(lines[:200]):
            print(f"{i:3}: {line}")
            
    except Exception as e:
        print(f"Error processing {pdf_file.name}: {e}")

if __name__ == "__main__":
    analyze_boa()
