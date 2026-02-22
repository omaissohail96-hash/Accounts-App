import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
from pathlib import Path
import sys
import re

# Configure tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"

def extract_text(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for i in range(min(5, len(pdf.pages))): # Scan first 5 pages
                p_text = pdf.pages[i].extract_text()
                if p_text:
                    text += p_text + "\n"
            if text.strip():
                return text
    except:
        pass
    
    # OCR Fallback
    try:
        images = convert_from_bytes(pdf_path, first_page=1, last_page=5, poppler_path=POPPLER_PATH)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"
        return text
    except Exception as e:
        return f"OCR Failed: {e}"

def analyze():
    sys.stdout.reconfigure(encoding='utf-8')
    bank_dir = Path("bank statments")
    for pdf_file in bank_dir.glob("*.pdf"):
        print(f"=== {pdf_file.name} ===")
        text = extract_text(pdf_file)
        
        # Look for summary keywords and surrounding lines
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if re.search(r'(Account Summary|Beginning Balance|Ending Balance|Opening Balance|Closing Balance|Total Deposits|Total Withdrawals|Previous Balance|New Balance)', line, re.I):
                # Print current line and 5 lines after
                print(f"MATCH at line {i}: {line}")
                for j in range(1, 6):
                    if i + j < len(lines):
                        print(f"  +{j}: {lines[i+j]}")
        print("\n")

if __name__ == "__main__":
    analyze()
