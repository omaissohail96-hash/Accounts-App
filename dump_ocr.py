
import os
import sys
import logging

# Add current directory to path
sys.path.insert(0, os.getcwd())

from document_parser import DocumentParser

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO)

pdf_dir = r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App\bank statments"
output_dir = r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App\ocr_dumps"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pdfs = [
    "AMEX.pdf",
    "BMO.pdf",
    "BoA.pdf",
    "FifthThird 53.pdf",
    "US Bank.pdf"
]

parser = DocumentParser()

for pdf_name in pdfs:
    pdf_path = os.path.join(pdf_dir, pdf_name)
    output_path = os.path.join(output_dir, pdf_name.replace(".pdf", ".txt"))
    
    print(f"Parsing {pdf_name} with OCR support...")
    try:
        lines, is_readable, unreadable = parser.parse_pdf(pdf_path)
        
        full_text = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_text)
            
        print(f"Extracted {len(lines)} lines. Saved to {output_path}")
        print(f"Is Readable: {is_readable}, Unreadable Pages: {unreadable}")
    except Exception as e:
        print(f"Error parsing {pdf_name}: {e}")
