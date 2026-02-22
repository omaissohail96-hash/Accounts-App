import re
import pytesseract
from pdf2image import convert_from_bytes
from pathlib import Path

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"
BASE_DIR = r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App"

DATE_AT_START = re.compile(
    r'^[^\w\s]{0,4}\s*('
    r'(?:0?[1-9]|1[0-2])[-/ ](?:0?[1-9]|[12][0-9]|3[01])(?:[-/ ]\d{2,4})?|'
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-/ ](?:0?[1-9]|[12][0-9]|3[01])(?:,?\s*[-/ ]\d{2,4})?'
    r')\b', re.I
)

p_amex = Path(BASE_DIR) / "bank statments" / "AMEX.pdf"
images = convert_from_bytes(p_amex.read_bytes(), poppler_path=POPPLER_PATH)

matches = []
for i, img in enumerate(images):
    text = pytesseract.image_to_string(img)
    lines = text.split('\n')
    for ln in lines:
        ln = ln.strip()
        if not ln: continue
        if DATE_AT_START.match(ln):
            matches.append((i+1, ln))

print(f"Total Matches Found: {len(matches)}")
for pg, m in matches:
    print(f"Page {pg}: {m}")
