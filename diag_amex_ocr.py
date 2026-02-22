import pytesseract
from pdf2image import convert_from_bytes
from pathlib import Path

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"
BASE_DIR = r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App"

p_amex = Path(BASE_DIR) / "bank statments" / "AMEX.pdf"
images = convert_from_bytes(p_amex.read_bytes(), poppler_path=POPPLER_PATH)
# Just look at the first page for diagnostic
text = pytesseract.image_to_string(images[0])
print("--- AMEX PAGE 1 RAW OCR ---")
print(text)

# Also check page with transactions (page 3 based on screenshots)
if len(images) > 2:
    text3 = pytesseract.image_to_string(images[2])
    print("\n--- AMEX PAGE 3 RAW OCR ---")
    print(text3)
