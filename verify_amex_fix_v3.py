"""
Final verification script for AMEX/BMO.
"""
import sys
import re
from unittest.mock import MagicMock

# 1. Mock Streamlit correctly
mock_st = MagicMock()
# Mock tabs to return exactly 2 items for the login/signup unpack
mock_st.tabs.return_value = [MagicMock(), MagicMock()]
sys.modules["streamlit"] = mock_st

# 2. Import module
import bank_data_analysis as bda
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
from pathlib import Path

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"
sys.stdout.reconfigure(encoding='utf-8')

def get_text(p):
    try:
        with pdfplumber.open(p) as pdf:
            t = "\n".join(page.extract_text() or "" for page in pdf.pages)
            if t.strip(): return t
    except: pass
    img = convert_from_bytes(p.read_bytes(), poppler_path=POPPLER_PATH)
    return "\n".join(pytesseract.image_to_string(i) for i in img)

# 1. VERIFY AMEX
print("--- VERIFYING AMEX.pdf (BANK MODE) ---")
try:
    amex_text = get_text(Path("bank statments/AMEX.pdf"))
    parser = bda.FallbackStatementParser()
    lines = bda.clean_text_lines(amex_text)
    txs = parser.parse_statement(lines)

    print(f"Total Transactions: {len(txs)}")
    
    # Check for trash
    trash = [t for t in txs if re.search(r'\d{3}-\d{3}', t.date) or "UNKNOWN" in t.vendor]
    print(f"Potential Trash Transactions: {len(trash)}")
    if trash:
        print("Sample Trash:")
        for t in trash[:5]:
            print(f"  {t.date} | {t.vendor} | {t.amount} | {t.raw_line[:50]}")

    # Check for valid vendors
    valid = [t for t in txs if "MATARI" in t.vendor.upper()]
    print(f"Valid MATARI Transactions: {len(valid)}")
    for t in valid:
        print(f"  {t.date} | {t.vendor} | {t.amount}")

except Exception as e:
    import traceback
    traceback.print_exc()

# 2. VERIFY BMO
print("\n--- VERIFYING BMO.pdf ---")
try:
    bmo_text = get_text(Path("bank statments/BMO.pdf"))
    lines_bmo = bda.clean_text_lines(bmo_text)
    txs_bmo = parser.parse_statement(lines_bmo)
    cats_bmo = {}
    for tx in txs_bmo:
        cat = getattr(tx, 'category', 'UNKNOWN')
        cats_bmo[cat] = cats_bmo.get(cat, 0) + 1
    for c, count in cats_bmo.items():
        print(f"BMO Category '{c}': {count}")
except Exception as e:
    print(f"BMO Error: {e}")
