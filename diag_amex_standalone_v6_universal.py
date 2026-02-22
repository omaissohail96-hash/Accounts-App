import re
import pytesseract
from pdf2image import convert_from_bytes
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

# --- CONFIG ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"
BASE_DIR = r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App"

@dataclass
class Transaction:
    date: str
    transaction_type: str
    vendor: str
    amount: float
    description: str
    raw_line: str
    section: Optional[str] = None
    needs_review: bool = False

# Updated PRODUCTION Regexes
DATE_RE = re.compile(r'\b(\d{1,2}\s+[A-Za-z]{3,9},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]?\d{2,4})\b')
AMOUNT_RE = re.compile(r'([+\-]?\s?\d{1,3}(?:[,\s]\d{3})*\.\d{2})')

def _normalize_date_token(d): return d
def _clean_amount_token(s):
    s = s.strip().replace('$', '').replace(',', '').replace('(', '').replace(')', '').replace(' ', '')
    try: return float(s)
    except: return None
def _short_vendor(v): return v[:30]
def _clean_description(d, v, t): return d

class UniversalParser:
    def __init__(self):
        self.DATE_RE = DATE_RE
        self.AMOUNT_RE = AMOUNT_RE
        self.IGNORE_WORDS = ["balance"]

    def parse(self, lines: List[str]):
        # Simulate is_tabular gating logic for AMEX Page 1
        text_joined = "\n".join(lines[:200]).lower()
        is_tabular = ("debit" in text_joined or "credit" in text_joined) and ("balance" in text_joined)
        if "member summary" in text_joined:
            is_tabular = False # Gated
            
        print(f"DEBUG: UniversalParser(AMEX) is_tabular detected as {is_tabular}")
        return self._parse_simple(lines)

    def _parse_simple(self, lines: List[str]):
        txs = []
        for ln in lines:
            d = self.DATE_RE.search(ln)
            if not d: continue
            
            m = self.AMOUNT_RE.findall(ln)
            if not m: continue
            
            amt_raw = m[-2] if len(m) >= 2 else m[-1]
            amt = _clean_amount_token(amt_raw)
            if amt is None: continue
            
            date_raw = d.group(1)
            desc = ln.replace(date_raw, "").replace(amt_raw, "").strip()
            txs.append(Transaction(date_raw, "withdrawal", desc, amt, desc, ln))
        return txs

# --- RUN ---
p_amex = Path(BASE_DIR) / "bank statments" / "AMEX.pdf"
images = convert_from_bytes(p_amex.read_bytes(), poppler_path=POPPLER_PATH)
text_amex = "\n".join(pytesseract.image_to_string(img) for img in images)
parser = UniversalParser()
txs = parser.parse(text_amex.split('\n'))

print(f"\n--- AMEX PRODUCTION DIAGNOSTIC (ITERATION 6 - UNIVERSAL) ---")
print(f"Total Transactions: {len(txs)}")
for i, t in enumerate(txs):
    print(f"{i+1:3} | {t.date} | {t.amount:10.2f} | {t.vendor}")
    if "19.00" == f"{t.amount:.2f}" and "11/19/25" in t.date:
        print("FAILED: 19.00 date-leak found!")
