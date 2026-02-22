import re
import os
import pytesseract
from pdf2image import convert_from_bytes
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

# --- CONFIG ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"

# --- RE-IMPLEMENTED LOGIC FROM bank_data_analysis.py (w/ FIXES) ---

@dataclass
class Transaction:
    date: str
    transaction_type: str
    vendor: str
    amount: float
    description: str
    raw_line: str
    section: Optional[str] = None

# TIGHTENED REGEX (THE FIX)
DATE_AT_START = re.compile(
    r'^\s*('
    r'(?:0?[1-9]|1[0-2])[-/ ](?:0?[1-9]|[12][0-9]|3[01])(?:[-/ ]\d{2,4})?|'
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(?:0?[1-9]|[12][0-9]|3[01])(?:,?\s*\d{2,4})?'
    r')\b', re.I
)

AMOUNT_RE = re.compile(r'([+\-]?\(?\s*\$?\d{1,3}(?:[,\s]\d{3})*\.\d{2}\)?)')

def clean_amount(s: str) -> float:
    if not s: return 0.0
    s = s.strip().replace('$', '').replace(',', '').replace(' ', '')
    if s.endswith('-'): s = '-' + s[:-1]
    try: return float(s)
    except: return 0.0

def _normalize_date_token(token: str) -> str:
    if not token: return ""
    t = token.replace("-", "/").replace(" ", "/").strip().replace(",", "")
    formats = ["%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y", "%d/%m/%y", "%m/%d", "%d/%m"]
    for fmt in formats:
        try: return datetime.strptime(t, fmt).replace(year=2025).strftime("%Y-%m-%d")
        except: continue
    return t

def _clean_amount_token(token: str) -> Optional[float]:
    if not token: return None
    s = str(token).strip()
    negative = "(" in s or "-" in s
    s = re.sub(r'[^0-9.]', '', s)
    try: return -abs(float(s)) if negative else abs(float(s))
    except: return None

def extract_true_amount(text: str) -> Optional[float]:
    candidates = AMOUNT_RE.findall(text)
    if not candidates: return None
    return _clean_amount_token(candidates[-1])

def normalize_vendor(text: str) -> str:
    v = text.strip()
    v = re.sub(r'\d{3}-\d{3}-\d{4}', '', v)
    return re.sub(r'\s+', ' ', v).strip().title()[:30]

class FallbackStatementParser:
    SECTION_PATTERNS = [
        ("DEPOSITS", re.compile(r'\bdeposits\b|\bpayments\s*/\s*credits\b', re.I)),
        ("WITHDRAWALS", re.compile(r'\bwithdrawals\b|\bnew\s+charges\b', re.I)),
    ]

    def _is_summary_line(self, ln: str) -> bool:
        low = ln.lower()
        if re.match(r'^(daily ending|opening balance|ending balance|closing balance|page\s+\d+)', low): return True
        # SUMMARY lines to skip
        if re.match(r'^(total\s+deposits?|total\s+withdrawals?|total\s+service\s+fees?|account\s+summary|balance\s+summary)', low): return True
        return False

    def parse_statement(self, lines: List[str]) -> List[Transaction]:
        txs = []
        cleaned = [l for l in lines if not self._is_summary_line(l.strip())]
        cur_sec = "UNKNOWN"
        cur_block = None
        for ln in cleaned:
            # simple section detect
            for sec, pat in self.SECTION_PATTERNS:
                if pat.search(ln): 
                    print(f"DEBUG: Found Section {sec} at line: {ln}")
                    cur_sec = sec; break
            
            if DATE_AT_START.match(ln):
                if cur_block: self._proc(cur_block, cur_sec, txs)
                cur_block = [ln]
            elif cur_block is not None:
                cur_block.append(ln)
        if cur_block: self._proc(cur_block, cur_sec, txs)
        return txs

    def _proc(self, lines, sec, txs):
        txt = " ".join(lines)
        dmatch = DATE_AT_START.match(lines[0])
        date_raw = dmatch.group(1)
        amt = extract_true_amount(txt)
        if amt:
            v_text = re.sub(DATE_AT_START, '', lines[0]).strip()
            txs.append(Transaction(_normalize_date_token(date_raw), "deposit" if amt > 0 else "withdrawal", normalize_vendor(v_text), amt, txt, txt, sec))

def extract_universal_bank_summary(raw_text: str) -> Dict:
    result = {
        "Beginning Balance": {"amount": 0.0},
        "Deposits and Additions": {"amount": 0.0},
        "Other Withdrawals": {"amount": 0.0},
        "Ending Balance": {"amount": 0.0},
    }
    # BMO Patterns
    if re.search(r'\bBMO\b', raw_text, re.I):
        m = re.search(r'BEGINNING\s+BALANCE\s+AS\s+OF[^\n]*\$\s*([\d, ]+\.?\d{0,2})', raw_text, re.I)
        if m: result["Beginning Balance"]["amount"] = clean_amount(m.group(1))
        m = re.search(r'ENDING\s+BALANCE\s+AS\s+OF[^\n]*\$\s*([\d, ]+\.?\d{0,2})', raw_text, re.I)
        if m: result["Ending Balance"]["amount"] = clean_amount(m.group(1))
        m = re.search(r'DEPOSIT\s+AMOUNT\s*[^\d]*([\d, ]+\.?\d{0,2})', raw_text, re.I)
        if m: result["Deposits and Additions"]["amount"] = clean_amount(m.group(1))
        m = re.search(r'WITHDRAWAL\s+AMOUNT\s*[^\d]*([\d, ]+\.?\d{0,2})', raw_text, re.I)
        if m: result["Other Withdrawals"]["amount"] = -abs(clean_amount(m.group(1)))
    return result

# --- RUN ---
p_bmo = Path("bank statments/BMO.pdf")
print(f"\nProcessing BMO: {p_bmo}")
images = convert_from_bytes(p_bmo.read_bytes(), poppler_path=POPPLER_PATH)
text_bmo = "\n".join(pytesseract.image_to_string(img) for img in images)

print("BMO RAW TEXT (First 2000 chars):")
print(text_bmo[:2000])

print("\nBMO Summary:")
summary = extract_universal_bank_summary(text_bmo)
for k, v in summary.items(): print(f"  {k}: {v['amount']}")

parser = FallbackStatementParser()
txs_bmo = parser.parse_statement(text_bmo.split('\n'))
print(f"BMO Transactions: {len(txs_bmo)}")
if txs_bmo:
    print("Sample BMO:")
    for t in txs_bmo[:10]: print(f"  {t.date} | {t.amount} | {t.section} | {t.vendor}")

p_amex = Path("bank statments/AMEX.pdf")
print(f"\nProcessing AMEX: {p_amex}")
images = convert_from_bytes(p_amex.read_bytes(), poppler_path=POPPLER_PATH)
text_amex = "\n".join(pytesseract.image_to_string(img) for img in images)

print("AMEX RAW TEXT (First 2000 chars):")
print(text_amex[:2000])

txs_amex = parser.parse_statement(text_amex.split('\n'))
print(f"AMEX Transactions: {len(txs_amex)}")

new_charges = [t for t in txs_amex if t.section == "WITHDRAWALS"]
payments = [t for t in txs_amex if t.section == "DEPOSITS"]
print(f"  New Charges: {len(new_charges)}, Payments: {len(payments)}")
if txs_amex:
    print("Sample AMEX:")
    for t in txs_amex[:5]: print(f"  {t.date} | {t.amount} | {t.section} | {t.vendor}")
