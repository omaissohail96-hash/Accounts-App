import re
import os
import pdfplumber
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
    category: Optional[str] = None
    needs_review: bool = False
    source: Optional[str] = "BANK"

DATE_TOKEN_RE = re.compile(r'(?P<d>\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]{3,9}\s*\d{0,4})')

# TIGHTENED REGEX (THE FIX)
DATE_AT_START = re.compile(
    r'^\s*('
    r'(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12][0-9]|3[01])(?:/\d{2,4})?|'
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(?:0?[1-9]|[12][0-9]|3[01])(?:,?\s*\d{2,4})?'
    r')\b', re.I
)

AMOUNT_RE = re.compile(r'([+\-]?\(?\s*\$?\d{1,3}(?:[,\s]\d{3})*\.\d{2}\)?)')
CHECK_ROW_RE = re.compile(r'^\s*(\d{2,6})\b') 
FEE_KEYWORDS_RE = re.compile(r'(?:^|\s)(monthly\s+service\s+fee|service\s+fee|maintenance\s+fee|bank\s+fees?|account\s+fee|overdraft\s+fee)(?:\s|$)', re.I)

def _normalize_date_token(token: str) -> str:
    if not token: return ""
    t = token.strip().replace(",", "")
    formats = ["%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%m/%d", "%d/%m", "%d %b %Y", "%d %b %y", "%b %d %Y", "%B %d %Y"]
    for fmt in formats:
        try:
            dt = datetime.strptime(t, fmt)
            if "%Y" not in fmt and "%y" not in fmt:
                dt = dt.replace(year=2025) # assume 2025 for verification
            return dt.strftime("%Y-%m-%d")
        except: continue
    return t

def _clean_amount_token(token: str) -> Optional[float]:
    if not token: return None
    s = str(token).strip()
    negative = "(" in s or "-" in s
    s = re.sub(r'[A-Za-z\$£€₹]', '', s).replace(',', '').replace(' ', '')
    s = re.sub(r'[^0-9\.\-]', '', s)
    if not re.search(r'\d', s): return None
    try: return -abs(float(s)) if negative else abs(float(s))
    except: return None

def extract_true_amount(text: str) -> Optional[float]:
    candidates = AMOUNT_RE.findall(text)
    for raw in reversed(candidates):
        val = _clean_amount_token(raw)
        if val is not None and abs(val) >= 1: return val
    return None

def normalize_vendor(text: str) -> str:
    v = text.strip()
    if not v: return "UNKNOWN"
    v = re.sub(r'\d{2,}', '', v) # drop long numbers
    v = re.sub(r'\s+', ' ', v).strip().title()
    return v[:30]

def clean_text_lines(text: str) -> List[str]:
    return [l.strip() for l in text.splitlines() if l.strip()]

class FallbackStatementParser:
    SECTION_PATTERNS = [
        ("DEPOSITS", re.compile(r'\bdeposits\b|\bpayments/credits\b', re.I)),
        ("WITHDRAWALS", re.compile(r'\bwithdrawals\b|\bnew charges\b', re.I)),
        ("FEES", re.compile(r'\bfees\b', re.I)),
    ]

    def _is_summary_line(self, ln: str) -> bool:
        low = ln.lower()
        if any(k in low for k in ["fee", "bank fee"]): return False
        if re.match(r'^(daily ending|opening balance|ending balance|closing balance|total\b|page\s+\d+)', low): return True
        # AMEX FIX
        if any(k in low for k in ["minimum payment due", "pay in full portion", "pay over time portion", "account total", "days in billing period"]): return True
        return False

    def parse_statement(self, lines: List[str]) -> List[Transaction]:
        txs = []
        cleaned = [l for l in lines if not self._is_summary_line(l)]
        blocks = []
        cur_sec = "UNKNOWN"
        cur_block = None

        for ln in cleaned:
            # simple section detect
            for sec, pat in self.SECTION_PATTERNS:
                if pat.search(ln): cur_sec = sec; break
            
            if DATE_AT_START.match(ln):
                if cur_block: blocks.append((cur_block, cur_sec))
                cur_block = [ln]
            elif cur_block is not None:
                cur_block.append(ln)
        
        if cur_block: blocks.append((cur_block, cur_sec))

        for b_lines, sec in blocks:
            txt = " ".join(b_lines)
            dmatch = DATE_AT_START.match(b_lines[0])
            date_raw = dmatch.group(1) if dmatch else ""
            date_norm = _normalize_date_token(date_raw)
            amt = extract_true_amount(txt)
            if amt is None: continue
            
            vendor = normalize_vendor(re.sub(r'^\s*\d+/\d+', '', b_lines[0]))
            txs.append(Transaction(date_norm, "deposit" if amt > 0 else "withdrawal", vendor, amt, txt, txt, sec))
        return txs

def get_text(p):
    with pdfplumber.open(p) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

# --- RUN ---
p = Path("bank statments/AMEX.pdf")
print(f"Reading {p}...")
t = get_text(p)
parser = FallbackStatementParser()
txs = parser.parse_statement(clean_text_lines(t))

print(f"\nTransactions found: {len(txs)}")
trash = [t for t in txs if "-" in t.date and len(t.date) != 10]
print(f"Trash (invalid dates): {len(trash)}")
for tx in txs[:20]:
    print(f"{tx.date} | {tx.vendor:20} | {tx.amount:10.2f} | {tx.section}")

# Check for specific valid tx
matari = [t for t in txs if "MATARI" in t.vendor.upper()]
print(f"\nMatari Coffee transactions: {len(matari)}")
for m in matari:
    print(f"  {m.date} | {m.vendor} | {m.amount}")
