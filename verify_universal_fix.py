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

# --- RE-IMPLEMENTED LOGIC FROM bank_data_analysis.py ---

@dataclass
class Transaction:
    date: str
    transaction_type: str
    vendor: str
    amount: float
    description: str
    raw_line: str
    section: Optional[str] = None

DATE_RE = re.compile(r'(\d{1,2}\s+[A-Za-z]{3,9},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]?\d{2,4})')
AMOUNT_RE_UNIVERSAL = re.compile(r'([+\-]?\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2}))')

def _short_vendor(v): return v[:20]
def _normalize_date_token(d): return d # simplified
def _clean_amount_token(s):
    s = s.strip().replace(',', '').replace(' ', '')
    try: return float(s)
    except: return None
def _clean_description(d, v, t): return d

class FallbackStatementParser:
    def _is_summary_line(self, ln: str) -> bool:
        low = ln.lower().strip()
        if re.match(r'^(daily ending|opening balance|ending balance|closing balance|page\s+\d+|minimum\s+payment|new\s+balance|previous\s+balance|payment\s+due|statement\s+date|total\s+for|automatic\s+payment|account\s+summary|balance\s+summary)', low): return True
        if any(k in low for k in ["total balance", "total new charges"]): return True
        return False

class UniversalParser:
    IGNORE_WORDS = ["opening balance", "closing balance", "balance", "running balance"]
    
    def parse(self, lines: List[str]) -> List[Transaction]:
        txs = self._parse_simple(lines)
        # Deduplication
        final_txs = []
        seen = set()
        for tx in txs:
            v_norm = (tx.vendor or "").strip().lower()[:12]
            key = (tx.date, abs(tx.amount), v_norm)
            if key not in seen:
                seen.add(key)
                final_txs.append(tx)
            else:
                for i, existing in enumerate(final_txs):
                    v_e_norm = (existing.vendor or "").strip().lower()[:12]
                    if (existing.date == tx.date and abs(existing.amount) == abs(tx.amount) and v_e_norm == v_norm):
                        if len(tx.vendor or "") > len(existing.vendor or ""):
                            final_txs[i] = tx
                        break
        return final_txs

    def _parse_simple(self, lines: List[str]) -> List[Transaction]:
        txs = []
        fb = FallbackStatementParser()
        for ln in lines:
            low = ln.lower().strip()
            if any(w in low for w in self.IGNORE_WORDS) or fb._is_summary_line(ln): continue
            d = DATE_RE.search(ln)
            if not d: continue
            date_raw = d.group(1).replace(",", "")
            date_norm = _normalize_date_token(date_raw)
            m = AMOUNT_RE_UNIVERSAL.findall(ln)
            if not m: continue
            amount_raw = m[-2] if len(m) >= 2 else m[-1]
            amount = _clean_amount_token(amount_raw)
            if amount is None: continue
            ttype = "deposit" if amount > 0 else "withdrawal"
            desc = ln.replace(date_raw, "").replace(amount_raw, "").strip()
            vendor_words = [w for w in desc.split() if w.isalpha() and len(w) > 1]
            vendor = " ".join(vendor_words[:3]).title() if vendor_words else "UNKNOWN"
            txs.append(Transaction(date_norm, ttype, vendor, amount, desc, ln))
        return txs

# --- RUN ---
p_amex = Path(BASE_DIR) / "bank statments" / "AMEX.pdf"
images = convert_from_bytes(p_amex.read_bytes(), poppler_path=POPPLER_PATH)
text_amex = "\n".join(pytesseract.image_to_string(img) for img in images)
up = UniversalParser()
txs = up.parse(text_amex.split('\n'))

print(f"\n--- AMEX UNIVERSAL FALLBACK RESULTS (DEDUPED) ---")
print(f"Total Transactions: {len(txs)}")
for t in txs[:10]:
    print(f"  {t.date} | {t.amount:10.2f} | {t.vendor}")
