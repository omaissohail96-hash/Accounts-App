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

DATE_AT_START = re.compile(
    r'^[^\w\s]{0,4}\s*('
    r'(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01])(?:[/-]\d{2,4})?|'
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-/ ](?:0?[1-9]|[12][0-9]|3[01])(?:,?\s*[-/ ]\d{2,4})?'
    r')\b(?![\d\.])', re.I
)

AMOUNT_RE = re.compile(r'([+\-]?\(?\s*\$?\d{1,3}(?:[,\s]\d{3})*\.\d{2}\)?)')

def clean_amount(s: str) -> float:
    if not s: return 0.0
    s = s.strip().replace('$', '').replace(',', '').replace(' ', '')
    s = s.replace('¢', '').replace('\u00a2', '').replace('#', '').replace('*', '').replace('+', '')
    if s.endswith('-'): s = '-' + s[:-1]
    s = re.sub(r'[^0-9.\-]', '', s)
    try: return float(s)
    except: return 0.0

class FallbackStatementParser:
    SECTION_PATTERNS = [
        ("DEPOSITS",              re.compile(r'^\s*Payments\s*/\s*Credits\s*$', re.I)),
        ("WITHDRAWALS",           re.compile(r'^\s*(New\s+Charges|Amount|Fees)\s*$', re.I)),
    ]

    def _is_summary_line(self, ln: str) -> bool:
        low = ln.lower().strip()
        if re.match(r'^(daily ending|opening balance|ending balance|closing balance|page\s+\d+|minimum\s+payment|new\s+balance|previous\s+balance|payment\s+due|statement\s+date|total\s+for|automatic\s+payment|account\s+summary|balance\s+summary)', low): return True
        if any(k in low for k in ["total balance", "total new charges"]): return True
        return False

    def parse_statement(self, lines: List[str]) -> List[Transaction]:
        cleaned = [ln.strip() for ln in lines if ln.strip() and not self._is_summary_line(ln)]
        
        txs = []
        cur_sec = "UNKNOWN"
        cur_block = None
        for ln in cleaned:
            if self._is_summary_line(ln) or re.search(r'\btotal\b.*\d', ln, re.I):
                if cur_block: self._proc(cur_block, cur_sec, txs)
                cur_block = None; continue

            matched_section = None
            for sec, pat in self.SECTION_PATTERNS:
                if pat.search(ln) and len(ln.strip()) < 50:
                    matched_section = sec; break
            if matched_section:
                if cur_block: self._proc(cur_block, cur_sec, txs)
                cur_block = None; cur_sec = matched_section; continue

            if DATE_AT_START.match(ln):
                if cur_block: self._proc(cur_block, cur_sec, txs)
                cur_block = [ln]
            elif cur_block is not None:
                cur_block.append(ln)
        if cur_block: self._proc(cur_block, cur_sec, txs)

        # ROBUST DEDUPLICATION
        final_txs = []
        seen = set()
        for t in txs:
            v_norm = (t.vendor or "").strip().lower()[:12]
            key = (t.date, abs(t.amount), v_norm)
            if key not in seen:
                seen.add(key)
                final_txs.append(t)
            else:
                for i, existing in enumerate(final_txs):
                    v_e_norm = (existing.vendor or "").strip().lower()[:12]
                    if (existing.date == t.date and abs(existing.amount) == abs(t.amount) and v_e_norm == v_norm):
                        if len(t.vendor or "") > len(existing.vendor or ""):
                            final_txs[i] = t
                        break
        return final_txs

    def _proc(self, lines, sec, txs):
        txt = " ".join(lines)
        if self._is_summary_line(txt) or re.search(r'\btotal\b.*\d', txt, re.I): return
        dmatch = DATE_AT_START.match(lines[0])
        if not dmatch: return
        amt = self._extract_amt(txt)
        if amt:
            v_text = re.sub(DATE_AT_START, '', lines[0]).strip()
            if not v_text and len(lines) > 2: v_text = lines[2].strip()
            txs.append(Transaction(dmatch.group(1), "deposit" if amt > 0 else "withdrawal", v_text[:30], amt, txt, txt, sec))

    def _extract_amt(self, text: str) -> Optional[float]:
        candidates = AMOUNT_RE.findall(text)
        if not candidates: return None
        return clean_amount(candidates[-1])

# --- RUN ---
p_amex = Path(BASE_DIR) / "bank statments" / "AMEX.pdf"
images = convert_from_bytes(p_amex.read_bytes(), poppler_path=POPPLER_PATH)
text_amex = "\n".join(pytesseract.image_to_string(img) for img in images)
parser = FallbackStatementParser()
txs = parser.parse_statement(text_amex.split('\n'))

print(f"\n--- AMEX FINAL RESULTS ---")
print(f"Total Transactions: {len(txs)}")
for t in txs:
    print(f"  {t.date} | {t.amount:10.2f} | {t.vendor} | {t.section}")
