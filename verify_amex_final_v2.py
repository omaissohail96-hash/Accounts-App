import re
import os
import pdfplumber
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

@dataclass
class Transaction:
    date: str
    transaction_type: str
    vendor: str
    amount: float
    description: str
    raw_line: str
    section: Optional[str] = None

# TIGHTENED REGEX
DATE_AT_START = re.compile(
    r'^\s*('
    r'(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12][0-9]|3[01])(?:/\d{2,4})?|'
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(?:0?[1-9]|[12][0-9]|3[01])(?:,?\s*\d{2,4})?'
    r')\b', re.I
)

AMOUNT_RE = re.compile(r'([+\-]?\(?\s*\$?\d{1,3}(?:[,\s]\d{3})*\.\d{2}\)?)')

def _normalize_date_token(token: str) -> str:
    t = token.strip().replace(",", "")
    for fmt in ["%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y", "%d/%m/%y", "%m/%d", "%d/%m"]:
        try: return datetime.strptime(t, fmt).replace(year=2025).strftime("%Y-%m-%d")
        except: continue
    return t

def _clean_amount_token(token: str) -> Optional[float]:
    s = str(token).strip()
    negative = "(" in s or "-" in s
    s = re.sub(r'[^0-9.]', '', s)
    try: return -abs(float(s)) if negative else abs(float(s))
    except: return None

def extract_true_amount(text: str) -> Optional[float]:
    candidates = AMOUNT_RE.findall(text)
    if not candidates: return None
    val = _clean_amount_token(candidates[-1])
    return val if val and abs(val) >= 1 else None

def normalize_vendor(text: str) -> str:
    # Use the logic from the actual file
    v = text.strip()
    # Remove phone numbers like 847-779-3141
    v = re.sub(r'\d{3}-\d{3}-\d{4}', '', v)
    v = re.sub(r'\s+', ' ', v).strip().title()
    return v[:30]

class Parser:
    SECTION_PATTERNS = [
        ("DEPOSITS", re.compile(r'^Payments/Credits$', re.I)),
        ("WITHDRAWALS", re.compile(r'^New Charges$', re.I)),
    ]
    def _is_summary_line(self, ln: str) -> bool:
        low = ln.lower()
        if any(k in low for k in ["minimum payment due", "pay in full portion", "pay over time portion", "account total", "days in billing period"]): return True
        return False

    def parse(self, text: str) -> List[Transaction]:
        txs = []
        lines = [l.strip() for l in text.splitlines() if l.strip() and not self._is_summary_line(l.strip())]
        cur_sec = "UNKNOWN"
        cur_block = None
        for ln in lines:
            for sec, pat in self.SECTION_PATTERNS:
                if pat.search(ln): cur_sec = sec; break
            if DATE_AT_START.match(ln):
                if cur_block: self._proc(cur_block, cur_sec, txs)
                cur_block = [ln]
            elif cur_block: cur_block.append(ln)
        if cur_block: self._proc(cur_block, cur_sec, txs)
        return txs

    def _proc(self, lines, sec, txs):
        txt = " ".join(lines)
        dmatch = DATE_AT_START.match(lines[0])
        date_raw = dmatch.group(1)
        amt = extract_true_amount(txt)
        if amt:
            # Vendor is the text after date on first line
            v_text = re.sub(DATE_AT_START, '', lines[0]).strip()
            txs.append(Transaction(_normalize_date_token(date_raw), "deposit" if amt > 0 else "withdrawal", normalize_vendor(v_text), amt, txt, txt, sec))

p = Path("bank statments/AMEX.pdf")
with pdfplumber.open(p) as pdf:
    text = "\n".join(page.extract_text() or "" for page in pdf.pages)

parser = Parser()
all_txs = parser.parse(text)
print(f"Total: {len(all_txs)}")
for t in all_txs:
    print(f"{t.date} | {t.amount:10.2f} | {t.vendor} | {t.section}")
