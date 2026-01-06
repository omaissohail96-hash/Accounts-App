import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# bank_data_analysis.py
# Hybrid Bank Statement Analyzer (deterministic + optional LLM)
# Paste/replace your old file with this and run: streamlit run bank_data_analysis.py

import io
import re
import json
import logging
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

import pdfplumber
import pandas as pd
import streamlit as st
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# ============================
# Schedule C Keyword Rules
# ============================

EXCLUDE_KEYWORDS = [
    "TRANSFER", "ZELLE", "QUICKPAY", "INTERNAL",
    "MOVE MONEY", "ONLINE TRANSFER",
    "OWNER DRAW", "OWNER TRANSFER"
]

LINE_1_GROSS = [
    "PAYROLL", "ACH CREDIT", "ORIG CO NAME",
    "PAYMENT RECEIVED", "STRIPE", "PAYPAL",
    "SQUARE", "SHOPIFY", "TIKTOK INC",
    "AMAZON", "META", "GOOGLE"
]

LINE_8_ADVERTISING = [
    "FACEBOOK", "META ADS", "GOOGLE ADS",
    "TIKTOK ADS", "ADWORDS", "PROMOTION"
]

LINE_9_VEHICLE = [
    "GAS", "FUEL", "PETROL", "SHELL",
    "CHEVRON", "EXXON", "PARKING", "TOLL"
]

LINE_11_CONTRACT = [
    "FREELANCER", "UPWORK", "FIVERR",
    "CONTRACTOR", "CONSULTANT"
]

LINE_17_LEGAL = [
    "LAW", "LEGAL", "ATTORNEY",
    "CPA", "ACCOUNTANT", "BOOKKEEP"
]

LINE_18_OFFICE = [
    "BANK FEE", "SERVICE FEE", "MONTHLY FEE",
    "ADOBE", "MICROSOFT", "ZOOM",
    "HOSTING", "DOMAIN", "GODADDY"
]

LINE_21_REPAIRS = [
    "REPAIR", "MAINTENANCE", "FIX"
]

LINE_22_SUPPLIES = [
    "SUPPLIES", "STATIONERY", "OFFICE DEPOT",
    "INK", "PAPER"
]

LINE_23_TAXES = [
    "LICENSE", "PERMIT", "TAX", "GOVERNMENT FEE"
]

LINE_24A_TRAVEL = [
    "AIRLINE", "FLIGHT", "HOTEL",
    "BOOKING.COM", "UBER TRIP", "LYFT TRIP"
]

LINE_24B_MEALS = [
    "RESTAURANT", "CAFE", "FOOD",
    "STARBUCKS", "MCDONALD", "KFC"
]

LINE_25_UTILITIES = [
    "INTERNET", "MOBILE", "PHONE",
    "ELECTRIC", "WATER"
]

logger = logging.getLogger("bank_analyzer")
logging.basicConfig(level=logging.ERROR)

# ----------------------------
# Data model
# ----------------------------
@dataclass
class Transaction:
    date: str
    transaction_type: str    # 'deposit' or 'withdrawal'
    vendor: str
    amount: float            # signed: deposits > 0, withdrawals < 0
    description: str
    raw_line: str
    section: Optional[str] = None
    category: Optional[str] = None
    needs_review: bool = False
    source: Optional[str] = "BANK"

# ----------------------------
# Utilities
# ----------------------------
DATE_TOKEN_RE = re.compile(r'(?P<d>\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]{3,9}\s*\d{0,4})')
DATE_AT_START = re.compile(r'^\s*(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b')
AMOUNT_RE = re.compile(
    r'([+\-]?\(?\s*\$?\d{1,3}(?:[,\s]\d{3})*\.\d{2}\)?)'
)
MULTI_DATE_AMT_RE = re.compile(r'(\d{1,2}[/-]\d{1,2}|[+\-]?\(?\s*\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})\)?)')
CHECK_ROW_RE = re.compile(r'^\s*(\d{2,6})\b') 
FEE_KEYWORDS_RE = re.compile(
    r'\b(fee|service fee|monthly fee|maintenance fee|bank fee|account fee)\b',
    re.I
)
CHECK_NO_RE = re.compile(r'\bcheck\s*(\d+)\b', re.I)
OPENING_BALANCE_RE = re.compile(
    r'(opening balance|beginning balance)[^\d\-]*([\$]?\(?\d{1,3}(?:,\d{3})*(?:\.\d{2})\)?)',
    re.IGNORECASE
)

# Global variable to store the inferred statement year
_STATEMENT_YEAR = None

def _normalize_date_token(token: str) -> str:
    if not token:
        return ""
    t = token.strip().replace(",", "")
    # try common formats, prefer m/d or d/m with assumption current year when no year
    formats = [
        "%m/%d/%Y", "%m/%d/%y",
        "%d/%m/%Y", "%d/%m/%y",
        "%Y-%m-%d",
        "%m/%d", "%d/%m",
        "%d %b %Y", "%d %b %y", "%b %d %Y", "%B %d %Y"
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(t, fmt)
            if "%Y" not in fmt and "%y" not in fmt:
                # Use statement year if available, otherwise infer from most recent past year
                global _STATEMENT_YEAR
                if _STATEMENT_YEAR:
                    year = _STATEMENT_YEAR
                else:
                    # Infer year: if month is in the future, use last year
                    now = datetime.now()
                    year = now.year
                    if dt.month > now.month:
                        year -= 1
                dt = dt.replace(year=year)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue
    return t  # fallback: return raw token

def _md_key(d: str):
    """Return (month, day) tuple from YYYY-MM-DD or similar"""
    try:
        dt = datetime.strptime(d, "%Y-%m-%d")
        return (dt.month, dt.day)
    except Exception:
        return None

def _clean_amount_token(token: str) -> Optional[float]:
    if not token:
        return None
    s = str(token).strip()
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
    if s.startswith("-"):
        negative = True
    s = re.sub(r'[A-Za-z\$£€₹]', '', s)  # drop currency letters
    s = s.replace(',', '').replace(' ', '')
    s = re.sub(r'[^0-9\.\-]', '', s)
    if not re.search(r'\d', s):
        return None
    parts = s.split('.')
    if len(parts) > 2:
        s = "".join(parts[:-1]) + "." + parts[-1]
    try:
        val = float(s)
    except Exception:
        return None
    return -abs(val) if negative else abs(val)

def _short_vendor(v: str) -> str:
    if not v:
        return "UNKNOWN"
    
    # Remove common ID patterns and codes
    v2 = re.sub(r'\b(id|ref|code|num|number)[:\s]*\d+', '', v, flags=re.I)
    v2 = re.sub(r'\b\d{5,}\b', '', v2)  # Remove long numeric IDs
    
    # Remove ACH noise words and prefixes
    v2 = re.sub(r'\b(orig|co|name|entry|descr|desc)\b', '', v2, flags=re.I)
    
    # Clean special characters but keep important ones
    v2 = re.sub(r'[^A-Za-z0-9\-\&\.\s]', ' ', v2)
    v2 = re.sub(r'\s{2,}', ' ', v2).strip()
    
    if not v2:
        return "UNKNOWN"
    
    # Title case and limit length
    v2 = v2.title()
    if len(v2) > 30:
        v2 = v2[:30] + "..."
    return v2

from datetime import datetime, date

def filter_by_month_day_range(
    transactions: List[Transaction],
    start_md: Tuple[int, int],
    end_md: Tuple[int, int]
) -> List[Transaction]:

    out = []
    for tx in transactions:
        md = _md_key(tx.date)
        if not md:
            continue

        # handle wrap-around (e.g. Dec → Jan)
        if start_md <= end_md:
            if start_md <= md <= end_md:
                out.append(tx)
        else:
            if md >= start_md or md <= end_md:
                out.append(tx)

    return out


def _clean_description(desc: str, vendor: str = "", transaction_type: str = "") -> str:
    """
    Rewrite transaction description into a simple, human-readable format.
    Removes technical codes, IDs, timestamps, and trace numbers.
    Returns a clean, natural language description that anyone can understand.
    """
    if transaction_type == "withdrawal" and vendor.startswith("Check #"):
        return vendor

    if not desc:
        if transaction_type == "deposit":
            return f"Money received from {vendor}" if vendor else "Money received"
        elif transaction_type == "withdrawal":
            return f"Payment made to {vendor}" if vendor else "Payment made"
        return "Transaction processed"
    
    original = desc.strip()
    d = original
    
    # Remove leading dates
    d = re.sub(r'^(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{1,2})\s+', '', d)
    
    # Extract meaningful parts before removing everything
    # Check for specific transaction types
    is_online_transfer = bool(re.search(r'online\s+transfer', d, re.I))
    is_ach_payment = bool(re.search(r'ach\s+payment', d, re.I))
    is_check = bool(re.search(r'\bcheck\b|\bchk\b', d, re.I))
    is_card_payment = bool(re.search(r'card\s+(payment|ending)', d, re.I))
    is_fee = bool(re.search(r'\bfee\b', d, re.I))
    is_interest = bool(re.search(r'\binterest\b', d, re.I))
    
    # Extract check number if present
    check_num = None
    check_match = re.search(r'(?:check|chk)\s*\.?\.\.\s*(\d+)', d, re.I)
    if check_match:
        check_num = check_match.group(1)
    
    # Extract card ending digits if present
    card_ending = None
    card_match = re.search(r'(?:card\s+)?ending\s+(?:in\s+)?(\d{4})', d, re.I)
    if card_match:
        card_ending = card_match.group(1)
    
    # Now clean the description
    # Remove IDs, reference numbers, trace numbers
    d = re.sub(r'\b(id|ref|reference|trace|seq|number|num|code)[:\s#]*\d+', '', d, flags=re.I)
    d = re.sub(r'\b\d{6,}\b', '', d)  # Long numeric IDs (6+ digits)
    d = re.sub(r'\b(orig|co|entry|descr?)\s+(id|date|num)\b[:\s]*\S*', '', d, flags=re.I)
    
    # Remove timestamps and dates within description
    d = re.sub(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', '', d)
    d = re.sub(r'\d{4}-\d{2}-\d{2}', '', d)
    
    # Remove technical ACH/banking keywords (but preserve context)
    d = re.sub(r'\b(ppd|ccd|web|tel|auth|authorization)\b', '', d, flags=re.I)
    
    # Remove "Co Name", "Orig Co Name", "Co Entry Descr" patterns
    d = re.sub(r'\b(orig\s+)?(co|company)\s+(name|entry|descr?)\b', '', d, flags=re.I)
    
    # Remove transaction type prefixes
    d = re.sub(r'^\s*(online\s+)?(ach\s+)?(payment|transfer|withdrawal|deposit|debit|credit)\s+(to|from)\s+', '', d, flags=re.I)
    
    # Clean special characters but keep basic punctuation
    d = re.sub(r'[^A-Za-z0-9\s\.\,\-]', ' ', d)
    d = re.sub(r'\s{2,}', ' ', d).strip()
    
    # Build natural language description
    result = ""
    
    if transaction_type == "deposit":
        if vendor and vendor != "UNKNOWN":
            result = f"Deposit received from {vendor}"
        else:
            result = "Deposit received"
        
        # Add context if available
        if d and len(d) > 5 and d.lower() != vendor.lower():
            result += f" - {d}"
    
    elif transaction_type == "withdrawal":
        prefix = "Payment made to"
        
        if is_check and check_num:
            prefix = f"Check payment #{check_num} to"
        elif is_check:
            prefix = "Check payment to"
        elif is_card_payment and card_ending:
            prefix = f"Card payment (ending {card_ending}) to"
        elif is_card_payment:
            prefix = "Card payment to"
        elif is_online_transfer:
            prefix = "Online transfer to"
        elif is_ach_payment:
            prefix = "Electronic payment to"
        elif is_fee:
            prefix = "Fee charged by"
        
        if vendor and vendor != "UNKNOWN":
            result = f"{prefix} {vendor}"
        else:
            result = prefix.replace(" to", "").replace(" by", "")
        
        # Add context if available
        if d and len(d) > 5 and d.lower() != vendor.lower():
            result += f" - {d}"
    
    else:
        result = f"Transaction with {vendor}" if vendor else "Transaction processed"
        if d and len(d) > 5:
            result += f" - {d}"
    
    # Add special context
    if is_interest and "interest" not in result.lower():
        result += " (interest)"
    
    # Limit length but try to keep meaningful content
    if len(result) > 150:
        result = result[:147] + "..."
    
    return result.strip() or "Transaction processed"

# ----------------------------
# Document parsing helpers
# ----------------------------
class DocumentParser:
    def parse_pdf_text_lines(self, file_bytes: bytes) -> Tuple[List[str], List[int]]:
        lines = []
        unreadable_pages = []

        # ---------- TRY NORMAL PDF TEXT ----------
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    try:
                        text = page.extract_text() or ""
                        text = text.replace('\xa0', ' ').strip()
                        if text:
                            lines.extend(
                                [ln.strip() for ln in text.splitlines() if ln.strip()]
                            )
                    except Exception:
                        continue

            # If text extracted successfully → return
            if lines:
                return lines, unreadable_pages

        except Exception as e:
            logger.warning("pdfplumber failed, switching to OCR-only mode")

        # ---------- OCR FALLBACK (IMAGE-ONLY PDF) ----------
        try:
            from pdf2image import convert_from_bytes
            from PIL import Image
            import pytesseract

            if not file_bytes or len(file_bytes) < 100:
                raise ValueError("PDF bytes are empty or invalid")

            pdf_bytes = bytes(file_bytes)  # force fresh copy

            images = convert_from_bytes(
                pdf_bytes,
                dpi=300,
                poppler_path=r"C:\poppler\poppler-25.12.0\Library\bin"
            )


            for i, img in enumerate(images):
                try:
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text.strip():
                        lines.extend(
                            [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
                        )
                    else:
                        unreadable_pages.append(i + 1)
                except Exception:
                    unreadable_pages.append(i + 1)

            return lines, unreadable_pages

        except Exception as e:
            logger.exception("OCR failed completely: %s", e)
            return [], []



    def parse_document(self, file_bytes: bytes, filename: str) -> Tuple[List[str], bool, List[int]]:
        ext = filename.lower().split('.')[-1]
        if ext == "pdf":
            lines, unreadable = self.parse_pdf_text_lines(file_bytes)
            return lines, len(lines) > 0, unreadable
        if ext == "csv":
            try:
                txt = file_bytes.decode("utf-8", errors="ignore")
            except Exception:
                txt = str(file_bytes)
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            return lines, True, []
        if ext in ("doc", "docx"):
            try:
                import docx
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.' + ext)
                tmp.write(file_bytes)
                tmp.flush()
                doc = docx.Document(tmp.name)
                lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
                return lines, True, []
            except Exception as e:
                logger.exception("DOCX parse failed: %s", e)
                return [], False, []
        try:
            txt = file_bytes.decode("utf-8", errors="ignore")
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            return lines, True, []
        except Exception:
            return [], False, []
    
# ----------------------------
# Fallback parser (Chase-optimized, robust)
# ----------------------------
def extract_true_amount(text: str) -> Optional[float]:
    """
    Extracts the REAL transaction amount from a line/block.
    Rules:
    - Must have decimal OR sign OR parentheses
    - Ignore dates, IDs, reference numbers
    - Prefer LAST valid monetary value (bank standard)
    """

    candidates = re.findall(
        r'([+\-]?\(?\s*\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})\)?)',
        text
    )

    for raw in reversed(candidates):
        val = _clean_amount_token(raw)
        if val is None:
            continue

        # 🔒 hard guard: ignore tiny numbers that look like IDs
        if abs(val) < 1:
            continue

        return val

    return None

class FallbackStatementParser:
    def __init__(self, include_opening_balance: bool = False):
        self.include_opening_balance = include_opening_balance
        self.opening_balance: Optional[float] = None
        self.statement_start_date: Optional[str] = None
        self.statement_year: Optional[int] = None

    SECTION_PATTERNS = {
        "DEPOSITS": re.compile(r'\bdeposits\s+and\s+additions\b', re.I),
        "CHECKS": re.compile(r'\bchecks\s+paid\b', re.I),
        "ATM": re.compile(r"ATM\s*&\s*DEBIT\s*CARD\s*WITHDRAWALS", re.I),
        "ELECTRONIC_WITHDRAWALS": re.compile(r'\belectronic\s+withdrawals?\b', re.I),
        "FEES": re.compile(
             r'(monthly\s+service\s+fee|service\s+fee|bank\s+fee|fees\s+charged)',
            re.I),

    }
    def _extract_opening_balance(self, line: str):
        if self.opening_balance is not None:
            return

        m = OPENING_BALANCE_RE.search(line)
        if not m:
            return

        raw = m.group(2)
        amt = float(raw.replace('$', '').replace(',', '').replace('(', '').replace(')', ''))
        self.opening_balance = amt
    
    def _extract_statement_year(self, lines: List[str]):
        """Extract year from statement period or date patterns in the document"""
        # Look for patterns like "Statement Period: 11/01/2025 - 11/30/2025" or dates with year
        for line in lines[:50]:  # Check first 50 lines
            # Pattern 1: Statement period with year
            m = re.search(r'(statement period|period|dates?).*?(\d{1,2}[/-]\d{1,2}[/-](\d{4}))', line, re.I)
            if m:
                year = int(m.group(3))
                if 2000 <= year <= datetime.now().year + 1:
                    return year
            
            # Pattern 2: Any full date with 4-digit year
            m = re.search(r'\b\d{1,2}[/-]\d{1,2}[/-](\d{4})\b', line)
            if m:
                year = int(m.group(1))
                if 2000 <= year <= datetime.now().year + 1:
                    return year
            
            # Pattern 3: YYYY-MM-DD format
            m = re.search(r'\b(\d{4})-\d{1,2}-\d{1,2}\b', line)
            if m:
                year = int(m.group(1))
                if 2000 <= year <= datetime.now().year + 1:
                    return year
        
        return None
    def _is_summary_line(self, ln: str) -> bool:
        low = ln.lower()
        if any(k in low for k in ["fee", "service fee", "monthly fee", "bank fee"]):
            return False
        if not ln or not ln.strip():
            return True
        low = ln.lower().strip()
        # lines that are obvious headings or totals
        if re.match(r'^(daily ending balance|daily ending|daily ending balance|statement period|opening balance|ending balance|closing balance|total\b|page\s+\d+)', low):
            return True
        # If the line contains multiple date+amount pairs (daily ending tables) skip
        # Count date tokens and amount tokens; if >1 of each, it's probably a table column row
        date_count = len(DATE_TOKEN_RE.findall(ln))
        amount_count = len(AMOUNT_RE.findall(ln))
        if (
            "atm" not in low
            and "check" not in low
            and "fee" not in low
            and "ach" not in low
            and date_count >= 2
            and amount_count >= 2
        ):
            return True

        # "TOTAL DEPOSITS" or similar as a whole line
        if re.search(r'\btotal deposits\b|\btotal withdrawals\b|\bdeposits and additions summary\b', low):
            return True
        print("SKIPPED:", ln) 
        return False

    def _line_has_vendor_like_text(self, ln: str) -> bool:
        text = ln.lower()

        # ALWAYS accept ATM and FEES
        if "fee" in text:
            return True  # <-- ADD THIS LINE

        if "atm" in text:
            return True

        text = re.sub(r'^\s*\d{1,2}[/-]\d{1,2}(?:/\d{2,4})?\s*', '', text)
        text = AMOUNT_RE.sub('', text)

        return bool(re.search(r'[a-z]{3,}', text))
    def _parse_date(self, token: str) -> Optional[str]:
        try:
            # Normalize forms like 12/03 -> YYYY-MM-DD
            norm = _normalize_date_token(token)
            return norm
        except Exception:
            return None

    def _parse_amount(self, token: str) -> Optional[float]:
        return _clean_amount_token(token)

    # ------------------------
    # Improved vendor extractor
    # ------------------------
    def extract_vendor(self, desc: str, *args) -> str:

        if not desc:
            return "UNKNOWN"

        text = desc
        if not desc:
            return "UNKNOWN"

        d = desc.strip()
        
        # Remove ALL leading date patterns aggressively
        # Match: "21 ", "12/03 ", "12-03 ", etc.
        d = re.sub(r'^\d{1,2}\s+', '', d)  # Remove day/month at start
        d = re.sub(r'^\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\s+', '', d)  # Remove MM/DD or MM/DD/YYYY
        
        # Remove common transaction prefixes MORE AGGRESSIVELY
        # Match any combination of: "21 Payment To", "Online Transfer To", "Online Ach Payment To"
        d = re.sub(r'^\d{1,2}\s+', '', d)  # Remove any leading numbers again after first pass
        d = re.sub(r'^\s*(online\s+)?(ach\s+)?(payment|transfer|withdrawal|deposit)\s+(to|from)\s+', '', d, flags=re.I)
        
        # ACH/Electronic transaction patterns - extract company name only
        # Pattern 1: "Orig Co Name COMPANY NAME Orig Id 123..."
        m = re.search(r'\b(?:orig\s+)?co\s+name\s+([A-Za-z0-9\s\-\.&]+?)(?:\s+(?:orig\s+)?(?:co\s+)?id\b)', d, re.I)
        if m:
            return _short_vendor(m.group(1).strip())
        
        # Pattern 2: "Desc COMPANY NAME Co Entry Descr..."
        m = re.search(r'\bdesc\s+([A-Za-z0-9\s\-\.&]+?)(?:\s+(?:co\s+entry|orig\s+id|desc\s+date)\b)', d, re.I)
        if m:
            candidate = m.group(1).strip()
            if not re.match(r'^\d+$', candidate):  # Skip if only digits
                return _short_vendor(candidate)
        
        # Pattern 3: "Co Entry Descr COMPANY NAME"
        m = re.search(r'\bco\s+entry\s+descr\s+([A-Za-z0-9\s\-\.&]+)', d, re.I)
        if m:
            return _short_vendor(m.group(1).strip())

        # ---------------------------
        # REMOVE DATE & AMOUNT FIRST
        # ---------------------------
        text = re.sub(r'\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b', ' ', text)   # dates
        text = re.sub(r'\$?\(?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})\)?', ' ', text)  # amounts
        text = re.sub(r'\bUSD\b|\bPKR\b|\bEUR\b', ' ', text, flags=re.I)

        # ---------------------------
        # 1️⃣ ACH FORMAT (MOST IMPORTANT)
        # Orig CO Name:Shopify
        # ---------------------------
        m = re.search(r'orig\s+co\s+name\s*:\s*([A-Za-z][A-Za-z0-9 &\-\.]{2,})', text, re.I)
        if m:
            return _short_vendor(m.group(1))

        # ---------------------------
        # 2️⃣ Merchant / Payee patterns
        # ---------------------------
        
        m = re.search(r'\b(payee|merchant|beneficiary)\s*[:\-]\s*([A-Za-z][A-Za-z0-9 &\-\.]{2,})', text, re.I)
        if m:
            return _short_vendor(m.group(2))

        # ---------------------------
        # 3️⃣ From / To
        # ---------------------------
        m = re.search(r'\bfrom\s+([A-Za-z][A-Za-z0-9 &\-\.]{2,})', text, re.I)
        if m:
            return _short_vendor(m.group(1))

        m = re.search(r'\bto\s+([A-Za-z][A-Za-z0-9 &\-\.]{2,})', text, re.I)
        if m:
            return _short_vendor(m.group(1))

        # ---------------------------
        # 4️⃣ Slash based (last part)
        # ---------------------------
        if "/" in text:
            after = text.split("/")[-1]
            if re.search(r"[A-Za-z]{3,}", after):
                return _short_vendor(after)

        # ---------------------------
        # 5️⃣ SAFE TOKEN FALLBACK
        # ---------------------------
        stopwords = {
            "orig", "co", "name", "desc", "date", "entry",
            "sec", "ccd", "trace", "id", "payment", "transfer"
        }


        tokens = re.findall(r'[A-Za-z]{3,}', text)
        for tok in tokens:
            if tok.lower() not in stopwords:
                return _short_vendor(tok)

        m = re.search(r'\bto\s+([A-Za-z0-9\-\.\s&]+)', d, re.I)
        if m:
            return _short_vendor(m.group(1).strip())

        # 3) look for patterns like "PAYEE: XYZ" or "REMIT: XYZ"
        m = re.search(r'\b(payee|remit|beneficiary|merchant)[:\-]\s*([A-Za-z0-9\-\.\s&]+)', d, re.I)
        if m:
            return _short_vendor(m.group(2).strip())

        # 4) general fallback: skip numeric tokens and dates, get first meaningful text
        tokens = d.split()
        for token in tokens:
            # Skip pure numbers, dates, and common noise words
            if re.search(r"[A-Za-z]", token) and not re.match(r'^\d+[/-]\d+$', token):
                # Get first 3 meaningful words for compound names
                idx = tokens.index(token)
                name_tokens = []
                for t in tokens[idx:idx+3]:
                    if re.search(r"[A-Za-z]", t) and not t.lower() in ['id', 'ref', 'code', 'orig']:
                        name_tokens.append(t)
                if name_tokens:
                    return _short_vendor(" ".join(name_tokens))
                return _short_vendor(token)


        return "UNKNOWN"
    def parse_statement(self, lines: List[str]) -> Tuple[List[Transaction], Dict[str, Any]]:
        txs: List[Transaction] = []
        
        # Extract statement year first
        self.statement_year = self._extract_statement_year(lines)
        global _STATEMENT_YEAR
        _STATEMENT_YEAR = self.statement_year
        
        # 1. Pre-clean
        for ln in lines:
            self._extract_opening_balance(ln)
        cleaned = [ln for ln in (l.strip() for l in lines) if ln and not self._is_summary_line(ln)]
        forced_lines = []

        for ln in cleaned:
            # CHECK pattern
            if re.search(r'\b\d{3,6}\b.*\d{1,3}(?:,\d{3})*\.\d{2}', ln):
                forced_lines.append(ln)

            # FEE pattern
            elif re.search(r'\bfee\b', ln, re.I) and re.search(r'\d+\.\d{2}', ln):
                forced_lines.append(ln)

        cleaned = list(dict.fromkeys(cleaned + forced_lines))

        if not cleaned:
            return [], {"parsed_from": "fallback", "transactions_extracted": 0}

        # 2. segment into blocks starting with a date at start (Chase style)
        blocks: List[Tuple[List[str], str]] = []
        current_section = "UNKNOWN"
        current_block: Optional[List[str]] = None
        ALWAYS_ALLOW_SECTIONS = {"ATM", "CHECKS", "FEES"}
        for ln in cleaned:
            # detect section headers
            matched_section = None
            for sec_name, pat in self.SECTION_PATTERNS.items():
                if pat.search(ln):
                    matched_section = sec_name
                    break
            if matched_section:
                # finalize previously collected block
                if current_block:
                    blocks.append((current_block, current_section))
                    current_block = None
                current_section = matched_section
                continue

            # must start with a date to start a new block; ignore stray lines that aren't transactions
            is_check_row = current_section == "CHECKS" and CHECK_ROW_RE.match(ln)
            is_fee_row   = current_section == "FEES" and DATE_AT_START.match(ln)
            # ---- PATCH: CHECKS PAID START ----
            if current_section == "CHECKS" and re.match(r'^\d{2,6}\s', ln):
                    if current_block:
                        blocks.append((current_block, current_section))
                    current_block = [ln]
                    continue
            # ---- PATCH: CHECKS PAID END ----

            # FORCE fees to start a block even if section header is missing
            is_fee_line = (
                DATE_AT_START.match(ln)
                and FEE_KEYWORDS_RE.search(ln)
            )

            if DATE_AT_START.match(ln) or is_check_row or is_fee_row or is_fee_line:
                if current_block:
                    blocks.append((current_block, current_section))
                current_block = [ln]

                # hard-lock section
                if is_fee_line:
                    current_section = "FEES"

            else:
                if current_block is not None:
                    current_block.append(ln)
            if DATE_AT_START.match(ln):
                # ALLOW CHECKS & FEES even without vendor text
                if current_section not in ("ATM", "CHECKS", "FEES"):
                    if not self._line_has_vendor_like_text(ln):
                        continue
                
                    continue        
            # ---- PATCH: FEES PROTECTION ----
            if current_section == "FEES":
                pass  # never skip fee lines
            elif self._is_summary_line(ln):
                continue
            # ---- PATCH END ----
        
        if current_block:
            blocks.append((current_block, current_section))
        # 3. parse each block
        # ---- HARD CHECKS OVERRIDE ----
        check_lines = []
        in_checks = False
        for ln in cleaned:
            # Check if this is a checks section header
            if self.SECTION_PATTERNS["CHECKS"].search(ln):
                in_checks = True
                continue  # Skip the header line itself
            # Check if we're exiting the checks section
            if in_checks and any(self.SECTION_PATTERNS[s].search(ln) for s in ["ATM", "FEES", "ELECTRONIC_WITHDRAWALS"]):
                in_checks = False
                continue
            # If we're in checks section, add the line
            if in_checks:
                # Only add lines that look like check transactions (start with check number)
                if re.match(r'^\d{3,6}\s', ln):
                    check_lines.append(ln)

        check_txs = self._parse_checks_section(check_lines)
        txs.extend(check_txs)
        # ---- END CHECK OVERRIDE ----

        for block_lines, section in blocks:
            if section == "CHECKS":
                continue
            block_text = " ".join(block_lines)
            # skip if looks like a total or header inside
            if section != "FEES" and re.search(r'\btotal\b.*\d', block_text, re.I):
                continue
            # get date from first line
            first_line = block_lines[0]
            dmatch = DATE_AT_START.match(first_line)
            date_raw = dmatch.group(1) if dmatch else ""
            date_norm = self._parse_date(date_raw) or ""
            if not self.statement_start_date and date_norm:
                self.statement_start_date = date_norm
            # amounts: choose last amount-like token
            amounts = AMOUNT_RE.findall(block_text)
            if not amounts:
                continue
            amount_raw = amounts[-1]
            amt_val = extract_true_amount(block_text)
            if amt_val is None:
                continue

            # determine direction strictly by section where possible
            # determine direction strictly by section where possible
            # determine direction STRICTLY by section
            if section == "DEPOSITS":
                signed_amount = abs(amt_val)

            elif section == "ATM":
                signed_amount = -abs(amt_val)
            # --- FORCE BANK FEES ---
            elif re.search(r'\bfee\b', block_text, re.I):
                signed_amount = -abs(amt_val)
                vendor = "Bank Fees"

            elif section in ("CHECKS", "ELECTRONIC_WITHDRAWALS"):
                signed_amount = -abs(amt_val)

            else:
                # fallback ONLY if section is unknown
                if amount_raw.strip().startswith("(") or "-" in amount_raw:
                    signed_amount = -abs(amt_val)
                else:
                    signed_amount = abs(amt_val)




            vendor = self.extract_vendor(block_text, date_raw, amount_raw)
            if section == "CHECKS":
                pass
            if section == "FEES":
                vendor = "Bank Fees"
            else:
                vendor = self.extract_vendor(block_text, date_raw, amount_raw)

            if section == "ATM":
                vendor = "ATM Withdrawal"
            needs_review = abs(signed_amount) >= 20000.0

            # final safety: skip transactions that have UNKNOWN vendor and appear to be balance-only rows
            if vendor == "UNKNOWN":
                txt_low = block_text.lower()

                # ❌ skip ONLY balances, NEVER ATM
                if section not in ("ATM",) and any(k in txt_low for k in [
                    "daily ending", "ending balance", "opening balance", "closing balance"
                ]):
                    continue



            ttype = 'deposit' if signed_amount > 0 else 'withdrawal'
            txs.append(Transaction(
                date=date_norm,
                transaction_type=ttype,
                vendor=vendor,
                amount=signed_amount,
                description=_clean_description(block_text, vendor, ttype),
                raw_line=block_text,
                section=section,
                needs_review=needs_review
            ))

        
        # ----------------------------------
        # Inject Opening Balance (OPTIONAL)
        # ----------------------------------
        if (
            self.include_opening_balance
            and self.opening_balance is not None
        ):
            txs.insert(0, Transaction(
                date=self.statement_start_date or "",
                transaction_type="deposit",
                vendor="Opening Balance",
                amount=abs(self.opening_balance),
                description="Opening balance from bank statement",
                raw_line="OPENING_BALANCE",
                section="OPENING",
                needs_review=False
            ))
        meta = {"parsed_from": "chase_sectioned_fallback", "transactions_extracted": len(txs)}
        return txs, meta
    def _parse_checks_section(self, lines: List[str]) -> List[Transaction]:
        txs = []

        for ln in lines:
            # Chase checks start with check number, any text, then amount
            # Pattern: CHECKNO [anything] AMOUNT
            m = re.match(r'^(\d{3,6})\s+.*?(\d{1,3}(?:,\d{3})*\.\d{2})$', ln)
            if not m:
                continue
            
            check_no = m.group(1)
            amt_raw = m.group(2)
            
            amt = self._parse_amount(amt_raw)
            if amt is None:
                continue

            txs.append(Transaction(
                date="",  # Chase check dates are separate column
                transaction_type="withdrawal",
                vendor=f"Check #{check_no}",
                amount=-abs(amt),
                description=ln,  # Full line as description
                raw_line=ln,
                section="CHECKS",
                needs_review=False
            ))

        return txs



# ----------------------------
# Universal parser fallback for simpler bank statements (SadaPay, sample banks)
# - conservative: only picks blocks that start with a full date (YYYY-MM-DD or MM/DD) and have a description
# ----------------------------
class UniversalParser:
    """
    Universal Smart Parser (FINAL)
    Supports:
        • SadaPay 2-line format
        • Single-line bank CSV-like PDFs
        • Multi-column PDFs (date | description | debit | credit | balance)
        • Financial statements with inline "Debit"/"Credit"
    """

    # Matches wide range of date formats
    DATE_RE = re.compile(
        r'(\d{1,2}\s+[A-Za-z]{3,9},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]?\d{2,4})'
    )

    # Matches debit/credit amount ONLY (NOT balance)
    AMOUNT_RE = re.compile(
        r'([+\-]?\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2}))'
    )

    IGNORE_WORDS = ["opening balance", "closing balance", "balance", "running balance"]

    def parse(self, lines: List[str]) -> List[Transaction]:
        txs = []

        # -------- DETECT FORMAT --------
        is_sadapay = any("transf" in ln.lower() or "cr/" in ln.lower() or "dr/" in ln.lower() for ln in lines)
        is_tabular = any("Debit" in ln or "Credit" in ln for ln in lines)

        if is_sadapay:
            return self._parse_sadapay(lines)

        elif is_tabular:
            return self._parse_tabular(lines)

        else:
            return self._parse_simple(lines)

    # ===================================================================================
    # 1) S A D A P A Y    P A R S E R
    # ===================================================================================
    def _parse_sadapay(self, lines: List[str]) -> List[Transaction]:
        txs = []
        i = 0
        while i < len(lines):
            ln = lines[i].strip()

            # A SadaPay block always starts with date line
            d = self.DATE_RE.search(ln)
            if d:
                # Next line must be "TIME + REF"
                if i + 1 < len(lines):
                    line2 = lines[i+1].strip()
                else:
                    i += 1
                    continue

                block = ln + " " + line2

                # Skip balance lines
                if any(x in block.lower() for x in self.IGNORE_WORDS):
                    i += 2
                    continue

                # Parse date
                date_raw = d.group(1).replace(",", "")
                date_norm = _normalize_date_token(date_raw)

                # Parse amount
                amt_m = self.AMOUNT_RE.search(ln)
                if not amt_m:
                    i += 2
                    continue
                amount_raw = amt_m.group(1).replace(" ", "")
                amount_val = _clean_amount_token(amount_raw)
                if amount_val is None:
                    i += 2
                    continue

                ttype = "deposit" if amount_val > 0 else "withdrawal"

                # Description (SadaPay format)
                desc = ln.replace(date_raw, "").replace(amount_raw, "").strip() + " | " + line2

                # Vendor extraction - clean and normalize
                vendor_words = [w for w in desc.split() if w.isalpha()]
                vendor = " ".join(vendor_words[:3]) if vendor_words else "UNKNOWN"
                vendor = _short_vendor(vendor)

                txs.append(Transaction(
                    date=date_norm,
                    transaction_type=ttype,
                    vendor=vendor,
                    amount=amount_val,
                    description=_clean_description(desc, vendor, ttype),
                    raw_line=ln
                ))

                i += 2
                continue

            i += 1

        return txs
    def _extract_vendor(self, desc: str) -> str:
        """Extract clean vendor name using same logic as FallbackStatementParser."""
        if not desc:
            return "UNKNOWN"

        d = desc.strip()
        
        # Remove ALL leading date patterns aggressively
        d = re.sub(r'^\d{1,2}\s+', '', d)  # Remove day/month at start
        d = re.sub(r'^\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\s+', '', d)  # Remove MM/DD or MM/DD/YYYY
        
        # Remove transaction type prefixes MORE AGGRESSIVELY
        d = re.sub(r'^\d{1,2}\s+', '', d)  # Remove any leading numbers again
        d = re.sub(r'^\s*(online\s+)?(ach\s+)?(payment|transfer|withdrawal|deposit)\s+(to|from)\s+', '', d, flags=re.I)
        
        # ACH patterns
        m = re.search(r'\b(?:orig\s+)?co\s+name\s+([A-Za-z0-9\s\-\.&]+?)(?:\s+(?:orig\s+)?(?:co\s+)?id\b)', d, re.I)
        if m:
            return _short_vendor(m.group(1).strip())
        
        # Prefer text after slash
        if "/" in d:
            after = d.split("/")[-1].strip()
            if re.search(r"[A-Za-z]", after):
                return _short_vendor(after)

        # FROM xyz
        m = re.search(r'\bfrom\s+([A-Za-z0-9\s\-\.&]+)', d, re.I)
        if m:
            return _short_vendor(m.group(1).strip())

        # TO xyz
        m = re.search(r'\bto\s+([A-Za-z0-9\s\-\.&]+)', d, re.I)
        if m:
            return _short_vendor(m.group(1).strip())

        # fallback: skip numeric tokens, get meaningful words
        tokens = d.split()
        for token in tokens:
            if re.search(r"[A-Za-z]", token) and not re.match(r'^\d+[/-]\d+$', token):
                idx = tokens.index(token)
                name_tokens = []
                for t in tokens[idx:idx+3]:
                    if re.search(r"[A-Za-z]", t) and t.lower() not in ['id', 'ref', 'code', 'orig']:
                        name_tokens.append(t)
                if name_tokens:
                    return _short_vendor(" ".join(name_tokens))
                return _short_vendor(token)

        return "UNKNOWN"

    # ===================================================================================
    # 2) T A B U L A R   P D F   P A R S E R  (sample_bank_statement)
    # For PDFs like:
    # Date | Description | Debit | Credit | Balance
    # ===================================================================================
    def _parse_tabular(self, lines: List[str]) -> List[Transaction]:
        txs = []

        expense_keywords = [
            "purchase", "fee", "pos", "shop", "store", "withdraw", 
            "atm", "payment", "transfer out", "charge", "debit"
        ]
        income_keywords = ["payout", "deposit", "salary", "refund", "credit"]

        for ln in lines:
            low = ln.lower()

            # Ignore balance rows
            if any(w in low for w in self.IGNORE_WORDS):
                continue

            # DATE
            d = self.DATE_RE.search(ln)
            if not d:
                continue
            date_raw = d.group(1).replace(",", "")
            date_norm = _normalize_date_token(date_raw)

            # Amounts (at least 2 numbers needed)
            nums = re.findall(r'([+-]?\(?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\)?)', ln)
            nums = [n for n in nums if re.search(r'\d', n)]
            if len(nums) < 2:
                continue

            balance_raw = nums[-1]
            amount_raw = nums[-2]
            amount_val = _clean_amount_token(amount_raw)
            if amount_val is None:
                continue

            # DESCRIPTION (extract BEFORE using desc_low!)
            desc = (
                ln.replace(date_raw, "")
                .replace(amount_raw, "")
                .replace(balance_raw, "")
                .strip()
            )
            desc_low = desc.lower()

            # SIGN DETECTION
            if "(" in amount_raw or "-" in amount_raw:
                signed = -abs(amount_val)
            else:
                signed = abs(amount_val)

            # OVERRIDE SIGN BY DESCRIPTION
            if any(k in desc_low for k in expense_keywords):
                signed = -abs(amount_val)
            elif any(k in desc_low for k in income_keywords):
                signed = abs(amount_val)

            ttype = "deposit" if signed > 0 else "withdrawal"

            # Vendor extraction
            vendor = self._extract_vendor(desc)

            txs.append(Transaction(
                date=date_norm,
                transaction_type=ttype,
                vendor=vendor,
                amount=signed,
                description=_clean_description(desc, vendor, ttype),
                raw_line=ln
            ))

        return txs
    # ===================================================================================
    # 3) S I M P L E   O N E - L I N E   P A R S E R  (sample_financial_statement)
    # ===================================================================================
    def _parse_simple(self, lines: List[str]) -> List[Transaction]:
        txs = []
        for ln in lines:
            low = ln.lower()

            if any(w in low for w in self.IGNORE_WORDS):
                continue

            # DATE
            d = self.DATE_RE.search(ln)
            if not d:
                continue
            date_raw = d.group(1).replace(",", "")
            date_norm = _normalize_date_token(date_raw)

            # AMOUNT
            m = self.AMOUNT_RE.findall(ln)
            if not m:
                continue

            # ALWAYS ignore last number if more than one (balance)
            if len(m) >= 2:
                amount_raw = m[-2]  # second-last = TRUE amount
            else:
                amount_raw = m[-1]
            # last number = transaction
            amount = _clean_amount_token(amount_raw)
            if amount is None:
                continue

            # Type from sign
            ttype = "deposit" if amount > 0 else "withdrawal"

            # Description
            desc = ln.replace(date_raw, "").replace(amount_raw, "").strip()

            # Extract clean vendor name (skip numbers, get first meaningful words)
            vendor_words = [w for w in desc.split() if w.isalpha() and len(w) > 1]
            vendor = " ".join(vendor_words[:3]).title() if vendor_words else "UNKNOWN"
            vendor = _short_vendor(vendor)

            txs.append(Transaction(
                date=date_norm,
                transaction_type=ttype,
                vendor=vendor,
                amount=amount,
                description=_clean_description(desc, vendor, ttype),
                raw_line=ln
            ))

        return txs
class CreditCardParser:
    DATE_RE = re.compile(r'\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b')
    AMOUNT_RE = re.compile(r'\(?\$?\d{1,3}(?:,\d{3})*\.\d{2}\)?')

    def parse(self, lines: List[str]) -> List[Transaction]:
        txs = []

        for ln in lines:
            if not self.DATE_RE.search(ln):
                continue

            amts = self.AMOUNT_RE.findall(ln)
            if not amts:
                continue

            date_raw = self.DATE_RE.search(ln).group()
            date_norm = _normalize_date_token(date_raw)

            amt = _clean_amount_token(amts[-1])
            if amt is None:
                continue

            # 💳 credit card charges are ALWAYS withdrawals
            amt = -abs(amt)

            vendor = _short_vendor(
                re.sub(self.DATE_RE, '', ln)
                .replace(amts[-1], '')
                .strip()
            )

            txs.append(Transaction(
                date=date_norm,
                transaction_type="withdrawal",
                vendor=vendor,
                amount=amt,
                description=_clean_description(ln, vendor, "withdrawal"),
                raw_line=ln,
                section="CREDIT_CARD",
                source="CREDIT_CARD"
            ))

        return txs

# ----------------------------
# LLM enhancer (disabled)
# ----------------------------
class LLMEnhancer:
    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 1200):
        self.model = model
        self.max_tokens = max_tokens

    def enhance(self, transactions: List[Transaction], raw_text: str) -> List[Transaction]:
        # LLM enhancement disabled - not needed
        return transactions




class RuleEngineCategorizer:
    def __init__(self, rules_path="rules.json"):
        with open(rules_path, "r") as f:
            self.rules = sorted(json.load(f), key=lambda x: x["priority"])

    def apply(self, transactions: List[Transaction]) -> List[Transaction]:
        for tx in transactions:
            if tx.amount > 0:
                tx.category = "Income"
                continue

            desc = (tx.description or "").upper()
            vendor = (tx.vendor or "").upper()

            matched = False
            for rule in self.rules:
                if rule["direction"] != "out":
                    continue
                if any(k in vendor or k in desc for k in rule["merchant_contains"]):
                    tx.category = rule["category"]
                    matched = True
                    break

            if not matched:
                tx.category = "Uncategorized"

        return transactions

# ----------------------------
# Categorizer & dedupe
# ----------------------------
class ScheduleCMapper:
    def __init__(self):
        self.rules = [
            ("EXCLUDE", None, EXCLUDE_KEYWORDS),
            ("Line 1", "GROSS", LINE_1_GROSS),
            ("Line 8", "ADVERTISING", LINE_8_ADVERTISING),
            ("Line 9", "VEHICLE", LINE_9_VEHICLE),
            ("Line 11", "CONTRACT", LINE_11_CONTRACT),
            ("Line 17", "LEGAL", LINE_17_LEGAL),
            ("Line 18", "OFFICE", LINE_18_OFFICE),
            ("Line 21", "REPAIRS", LINE_21_REPAIRS),
            ("Line 22", "SUPPLIES", LINE_22_SUPPLIES),
            ("Line 23", "TAXES", LINE_23_TAXES),
            ("Line 24a", "TRAVEL", LINE_24A_TRAVEL),
            ("Line 24b", "MEALS", LINE_24B_MEALS),
            ("Line 25", "UTILITIES", LINE_25_UTILITIES),
        ]

    def map_transactions(self, transactions):
        rows = []

        for tx in transactions:
            desc = (tx.description or "").upper()

            # income
            if tx.amount > 0:
                rows.append(("Line 1", "GROSS", "Gross receipts or sales", tx.amount))
                continue

            matched = False

            for line, code, keywords in self.rules:
                if any(k in desc for k in keywords):
                    if line == "EXCLUDE":
                        matched = True
                        break
                    rows.append((line, code, self._desc(line), abs(tx.amount)))
                    matched = True
                    break

            if not matched:
                rows.append(("Line 27a", "UNMAPPED", "Other expenses", abs(tx.amount)))

        df = pd.DataFrame(rows, columns=[
            "schedule_c_line",
            "tax_line_code",
            "tax_line_description",
            "amount"
        ])

        return df.groupby(
            ["schedule_c_line", "tax_line_code", "tax_line_description"],
            as_index=False
        ).agg(
            raw_total_amount=("amount", "sum"),
            deductible_amount=("amount", "sum"),
            count=("amount", "count")
        )

    def _desc(self, line):
        return {
            "Line 8": "Advertising",
            "Line 9": "Car and truck expenses",
            "Line 11": "Contract labor",
            "Line 17": "Legal and professional services",
            "Line 18": "Office expense",
            "Line 21": "Repairs and maintenance",
            "Line 22": "Supplies",
            "Line 23": "Taxes and licenses",
            "Line 24a": "Travel",
            "Line 24b": "Meals",
            "Line 25": "Utilities",
        }.get(line, "Other expenses")

    
# ----------------------------
# Report generator
# ----------------------------
class ReportGenerator:
    def generate_summary_statistics(self, transactions: List[Transaction]) -> Dict[str, Any]:
        total_deposits = sum(t.amount for t in transactions if t.amount > 0)
        total_withdrawals = sum(-t.amount for t in transactions if t.amount < 0)
        return {
            'Total Deposit Amount': float(total_deposits),
            'Total Withdrawal Amount': float(total_withdrawals),
            'Total Deposits': int(sum(1 for t in transactions if t.amount > 0)),
            'Total Withdrawals': int(sum(1 for t in transactions if t.amount < 0)),
            'Total Transactions': len(transactions),
            'Net Income': float(total_deposits - total_withdrawals),
            'Transactions Needing Review': int(sum(1 for t in transactions if t.needs_review))
        }

    def generate_deposits_summary(self, transactions: List[Transaction]) -> pd.DataFrame:
        deps = [t for t in transactions if t.amount > 0]
        if not deps:
            return pd.DataFrame()
        df = pd.DataFrame([asdict(t) for t in deps])
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
        grp = df.groupby('vendor').agg({'amount': 'sum', 'raw_line': 'count'}).reset_index()
        grp.columns = ['Source/Vendor', 'Subtotal ($)', 'Transaction Count']
        grp['Subtotal ($)'] = grp['Subtotal ($)'].astype(float)
        total = grp['Subtotal ($)'].sum()
        total_row = pd.DataFrame([{'Source/Vendor': 'TOTAL DEPOSITS', 'Subtotal ($)': total, 'Transaction Count': grp['Transaction Count'].sum()}])
        out = pd.concat([grp, total_row], ignore_index=True)
        return out[['Source/Vendor', 'Transaction Count', 'Subtotal ($)']]

    def generate_withdrawals_summary(self, transactions: List[Transaction]) -> pd.DataFrame:
        wds = [t for t in transactions if t.amount < 0]
        if not wds:
            return pd.DataFrame()
        df = pd.DataFrame([asdict(t) for t in wds])
        df['amount'] = df['amount'].abs()
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
        grp = df.groupby('vendor').agg({'amount': 'sum', 'raw_line': 'count'}).reset_index()
        grp.columns = ['Vendor', 'Subtotal ($)', 'Transaction Count']
        total = grp['Subtotal ($)'].sum()
        total_row = pd.DataFrame([{'Vendor': 'TOTAL WITHDRAWALS', 'Subtotal ($)': total, 'Transaction Count': grp['Transaction Count'].sum()}])
        out = pd.concat([grp, total_row], ignore_index=True)
        return out[['Vendor', 'Transaction Count', 'Subtotal ($)']]

    def generate_pl_report(self, transactions: List[Transaction]) -> pd.DataFrame:
        s = self.generate_summary_statistics(transactions)
        total_income = s['Total Deposit Amount']
        total_expenses = s['Total Withdrawal Amount']
        net = s['Net Income']
        return pd.DataFrame([
            {'Category': 'Total Income', 'Amount ($)': total_income},
            {'Category': 'Total Expenses', 'Amount (``$)': -total_expenses},
            {'Category': 'NET INCOME', 'Amount ($)': net}
        ])

def generate_pl_report_with_account_codes(self, categorized_transactions: List[tuple]) -> pd.DataFrame:
    # categorized_transactions = List of (Transaction, Category)
    data = []

    total_income = 0
    total_expenses = 0

    for tx, cat in categorized_transactions:
        amount = tx.amount
        if cat.name.lower() in ['income', 'sales', 'other income']:  # adjust per your categories
            total_income += amount
        else:
            total_expenses += amount
        
        data.append({
            'Account Code': getattr(cat, 'account_code', ''),
            'Category': cat.name,
            'Amount ($)': amount
        })

    # Add totals
    net_income = total_income - total_expenses
    data.append({'Account Code': '', 'Category': 'Total Income', 'Amount ($)': total_income})
    data.append({'Account Code': '', 'Category': 'Total Expenses', 'Amount ($)': -total_expenses})
    data.append({'Account Code': '', 'Category': 'NET INCOME', 'Amount ($)': net_income})

    return pd.DataFrame(data)

# ----------------------------
# Custom Rules Management
# ----------------------------
def reapply_custom_rules():
    import json
    from pathlib import Path
    from datetime import datetime
    from schedule_c_categorizer import ScheduleCCategorizer
    from account_code_mapper import AccountCodeMapper
    import streamlit as st

    # Safety checks
    if "user" not in st.session_state or "active_business" not in st.session_state:
        # nothing to apply yet
        return

    user_id = st.session_state.user["id"]
    business_name = st.session_state.active_business

    # Load business rules using your helper (falls back to empty list)
    try:
        business_rules = load_business_rules(user_id, business_name) or []
    except Exception as e:
        st.error(f"Error loading business rules: {e}")
        business_rules = []

    # normalize helper
    def _norm(s: str) -> str:
        return " ".join((s or "").lower().strip().split())

    # store in session
    st.session_state.custom_rules = business_rules

    # Apply rules to both transactions and filtered_transactions (if present)
    for key in ("transactions", "filtered_transactions"):
        if key not in st.session_state:
            continue
        for tx in st.session_state[key]:
            text = f"{tx.vendor or ''} {tx.description or ''}"
            text = _norm(text)

            matched = False
            for rule in business_rules:
                kw = _norm(rule.get("keyword", ""))
                if kw and kw in text:
                    # Apply the rule: always write account_code and mark mapped-by-rule
                    tx.account_code = rule.get("account_code")
                    tx._mapped_by_rule = True
                    matched = True
                    break

            # if previously mapped by a rule but now no rule matches -> clear so mapper/fallback can remap
            if not matched and getattr(tx, "_mapped_by_rule", False):
                if hasattr(tx, "account_code"):
                    try:
                        delattr(tx, "account_code")
                    except Exception:
                        # fallback: set to None
                        tx.account_code = None
                tx._mapped_by_rule = False

    # Ensure mapper exists
    if "mapper" not in st.session_state:
        st.session_state.mapper = AccountCodeMapper()
    mapper = st.session_state.mapper

    # Recategorize using ScheduleCCategorizer (so cats reflect up-to-date tx.account_code)
    sc = ScheduleCCategorizer()
    transactions_to_use = st.session_state.get("filtered_transactions", st.session_state.get("transactions", []))
    categorized_transactions = sc.categorize_transactions(transactions_to_use)

    # Assign account codes into category objects:
    for tx, cat in categorized_transactions:
        # If tx has an explicit code (custom rule or manual), preserve it
        if getattr(tx, "account_code", None):
            cat.account_code = tx.account_code
        else:
            # ask mapper (pass business rules as custom_rules to the mapper call)
            code, _ = mapper.get_account_code(
                vendor=getattr(tx, "vendor", None),
                description=getattr(tx, "description", None),
                is_income=(tx.amount >= 0),
                transaction_type=getattr(tx, "transaction_type", None),
                custom_rules=st.session_state.get("custom_rules", [])
            )
            tx.account_code = code
            cat.account_code = code

    # Save back to session
    st.session_state.categorized_transactions = categorized_transactions
    st.session_state.schedule_c_df = sc.generate_schedule_c_dataframe(categorized_transactions)

    # Regenerate the P&L statement text for display/download
    active_categorized = [
        (tx, cat) for tx, cat in categorized_transactions
        if not (getattr(cat, "is_excluded", False) or getattr(tx, "is_excluded", False))
    ]

    st.session_state.pl_statement_text = sc.generate_pl_report_with_account_codes(
        active_categorized,
        business_name=st.session_state.get("active_business", "") or "",
        period=datetime.now().strftime("%B %Y")
    )


import uuid
from pathlib import Path
import os
DATA_DIR = Path(__file__).parent / "data"
RULES_DIR = DATA_DIR / "rules"
USERS_FILE = DATA_DIR / "users.json"

DATA_DIR.mkdir(exist_ok=True)
RULES_DIR.mkdir(exist_ok=True)
BUSINESS_RULES_FILE = DATA_DIR / "business_rules.json"

def load_business_store():
    if BUSINESS_RULES_FILE.exists():
        with open(BUSINESS_RULES_FILE, "r") as f:
            return json.load(f)
    return {}

def save_business_store(data):
    with open(BUSINESS_RULES_FILE, "w") as f:
        json.dump(data, f, indent=2)

def load_users():
    if not os.path.exists(USERS_FILE):
        return []

    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("users", [])
    except json.JSONDecodeError:
        # file exists but is empty or corrupted
        return []

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump({"users": users}, f, indent=2)
def signup_user(name, email, password):
    users = load_users()

    if any(u["email"] == email for u in users):
        return None, "Email already exists"

    user_id = str(uuid.uuid4())

    users.append({
        "id": user_id,
        "name": name,
        "email": email,
        "password": password
    })

    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump({"users": users}, f, indent=2)

    return user_id, None

def login_user(email, password):
    users = load_users()
    for u in users:
        if u["email"] == email and u["password"] == password:
            return u
    return None

def load_business_rules(user_id, business_name):
    store = load_business_store()
    return store.get(str(user_id), {}).get(business_name, {}).get("rules", [])

def save_business_rules(user_id, business_name, rules):
    store = load_business_store()
    store.setdefault(str(user_id), {})
    store[str(user_id)].setdefault(business_name, {})
    store[str(user_id)][business_name]["rules"] = rules
    save_business_store(store)

    store.setdefault(user_id, {})
    store[user_id].setdefault(business_name, {})
    store[user_id][business_name]["rules"] = rules

    save_business_store(store)

def load_user_businesses(user_id):
    store = load_business_store()
    return list(store.get(user_id, {}).keys())

def create_business(user_id, business_name):
    store = load_business_store()
    store.setdefault(user_id, {})
    store[user_id].setdefault(business_name, {"rules": []})
    save_business_store(store)

def delete_business(user_id, business_name):
    store = load_business_store()
    if user_id in store and business_name in store[user_id]:
        del store[user_id][business_name]
        save_business_store(store)
        return True
    return False

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Bank Statement Analyzer (Hybrid)", layout="wide")

if "user" not in st.session_state:
    # Custom CSS for clean centered login/signup page
    st.markdown("""
    <style>
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Main container */
        .stApp {
            background: #f5f7fa;
        }
        
        /* Center content */
        .block-container {
            max-width: 480px;
            padding-top: 5rem;
            padding-bottom: 5rem;
        }
        
        /* Input fields */
        .stTextInput > div > div > input {
            border-radius: 6px;
            border: 1px solid #e1e4e8;
            padding: 10px 12px;
            font-size: 14px;
            background-color: #fafbfc;
            color: #000000;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #0366d6;
            background-color: white;
            outline: none;
            color: #000000;
        }
        
        .stTextInput label {
            font-weight: 500;
            color: #24292e;
            font-size: 14px;
            margin-bottom: 6px;
        }
        
        /* Button */
        .stButton > button {
            width: 100%;
            background-color: #2ea44f;
            color: white;
            border: none;
            padding: 10px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            margin-top: 16px;
            cursor: pointer;
        }
        
        .stButton > button:hover {
            background-color: #2c974b;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: transparent;
            border-bottom: 1px solid #e1e4e8;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            background-color: transparent;
            border: none;
            color: #586069;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            color: #24292e;
            border-bottom: 2px solid #0366d6;
            background-color: transparent;
        }
        
        /* Messages */
        .stSuccess, .stError {
            padding: 12px;
            border-radius: 6px;
            font-size: 14px;
            margin-top: 16px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Login container
    st.markdown("""
    <div style='background: white; padding: 32px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.06); margin-bottom: 20px;'>
        <div style='text-align: center; margin-bottom: 24px;'>
            <div style='font-size: 48px; margin-bottom: 16px;'>💼</div>
            <h2 style='color: #24292e; margin: 0 0 8px 0; font-size: 24px; font-weight: 600;'>Bank Statement Analyzer</h2>
            <p style='color: #586069; margin: 0; font-size: 14px;'>Sign in to access your account</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Form container
    st.markdown("<div style='background: white; padding: 32px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.06);'>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        st.markdown("<div style='padding-top: 16px;'>", unsafe_allow_html=True)
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            user = login_user(email, password)
            if user:
                st.session_state.user = user
                st.session_state.user_id = user["id"]
                st.session_state.custom_rules = []
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div style='padding-top: 16px;'>", unsafe_allow_html=True)
        name = st.text_input("Full Name")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_pwd")
        if st.button("Sign Up"):
            user_id, err = signup_user(name, email, password)
            if err:
                st.error(err)
            else:
                st.success("Account created. Please login.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    st.stop()

# Profile Selection Screen (Netflix-style)
if "active_business" not in st.session_state or st.session_state.active_business is None:
    # Custom CSS for Netflix-style profile selection
    st.markdown("""
    <style>
        .stApp {
            background: #141414;
        }
        
        /* Profile card buttons */
        div[data-testid="column"] .stButton > button {
            background: #333;
            border: 3px solid #555;
            color: #e5e5e5;
            padding: 20px;
            width: 200px;
            height: 180px;
            border-radius: 8px;
            transition: all 0.3s ease;
            font-size: 3.5rem;
            cursor: pointer;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }
        
        div[data-testid="column"] .stButton > button:hover {
            border-color: #e5e5e5;
            background: #444;
            transform: scale(1.05);
        }
        
        /* Text inside buttons */
        div[data-testid="column"] .stButton > button p {
            margin: 0;
            font-size: 1rem;
            color: #e5e5e5;
            white-space: pre-line;
            text-align: center;
        }
        
        /* Delete button styling */
        .delete-profile-btn button {
            background: #e50914 !important;
            border: 1px solid #e50914 !important;
            color: white !important;
            padding: 6px 12px !important;
            width: auto !important;
            height: auto !important;
            font-size: 0.85rem !important;
            border-radius: 4px !important;
            margin-top: 8px !important;
        }
        
        .delete-profile-btn button:hover {
            background: #b20710 !important;
            border-color: #b20710 !important;
        }
        
        /* Logout button styling */
        .logout-btn button {
            background: transparent !important;
            border: 1px solid #555 !important;
            color: #e5e5e5 !important;
            padding: 8px 24px !important;
            width: auto !important;
            height: auto !important;
            font-size: 1rem !important;
        }
        
        .logout-btn button:hover {
            border-color: #e5e5e5 !important;
            background: #333 !important;
        }
        
        .stTextInput > div > div > input {
            background-color: #333;
            border: 1px solid #555;
            color: white;
            border-radius: 4px;
            padding: 10px;
        }
        
        .stTextInput label {
            color: white;
            font-weight: 500;
        }
        
        /* Hide sidebar on profile selection */
        [data-testid="stSidebar"] {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Logout button in top right
    col_logout1, col_logout2 = st.columns([6, 1])
    with col_logout2:
        st.markdown('<div class="logout-btn">', unsafe_allow_html=True)
        if st.button("Logout", key="profile_logout"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<div style='text-align: center; color: white; padding: 40px 0 40px 0;'><h1 style='font-size: 3.5vw; font-weight: 400;'>Who's managing finances?</h1></div>", unsafe_allow_html=True)
    
    user_id = st.session_state.user_id
    businesses = load_user_businesses(user_id)
    
    # Handle delete confirmation
    if st.session_state.get("confirm_delete"):
        business_to_delete = st.session_state.confirm_delete
        st.markdown("<br><br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style='background: #1a1a1a; padding: 30px; border-radius: 8px; text-align: center;'>
                <p style='color: white; font-size: 1.3rem; margin-bottom: 20px;'>⚠️ Delete Profile?</p>
                <p style='color: #808080; font-size: 1rem; margin-bottom: 20px;'>Are you sure you want to delete <strong style='color: white;'>{business_to_delete}</strong>?<br>This action cannot be undone.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("❌ Yes, Delete", key="confirm_delete_yes", use_container_width=True):
                    if delete_business(user_id, business_to_delete):
                        st.session_state.confirm_delete = None
                        st.success(f"Profile '{business_to_delete}' deleted successfully")
                        st.rerun()
            with col_b:
                if st.button("Cancel", key="confirm_delete_no", use_container_width=True):
                    st.session_state.confirm_delete = None
                    st.rerun()
        st.stop()
    
    # Show create form if requested
    if st.session_state.get("show_create_form", False):
        st.markdown("<br><br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.markdown("<div style='background: #1a1a1a; padding: 30px; border-radius: 8px;'>", unsafe_allow_html=True)
            st.markdown("<p style='color: white; text-align: center; font-size: 1.2rem; margin-bottom: 20px;'>Create New Profile</p>", unsafe_allow_html=True)
            new_business = st.text_input("Business/Profile Name", key="new_profile_name")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✅ Create", key="confirm_create", use_container_width=True):
                    if new_business:
                        create_business(user_id, new_business)
                        st.session_state.active_business = new_business
                        st.session_state.show_create_form = False
                        st.rerun()
                    else:
                        st.error("Please enter a name")
            with col_b:
                if st.button("❌ Cancel", key="cancel_create", use_container_width=True):
                    st.session_state.show_create_form = False
                    if not businesses:
                        # If no businesses exist, keep form open
                        st.session_state.show_create_form = True
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Show profile cards
        if businesses:
            # Calculate grid layout - center profiles
            cols_per_row = min(4, len(businesses) + 1)
            profile_icons = ["🏢", "💼", "🏪", "🏭", "🏦", "🎯", "📊", "💰"]
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Create rows of profiles with centering
            all_profiles = businesses + ["__add_profile__"]
            for i in range(0, len(all_profiles), cols_per_row):
                # Add spacing columns for centering
                num_items = min(cols_per_row, len(all_profiles) - i)
                spacing = (cols_per_row - num_items) / 2
                
                if spacing > 0:
                    cols = st.columns([spacing] + [1] * num_items + [spacing])
                    start_col = 1
                else:
                    cols = st.columns(cols_per_row)
                    start_col = 0
                
                for j in range(num_items):
                    profile_item = all_profiles[i + j]
                    
                    if profile_item == "__add_profile__":
                        # Add Profile button
                        with cols[start_col + j]:
                            if st.button("➕\n\nAdd Profile", key="create_new_profile", help="Add Profile"):
                                st.session_state.show_create_form = True
                                st.rerun()
                    else:
                        # Existing business profile
                        business = profile_item
                        icon = profile_icons[(i + j) % len(profile_icons)]
                        
                        with cols[start_col + j]:
                            if st.button(f"{icon}\n\n{business}", key=f"select_{business}", help=business):
                                st.session_state.active_business = business
                                st.rerun()
                            
                            # Delete button for this profile
                            st.markdown('<div class="delete-profile-btn">', unsafe_allow_html=True)
                            if st.button("🗑️ Delete", key=f"delete_{business}"):
                                st.session_state.confirm_delete = business
                                st.rerun()
                            st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
        else:
            # No profiles yet - show create button
            st.markdown("<br><br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                if st.button("➕\n\nCreate Your First Profile", key="first_profile", help="Create Your First Profile"):
                    st.session_state.show_create_form = True
                    st.rerun()
    
    st.stop()

st.markdown("<h3 style='text-align: center;'>Prototype v1.0</h3>", unsafe_allow_html=True)

st.title("💼 Bank Statement Analyzer")

user_id = st.session_state.user_id

businesses = load_user_businesses(user_id)

# Add sidebar with settings and profile switcher
with st.sidebar:
    st.header("Settings")
    
    # Show active profile
    if st.session_state.active_business:
        st.info(f"📊 **{st.session_state.active_business}**")
    
    if st.button("🔄 Switch Profile"):
        st.session_state.active_business = None
        st.rerun()
    
    sort_by = st.selectbox("Sort vendor summaries by", ["Subtotal (desc)", "Transaction Count (desc)"])
    
    st.divider()
    st.write(f"👤 {st.session_state.user['name']}")
    if st.button("Logout"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
if "rules_loaded_for_business" not in st.session_state:
    st.session_state.rules_loaded_for_business = None

if st.session_state.active_business:
    if st.session_state.rules_loaded_for_business != st.session_state.active_business:
        st.session_state.custom_rules = load_business_rules(
            st.session_state.user["id"],
            st.session_state.active_business
        )
        st.session_state.rules_loaded_for_business = st.session_state.active_business
        reapply_custom_rules()
        
        # after running reapply_custom_rules:
        # show count
        st.markdown(
    "Upload a bank statement (PDF / CSV / DOCX)"
)
def filter_atm_withdrawals(transactions: List[Transaction]) -> List[Transaction]:
    return [
        t for t in transactions
        if t.section == "ATM" and t.transaction_type == "withdrawal"
    ]


def get_active_transactions():
    """
    Date filter + Exclude dono apply karta hai
    """
    txs = st.session_state.get(
        "filtered_transactions",
        st.session_state.transactions
    )
    return [t for t in txs if not getattr(t, "is_excluded", False)]

def is_tx_excluded(tx):
    return getattr(tx, "is_excluded", False)
uploaded = st.file_uploader("Upload statement (PDF, CSV, DOCX)", type=["pdf", "csv", "doc", "docx"])
credit_card_file = st.file_uploader(
    "Upload Credit Card Statement (PDF)",
    type=["pdf"],
    key="credit_card"
)

# Initialize custom rules in session state (load from file)

if uploaded or credit_card_file:
    if uploaded:
        st.info(f"Bank Statement: {uploaded.name} — {uploaded.size/1024:.1f} KB")
    if credit_card_file:
        st.info(f"Credit Card Statement: {credit_card_file.name} — {credit_card_file.size/1024:.1f} KB")
    
    currency = st.selectbox("Currency", ["PKR", "USD", "EUR", "GBP", "AED", "CAD", "AUD"], index=1)
    
    # Only show opening balance checkbox if bank statement is uploaded
    include_opening_balance = False
    if uploaded:
        include_opening_balance = st.checkbox(
            "Include Opening Balance",
            value=False,
            help="Adds opening balance as a deposit before transactions"
        )

    if st.button("Process Statement"):
        with st.spinner("Parsing & processing..."):
            all_txs = []
            meta = {}
            
            # Process main bank statement if uploaded
            if uploaded:
                file_bytes = uploaded.read()
                dp = DocumentParser()
                lines, ok, unreadable = dp.parse_document(file_bytes, uploaded.name)
                if not ok or len(lines) < 1:
                    st.error("Could not read text from bank statement file.")
                    if unreadable:
                        st.warning(f"Unreadable pages: {unreadable}")
                else:
                    # First try Chase-optimized fallback
                    fallback = FallbackStatementParser(include_opening_balance=include_opening_balance)
                    bank_txs, meta = fallback.parse_statement(lines)
                    all_txs.extend(bank_txs)

            # Process credit card statement if uploaded
            if credit_card_file is not None:
                cc_file_bytes = credit_card_file.read()
                dp_cc = DocumentParser()
                cc_lines, ok, unreadable = dp_cc.parse_document(cc_file_bytes, credit_card_file.name)
                
                if not ok or len(cc_lines) < 1:
                    st.error("Could not read text from credit card statement file.")
                    if unreadable:
                        st.warning(f"Unreadable pages: {unreadable}")
                else:
                    cc_txs = CreditCardParser().parse(cc_lines)

                    # FORCE credit card as withdrawals
                    for tx in cc_txs:
                        tx.transaction_type = "withdrawal"
                        tx.amount = -abs(tx.amount)

                    all_txs.extend(cc_txs)

            valid_dates = [
                _md_key(tx.date)
                for tx in all_txs
                if _md_key(tx.date) is not None
            ]

            # If nothing extracted or too few rows, try UniversalParser conservative fallback
            if not all_txs or len(all_txs) < 3:
                if uploaded:  # Only try universal parser if we have a bank statement
                    up = UniversalParser()
                    u_txs = up.parse(lines)
                    if u_txs:
                        # prefer universal only if it returns something meaningful
                        all_txs = u_txs
                        meta = {"parsed_from": "universal_fallback", "transactions_extracted": len(all_txs)}

            parsed_from = meta.get("parsed_from", "fallback")

            if not all_txs:
                st.error("No transactions extracted.")
                st.stop()
            # categorize & dedupe
            categorizer = RuleEngineCategorizer()
            transactions = categorizer.apply(all_txs)

            # stats & reports
            rg = ReportGenerator()
            stats = rg.generate_summary_statistics(transactions)
            deposits_df = rg.generate_deposits_summary(all_txs)
            withdrawals_df = rg.generate_withdrawals_summary(all_txs)
            pl_df = rg.generate_pl_report(all_txs)
            # Use the comprehensive Schedule C categorizer
            from schedule_c_categorizer import ScheduleCCategorizer
            sc_categorizer = ScheduleCCategorizer()
            categorized_transactions = sc_categorizer.categorize_transactions(all_txs)
            schedule_c_df = sc_categorizer.generate_schedule_c_dataframe(categorized_transactions)
            st.session_state.schedule_c_df = schedule_c_df
            st.session_state.categorized_transactions = categorized_transactions  # Store for detail view

            # Sorting vendor summary based on UI
            if deposits_df is not None and not deposits_df.empty:
                if sort_by == "Subtotal (desc)":
                    deposits_df = deposits_df.sort_values("Subtotal ($)", ascending=False).reset_index(drop=True)
                else:
                    deposits_df = deposits_df.sort_values("Transaction Count", ascending=False).reset_index(drop=True)

            if withdrawals_df is not None and not withdrawals_df.empty:
                if sort_by == "Subtotal (desc)":
                    withdrawals_df = withdrawals_df.sort_values("Subtotal ($)", ascending=False).reset_index(drop=True)
                else:
                    withdrawals_df = withdrawals_df.sort_values("Transaction Count", ascending=False).reset_index(drop=True)

            # store in session
            st.session_state.transactions = all_txs
            st.session_state.stats = stats
            st.session_state.deposit_df = deposits_df
            st.session_state.withdrawal_df = withdrawals_df
            st.session_state.pl_df = pl_df
            st.session_state.currency = currency
            st.session_state.parsed_from = parsed_from
            reapply_custom_rules()

            st.success(f"Processed {len(all_txs)} transactions ({parsed_from}).")
            st.session_state.all_transactions = transactions
            st.session_state.filtered_transactions = transactions 

from datetime import datetime, date

all_transactions = st.session_state.get("all_transactions", [])
filtered = []
def _md_key(d):
    try:
        dt = datetime.strptime(d, "%Y-%m-%d")
        return (dt.month, dt.day)
    except:
        return None

if all_transactions:
    st.subheader("📅 Filter by Date")

    # Extract all valid dates with full date information
    parsed_dates = []
    for tx in all_transactions:
        if tx.date:
            try:
                parsed_dates.append(datetime.strptime(tx.date, "%Y-%m-%d").date())
            except:
                pass

    if not parsed_dates:
        st.warning("No valid dates found in transactions.")
    else:
        # Get actual min and max dates from the statement
        min_date = min(parsed_dates)
        max_date = max(parsed_dates)

        col1, col2 = st.columns(2)

        with col1:
            start_md = st.date_input(
                "Start Date",
                value=min_date,
                key="filter_start_md"
            )

        with col2:
            end_md = st.date_input(
                "End Date",
                value=max_date,
                key="filter_end_md"
            )

        if st.button("Apply Date Filter"):
            

            start_key = (start_md.month, start_md.day)
            end_key = (end_md.month, end_md.day)

            for tx in all_transactions:
                md = _md_key(tx.date)
                if not md:
                    continue

                # handle year wrap (Dec → Jan)
                if start_key <= end_key:
                    if start_key <= md <= end_key:
                        filtered.append(tx)
                else:
                    if md >= start_key or md <= end_key:
                        filtered.append(tx)

            st.session_state.filtered_transactions = filtered

            st.success(
                f"Showing {len(filtered)} transactions "
                f"from {start_md.month}/{start_md.day} "
                f"to {end_md.month}/{end_md.day}"
            )

        if st.button("Reset Date Filter"):
            st.session_state.filtered_transactions = all_transactions
            st.info("Date filter cleared. Showing all transactions.")

        
# Dashboard (same UI as before)
if "transactions" in st.session_state and st.session_state.transactions:
    transactions: List[Transaction] = get_active_transactions()


    rg = ReportGenerator()
    stats = rg.generate_summary_statistics(transactions)
    cur = st.session_state.currency

    st.header("📊 Summary")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Deposits",
        f"{cur} {stats['Total Deposit Amount']:,.2f}",
        f"{stats['Total Deposits']} tx"
    )
    c2.metric(
        "Total Withdrawals",
        f"{cur} {stats['Total Withdrawal Amount']:,.2f}",
        f"{stats['Total Withdrawals']} tx"
    )
    c3.metric(
        "Net Income",
        f"{cur} {stats['Net Income']:,.2f}"
    )
    c4.metric(
        "Transactions",
        stats['Total Transactions']
    )


    computed_deposits = sum(t.amount for t in transactions if t.amount > 0 and not is_tx_excluded(t))
    computed_withdrawals = sum(-t.amount for t in transactions if t.amount < 0 and not is_tx_excluded(t))
    if abs(computed_deposits - stats['Total Deposit Amount']) > 0.001 or abs(computed_withdrawals - stats['Total Withdrawal Amount']) > 0.001:
        st.warning("Reconciliation mismatch: using computed sums as source of truth.")
        stats['Total Deposit Amount'] = float(computed_deposits)
        stats['Total Withdrawal Amount'] = float(computed_withdrawals)
        stats['Net Income'] = float(computed_deposits - computed_withdrawals)

    # Toggle to hide the Schedule C tab from the frontend while keeping
    # all Schedule C backend logic intact.
    SHOW_SCHEDULE_C = False

    base_labels = ["💰 Deposits", "💸 Withdrawals", "📈 P&L", "📋 All Transactions"]
    # Insert Schedule C tab before the final P&L (Account Codes) tab when enabled
    if SHOW_SCHEDULE_C:
        labels = base_labels + ["📄 Schedule C", "📊 P&L (Account Codes)", "⚙️ Custom Rules"]
    else:
        labels = base_labels + ["📊 P&L (Account Codes)", "⚙️ Custom Rules"]

    # Use query params to preserve active tab across reruns
    try:
        query_params = st.query_params
        default_tab = int(query_params.get("tab", 0))
    except:
        default_tab = 0
    
    # Use selectbox instead of tabs for better state control
    selected_tab = st.selectbox(
        "Select View:",
        range(len(labels)),
        format_func=lambda x: labels[x],
        index=default_tab,
        key="active_tab_selector"
    )
    
    # Update query param when tab changes
    st.query_params["tab"] = str(selected_tab)
    
    rg = ReportGenerator()

    if selected_tab == 0:  # Deposits tab
        st.subheader("All Deposits Summary (by Source/Vendor)")
        # Regenerate deposits summary with filtered transactions
        df = rg.generate_deposits_summary(transactions)
        if df is None or df.empty:
            st.info("No deposits found.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.subheader("👉 Vendor Transaction Details")
            deps = [t for t in st.session_state.get(
                "filtered_transactions", st.session_state.transactions
            ) if t.amount > 0]
            grouped = {}
            for t in deps:
                key = t.vendor or "UNKNOWN"
                grouped.setdefault(key, []).append(t)
            for vendor, items in sorted(grouped.items(), key=lambda x:(-len(x[1]), x[0])):
                subtotal = sum(i.amount for i in items)
                cnt = len(items)
                with st.expander(f"{vendor}"):
                    details = pd.DataFrame([{
                        "Date": f"~~{it.date}~~" if is_tx_excluded(it) else it.date,
                        "Amount": f"{cur} {it.amount:,.2f}",
                        "Description": f"~~{it.description}~~" if is_tx_excluded(it) else it.description,
                        "Needs Review": "⚠ Yes" if it.needs_review else "✅ No",
                        "Status": "🚫 Excluded" if is_tx_excluded(it) else "✅ Active"
                    } for it in items])
                    st.dataframe(details, use_container_width=True, hide_index=True)

    elif selected_tab == 1:  # Withdrawals tab
        st.subheader("All Withdrawals Summary (by Vendor)")
        # Regenerate withdrawals summary with filtered transactions
        df = rg.generate_withdrawals_summary(transactions)
        if df is None or df.empty:
            st.info("No withdrawals found.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.subheader("👉 Vendor Transaction Details")
            wds = [t for t in st.session_state.get(
                "filtered_transactions", st.session_state.transactions
            ) if t.amount < 0]
            grouped = {}
            for t in wds:
                key = t.vendor or "UNKNOWN"
                grouped.setdefault(key, []).append(t)
            for vendor, items in sorted(grouped.items(), key=lambda x:(-len(x[1]), x[0])):
                subtotal = sum(abs(i.amount) for i in items)
                cnt = len(items)
                with st.expander(f"{vendor}"):
                    details = pd.DataFrame([{
                    "Date": f"~~{it.date}~~" if is_tx_excluded(it) else it.date,
                    "Amount": f"{cur} {abs(it.amount):,.2f}",
                    "Description": f"~~{it.description}~~" if is_tx_excluded(it) else it.description,
                    "Needs Review": "⚠ Yes" if it.needs_review else "✅ No",
                    "Status": "🚫 Excluded" if is_tx_excluded(it) else "✅ Active"
                } for it in items])
                    st.dataframe(details, use_container_width=True, hide_index=True)

    elif selected_tab == 2:  # P&L tab
        st.subheader("Profit & Loss")
        # Regenerate P&L with filtered transactions
        filtered_pl_df = rg.generate_pl_report(get_active_transactions())
        st.dataframe(filtered_pl_df, use_container_width=True, hide_index=True)

    elif selected_tab == 3:  # All Transactions tab
        st.subheader("All Transactions")
        all_df = pd.DataFrame([{
            "Date": f"~~{t.date}~~" if is_tx_excluded(t) else (t.date or ""),
            "Type": t.transaction_type,
            "Vendor": f"~~{t.vendor}~~" if is_tx_excluded(t) else t.vendor,
            "Amount": f"{cur} {t.amount:,.2f}",
            "Description": f"~~{t.description}~~" if is_tx_excluded(t) else t.description,
            "Status": "🚫 Excluded" if is_tx_excluded(t) else "✅ Active"
        } for t in transactions])
        st.dataframe(all_df, use_container_width=True, hide_index=True)
    
    elif SHOW_SCHEDULE_C and selected_tab == 4:  # Schedule C tab
            st.subheader("📄 Schedule C")

            schedule_c_df = st.session_state.get("schedule_c_df")

            if schedule_c_df is None or schedule_c_df.empty:
                st.info("No Schedule C data available.")
            else:
                categorized_transactions = st.session_state.get("categorized_transactions", [])
                transactions = st.session_state.get("transactions", [])

                from datetime import datetime
                import pandas as pd
                import re

                cur = "USD"

                # ===============================
                # IRS SCHEDULE C VIEW
                # ===============================
                from schedule_c_categorizer import ScheduleCCategorizer
                sc_categorizer = ScheduleCCategorizer()

                # ---- Generate Schedule C report
                schedule_c_text = sc_categorizer.generate_schedule_c_report(categorized_transactions)
                st.subheader("📄 IRS Schedule C Report")
                st.code(schedule_c_text)

                st.subheader("🧾 IRS Schedule C Summary")
                st.dataframe(schedule_c_df, use_container_width=True, hide_index=True)

                if not categorized_transactions:
                    st.stop()

                category_groups = {}

                for tx, cat in categorized_transactions:
                    if cat.is_excluded or not cat.line_number:
                        continue
                    key = (cat.line_number, cat.tax_code, cat.category_name)
                    category_groups.setdefault(key, []).append((tx, cat))

                for (line, code, name), items in sorted(
                    category_groups.items(),
                    key=lambda x: (
                        float(re.search(r'(\d+)', x[0][0]).group(1))
                        if re.search(r'(\d+)', x[0][0]) else 999
                    )
                ):
                    subtotal = sum(abs(tx.amount) for tx, _ in items)
                    count = len(items)

                    label = f"{line} · {name} — {cur} {subtotal:,.2f} ({count} tx)"

                    with st.expander(label):
                        df = pd.DataFrame([{
                            "Date": tx.date or "",
                            "Vendor": tx.vendor or "",
                            "Amount": f"{cur} {abs(tx.amount):,.2f}",
                            "Description": tx.description,
                            "Tax Code": cat.tax_code,
                            "Needs Review": "⚠ Yes" if tx.needs_review else "✅ No"
                        } for tx, cat in items])

                        st.dataframe(df, use_container_width=True, hide_index=True)

    elif selected_tab == (5 if SHOW_SCHEDULE_C else 4):  # P&L Account Codes tab
        st.subheader("📊 Profit & Loss (Account Codes)")
        reapply_custom_rules()
        # Build categorized list that matches filtered transactions
        all_categorized = st.session_state.categorized_transactions
        filtered_transactions = st.session_state.get(
            "filtered_transactions",
            st.session_state.transactions
        )

        # Use all transactions (including excluded ones)
        # We'll handle excluded status in the display logic
        # Build categorized list that matches date-filtered transactions
        filtered_keys = {
            (t.date, t.description, t.amount)
            for t in filtered_transactions
        }

        categorized_transactions = [
            (tx, cat)
            for tx, cat in all_categorized
            if (tx.date, tx.description, tx.amount) in filtered_keys
        ]



        if not categorized_transactions or not transactions:
            st.info("No transaction data available for P&L report.")
        else:
            from datetime import datetime
            import pandas as pd
            from schedule_c_categorizer import ScheduleCCategorizer

            sc_categorizer = ScheduleCCategorizer()
            cur = "USD"

            # ---- Robust period detection (min → max date)
            date_objs = []
            for tx in transactions:
                if tx.date:
                    try:
                        date_objs.append(datetime.strptime(tx.date, "%Y-%m-%d"))
                    except:
                        pass

            if date_objs:
                start = min(date_objs)
                end = max(date_objs)
                default_period = f"{start.strftime('%b %Y')} – {end.strftime('%b %Y')}"
            else:
                default_period = datetime.now().strftime("%B %Y")

            col1, col2 = st.columns(2)
            with col1:
                business_name = st.text_input("Business Name (optional):", key="pl_account_business")
            with col2:
                period_input = st.text_input("Period:", value=default_period, key="pl_account_period")

            # ---- Filter out excluded transactions for P&L statement generation
            from account_code_mapper import AccountCodeMapper

            # Ensure mapper exists
            if "mapper" not in st.session_state:
                st.session_state.mapper = AccountCodeMapper()
            mapper = st.session_state.mapper

            # Build active categorized transactions
            active_categorized_transactions = [
                (tx, cat)
                for tx, cat in categorized_transactions
                if not (cat.is_excluded or is_tx_excluded(tx))
            ]

            synced_categorized = []
            mapper.custom_rules = st.session_state.custom_rules 

            for tx, cat in active_categorized_transactions:
                # ✅ Use existing account_code if present (manual changes)
                if hasattr(tx, "account_code") and tx.account_code:
                    code = tx.account_code
                    # Get name from mapper if available
                    name = dict(mapper.account_code_map).get(code, (code, "UNKNOWN"))[1]
                else:
                    # Auto map for new/unmapped transactions
                    code, name = mapper.get_account_code(
                        vendor=getattr(tx, "vendor", None),
                        description=getattr(tx, "description", None),
                        is_income=(tx.amount >= 0),
                        transaction_type=getattr(tx, "type", None)
                    )
                    # Save mapped code to transaction for persistence
                    tx.account_code = code

                # Assign to category for PL
                cat.account_code = code
                synced_categorized.append((tx, cat))
            
            # Persist for download and PL generation
            st.session_state.synced_categorized = synced_categorized

            # Generate PL report using persisted codes ONLY
            pl_text = sc_categorizer.generate_pl_report_with_account_codes(
                synced_categorized,
                business_name=business_name or "",
                period=period_input
            )
            st.session_state.pl_statement_text = pl_text
            st.code(pl_text)

            # 🔹 Validation & reconciliation (optional)
            validation_result = sc_categorizer.validate_classifications(synced_categorized)
            reconciliation_result = sc_categorizer.reconcile_totals(
                get_active_transactions(),
                synced_categorized
            )
            
            # Display validation warnings
            if validation_result["error_count"] > 0 or validation_result["warning_count"] > 0:
                st.subheader("⚠️ Validation & Reconciliation")
                
                if validation_result["error_count"] > 0:
                    st.error(f"❌ Found {validation_result['error_count']} classification error(s):")
                    for error in validation_result["errors"]:
                        st.error(error["message"])
                
                if validation_result["warning_count"] > 0:
                    st.warning(f"⚠️ Found {validation_result['warning_count']} warning(s):")
                    for warning in validation_result["warnings"]:
                        st.warning(warning["message"])
            
            # Display reconciliation status
            if not reconciliation_result["fully_reconciled"]:
                st.warning("⚠️ Reconciliation Mismatch Detected:")
                if not reconciliation_result["income_reconciled"]:
                    st.warning(f"  Income: Raw deposits ${reconciliation_result['raw_deposits_total']:,.2f} vs Categorized ${reconciliation_result['categorized_income_total']:,.2f} (Diff: ${reconciliation_result['income_difference']:,.2f})")
                if not reconciliation_result["expenses_reconciled"]:
                    st.warning(f"  Expenses: Raw withdrawals ${reconciliation_result['raw_withdrawals_total']:,.2f} vs Categorized ${reconciliation_result['categorized_expenses_total']:,.2f} (Diff: ${reconciliation_result['expenses_difference']:,.2f})")
            else:
                st.success("✅ Reconciliation: All totals match!")
            

            # ---- Build structured P&L dataframe
            
            
            rows = []
            for tx, cat in categorized_transactions:
                # Keep excluded transactions but mark them
                is_excluded_tx = cat.is_excluded or is_tx_excluded(tx)

                # DATA-DRIVEN CLASSIFICATION: Use transaction_type as source of truth
                # Deposits → Income, Withdrawals → Expenses
                is_income = tx.transaction_type == "deposit"
                
                # Check if account_code is already set on transaction (from manual reassignment)
                if hasattr(tx, 'account_code') and tx.account_code:
                    account_code = tx.account_code
                    # Get the account name from JSON
                    import json
                    from pathlib import Path
                    account_file = Path(__file__).parent / 'account_keywords.json'
                    try:
                        with open(account_file, 'r') as f:
                            data = json.load(f)
                            account_name = data.get(account_code, {}).get('name', 'UNKNOWN')
                    except:
                        account_name = 'UNKNOWN'
                else:
                    # Get account code based on transaction type
                    account_code, account_name = mapper.get_account_code(
                        tx.vendor, 
                        tx.description, 
                        is_income=is_income,
                        transaction_type=tx.transaction_type,
                        custom_rules=st.session_state.get('custom_rules', [])
                    )
                    
                    # Ensure account code matches transaction type
                    # If withdrawal but got income code (600s), force to expense code
                    if tx.transaction_type == "withdrawal" and account_code.startswith('6'):
                        # Force to expense code (999 OTHER EXPENSES as fallback)
                        account_code, account_name = mapper.get_account_code(
                            tx.vendor,
                            tx.description,
                            is_income=False,
                            transaction_type="withdrawal",
                            custom_rules=st.session_state.get('custom_rules', [])
                        )
                    
                    # If deposit but got expense code, force to income code
                    if tx.transaction_type == "deposit" and not account_code.startswith('6'):
                        # Force to income code (601 SALES as default)
                        account_code, account_name = ("601", "SALES")

                rows.append({
                    "Account Code": account_code,
                    "Account Name": account_name,
                    "Date": f"~~{tx.date}~~" if is_excluded_tx else (tx.date or ""),
                    "Vendor": f"~~{tx.vendor}~~" if is_excluded_tx else (tx.vendor or ""),
                    "Amount": abs(tx.amount),
                    "Type": "Income" if is_income else "Expense",
                    "Transaction Type": tx.transaction_type.title(),
                    "Needs Review": "⚠ Yes" if tx.needs_review else "✅ No",
                    "Status": "🚫 Excluded" if is_excluded_tx else "✅ Active"
                })

            if rows:
                summary_df = pd.DataFrame(rows)
                
                st.subheader("📋 Transaction Details")
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                # Group by account code
                st.subheader("💼 By Account Code")
                grouped = {}
                for tx, cat in categorized_transactions:
                    # Include all transactions in grouping
                    
                    # Check if account_code is already set on transaction (from manual reassignment)
                    if hasattr(tx, 'account_code') and tx.account_code:
                        account_code = tx.account_code
                        # Get the account name from JSON
                        import json
                        from pathlib import Path
                        account_file = Path(__file__).parent / 'account_keywords.json'
                        try:
                            with open(account_file, 'r') as f:
                                data = json.load(f)
                                account_name = data.get(account_code, {}).get('name', 'UNKNOWN')
                        except:
                            account_name = 'UNKNOWN'
                    else:
                        # DATA-DRIVEN: Use transaction_type as source of truth
                        is_income = tx.transaction_type == "deposit"
                        account_code, account_name = mapper.get_account_code(
                            tx.vendor, 
                            tx.description, 
                            is_income=is_income,
                            transaction_type=tx.transaction_type,
                            custom_rules=st.session_state.get('custom_rules', [])
                        )
                        
                        # Ensure account code matches transaction type
                        if tx.transaction_type == "withdrawal" and account_code.startswith('6'):
                            account_code, account_name = mapper.get_account_code(
                                tx.vendor,
                                tx.description,
                                is_income=False,
                                transaction_type="withdrawal",
                                custom_rules=st.session_state.get('custom_rules', [])
                            )
                        if tx.transaction_type == "deposit" and not account_code.startswith('6'):
                            account_code, account_name = ("601", "SALES")
                    
                    key = (account_code, account_name)
                    grouped.setdefault(key, []).append(tx)

                # Load all account codes for dropdown (do this once outside the loop)
                import json
                from pathlib import Path
                all_account_options = {}
                account_file = Path(__file__).parent / 'account_keywords.json'
                try:
                    with open(account_file, 'r') as f:
                        data = json.load(f)
                        for acc_code, acc_details in data.items():
                            all_account_options[f"{acc_code} · {acc_details['name']}"] = acc_code
                except Exception as e:
                    st.error(f"Error loading account codes: {e}")
                
                for (code, name), txs in sorted(grouped.items()):
                    # Calculate subtotal only for active (non-excluded) transactions
                    active_txs = [t for t in txs if not is_tx_excluded(t)]
                    excluded_txs = [t for t in txs if is_tx_excluded(t)]
                    subtotal = sum(abs(t.amount) for t in active_txs)
                    total_count = len(txs)
                    active_count = len(active_txs)
                    label = f"{code} · {name} — {cur} {subtotal:,.2f} ({active_count}/{total_count} tx)"
                    if excluded_txs:
                        label += f" [{len(excluded_txs)} excluded]"

                    with st.expander(label):
                        st.info("💡 Click 'Change Account Code' to reassign any transaction to a different account")
                        
                        for idx, t in enumerate(txs):
                            col1, col2, col3, col4 = st.columns([2, 2, 1 , 1])
                            with col1:
                                st.text(f"{t.date or 'N/A'} | {t.vendor or 'Unknown'}")
                            with col2:
                                st.text(f"{cur} {abs(t.amount):,.2f} | {t.description[:40] if t.description else 'N/A'}")
                            with col3:
                                # Unique key for each transaction
                                tx_key = f"change_{code}_{idx}_{t.date}_{abs(t.amount)}"
                                if st.button("Change", key=tx_key):
                                    st.session_state[f"editing_{tx_key}"] = True
                            
                            # Show dropdown if editing this transaction
                            if st.session_state.get(f"editing_{tx_key}", False):
                                # Filter out current account code from options
                                available_options = {k: v for k, v in all_account_options.items() if v != code}
                                
                                selected_display = st.selectbox(
                                    "Select new account code:",
                                    options=list(available_options.keys()),
                                    key=f"select_{tx_key}"
                                )
                                
                                col_save, col_cancel = st.columns(2)
                                with col_save:
                                    if st.button("✅ Save", key=f"save_{tx_key}"):
                                        new_code = available_options[selected_display]
                                        for session_tx in st.session_state.transactions:
                                            if (session_tx.date == t.date and 
                                                session_tx.description == t.description and 
                                                session_tx.amount == t.amount):
                                                session_tx.account_code = new_code
                                                # mark manual override so rules won't overwrite
                                                session_tx._mapped_by_rule = False

                                        if "filtered_transactions" in st.session_state:
                                            for session_tx in st.session_state.filtered_transactions:
                                                if (session_tx.date == t.date and 
                                                    session_tx.description == t.description and 
                                                    session_tx.amount == t.amount):
                                                    session_tx.account_code = new_code
                                                    session_tx._mapped_by_rule = False

                                        st.session_state[f"editing_{tx_key}"] = False
                                        st.success(f"✅ Updated to {selected_display}")
                                        st.rerun()

                                with col_cancel:
                                    if st.button("❌ Cancel", key=f"cancel_{tx_key}"):
                                        st.session_state[f"editing_{tx_key}"] = False
                                        st.rerun()
                            with col4:
                                if not is_tx_excluded(t):
                                    if st.button("🚫 Exclude", key=f"exclude_{code}_{idx}_{t.date}_{t.amount}"):
                                        for master_tx in st.session_state.transactions:
                                            if (
                                                master_tx.date == t.date and
                                                master_tx.description == t.description and
                                                master_tx.amount == t.amount
                                            ):
                                                master_tx.is_excluded = True
                                        st.rerun()
                                else:
                                    if st.button("↩ Include", key=f"include_{code}_{idx}_{t.date}_{t.amount}"):
                                        for master_tx in st.session_state.transactions:
                                            if (
                                                master_tx.date == t.date and
                                                master_tx.description == t.description and
                                                master_tx.amount == t.amount
                                            ):
                                                master_tx.is_excluded = False
                                        st.rerun()

                # ---- CSV Export
                csv_data = summary_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download P&L Account Codes (CSV)",
                    csv_data,
                    file_name=f"PL_Account_Codes_{period_input.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
    
    elif selected_tab == (6 if SHOW_SCHEDULE_C else 5):  # Custom Rules tab
        st.subheader("⚙️ Custom Account Code Rules")
        st.markdown("Define custom rules to automatically assign specific vendors or keywords to account codes.")
        
        # Load all account codes for dropdown
        import json
        from pathlib import Path
        all_account_options = {}
        account_file = Path(__file__).parent / 'account_keywords.json'
        try:
            with open(account_file, 'r') as f:
                data = json.load(f)
                for acc_code, acc_details in data.items():
                    all_account_options[f"{acc_code} · {acc_details['name']}"] = acc_code
        except Exception as e:
            st.error(f"Error loading account codes: {e}")
        
        # Add new rule form
        st.subheader("➕ Add New Rule")
        col1, col2, col3 = st.columns([3, 3, 1])
        
        with col1:
            rule_keyword = st.text_input(
                "Keyword/Vendor Name",
                placeholder="e.g., Amazon, Starbucks, Office Supplies",
                key="new_rule_keyword"
            )
        
        with col2:
            rule_account = st.selectbox(
                "Account Code",
                options=list(all_account_options.keys()),
                key="new_rule_account"
            )
        
        with col3:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button("➕ Add", key="add_rule_btn"):
                if rule_keyword.strip():
                    account_code = all_account_options[rule_account]
                    # Check for duplicates
                    existing = [r for r in st.session_state.custom_rules if r['keyword'].lower() == rule_keyword.lower()]
                    if existing:
                        st.warning(f"Rule for '{rule_keyword}' already exists!")
                    else:
                        st.session_state.custom_rules.append({
                            'keyword': rule_keyword.strip(),
                            'account_code': account_code,
                            'account_display': rule_account
                        })
                        save_business_rules(
                            st.session_state.user["id"],
                            st.session_state.active_business,
                            st.session_state.custom_rules
                        )

                        # Reapply rules to existing transactions
                        reapply_custom_rules()
                        st.success(f"✅ Rule added and applied to existing transactions: '{rule_keyword}' → {rule_account}")
                        st.rerun()
                else:
                    st.error("Please enter a keyword/vendor name")
        
        # Display existing rules
        st.subheader("📋 Active Rules")
        if st.session_state.custom_rules:
            st.info(f"Total rules: {len(st.session_state.custom_rules)}")
            
            for idx, rule in enumerate(st.session_state.custom_rules):
                col1, col2, col3 = st.columns([3, 4, 1])
                with col1:
                    st.text(f"🔍 {rule['keyword']}")
                with col2:
                    st.text(f"→ {rule['account_display']}")
                with col3:
                    if st.button("🗑️", key=f"delete_rule_{idx}"):
                        st.session_state.custom_rules.pop(idx)

                        save_business_rules(
                            st.session_state.user["id"],
                            st.session_state.active_business,
                            st.session_state.custom_rules
                        )


                        # Reapply remaining rules to existing transactions
                        reapply_custom_rules()

                        st.success("Rule deleted and transactions updated!")
                        st.rerun()

        else:
            st.info("No custom rules defined yet. Add rules above to automatically categorize transactions.")
        
        st.markdown("---")
        st.markdown("**💡 How it works:**")
        st.markdown("- Custom rules are checked first during transaction categorization")
        st.markdown("- Rules are applied immediately to all current transactions when added or deleted")
        st.markdown("- If a transaction's vendor or description contains your keyword, it will be assigned to your chosen account code")
        st.markdown("- Rules are case-insensitive and match partial text")
        st.markdown("- Rules are saved automatically and persist across sessions")
    
    # Downloads
    st.header("📥 Download")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        dep_csv = st.session_state.deposit_df.to_csv(index=False) if (st.session_state.deposit_df is not None and not st.session_state.deposit_df.empty) else ""
        st.download_button("⬇ Deposits CSV", dep_csv, "deposits.csv", mime="text/csv")
    with c2:
        wd_csv = st.session_state.withdrawal_df.to_csv(index=False) if (st.session_state.withdrawal_df is not None and not st.session_state.withdrawal_df.empty) else ""
        st.download_button("⬇ Withdrawals CSV", wd_csv, "withdrawals.csv", mime="text/csv")
    with c3:
        pnl_csv = st.session_state.pl_df.to_csv(index=False) if (st.session_state.pl_df is not None and not st.session_state.pl_df.empty) else ""
        st.download_button("⬇ P&L CSV", pnl_csv, "pnl.csv", mime="text/csv")
    with c4:
        pl_statement_text = st.session_state.get("pl_statement_text", "")
        st.download_button("⬇ P&L Statement", pl_statement_text, "pl_statement.txt", mime="text/plain")

    st.success("✅ Report generated. Verify totals against your bank statement.")
