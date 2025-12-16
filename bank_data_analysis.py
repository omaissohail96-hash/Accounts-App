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


try:
    import openai
except Exception:
    openai = None

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

# ----------------------------
# Utilities
# ----------------------------
DATE_TOKEN_RE = re.compile(r'(?P<d>\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]{3,9}\s*\d{0,4})')
DATE_AT_START = re.compile(r'^\s*(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b')
AMOUNT_RE = AMOUNT_RE = re.compile(
    r'([+\-]?\(?\s*\$?\d{1,3}(?:[,\s]\d{3})*\.\d{2}\)?)'
)
MULTI_DATE_AMT_RE = re.compile(r'(\d{1,2}[/-]\d{1,2}|[+\-]?\(?\s*\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})\)?)')

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
                dt = dt.replace(year=datetime.now().year)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue
    return t  # fallback: return raw token

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


def _clean_description(desc: str, vendor: str = "", transaction_type: str = "") -> str:
    """
    Rewrite transaction description into a simple, human-readable format.
    Removes technical codes, IDs, timestamps, and trace numbers.
    Returns a clean, natural language description that anyone can understand.
    """
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
    SECTION_PATTERNS = {
        "DEPOSITS": re.compile(r'\bdeposits\s+and\s+additions\b', re.I),
        "CHECKS": re.compile(r'\bchecks\s+paid\b', re.I),
        "ATM": re.compile(r"ATM\s*&\s*DEBIT\s*CARD\s*WITHDRAWALS", re.I),
        "ELECTRONIC_WITHDRAWALS": re.compile(r'\belectronic\s+withdrawals?\b', re.I),
        "FEES": re.compile(r'\bfees?\b', re.I),
    }

    def _is_summary_line(self, ln: str) -> bool:
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
        if "atm" not in ln.lower() and date_count >= 2 and amount_count >= 2:
            return True
        # "TOTAL DEPOSITS" or similar as a whole line
        if re.search(r'\btotal deposits\b|\btotal withdrawals\b|\bdeposits and additions summary\b', low):
            return True
        return False

    def _line_has_vendor_like_text(self, ln: str) -> bool:
        text = ln.lower()

        # ALWAYS accept ATM withdrawals
        if "atm" in text:
            return True

        # Remove date and amounts
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
        # 1. Pre-clean
        cleaned = [ln for ln in (l.strip() for l in lines) if ln and not self._is_summary_line(ln)]

        if not cleaned:
            return [], {"parsed_from": "fallback", "transactions_extracted": 0}

        # 2. segment into blocks starting with a date at start (Chase style)
        blocks: List[Tuple[List[str], str]] = []
        current_section = "UNKNOWN"
        current_block: Optional[List[str]] = None

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
            if DATE_AT_START.match(ln):
                if current_block:
                    blocks.append((current_block, current_section))
                current_block = [ln]
            else:
                # continuation line appended to last block if any
                if current_block is not None:
                    current_block.append(ln)
                else:
                    # stray continuation without start - ignore
                    continue

        if current_block:
            blocks.append((current_block, current_section))

        # 3. parse each block
        for block_lines, section in blocks:
            block_text = " ".join(block_lines)
            # skip if looks like a total or header inside
            if re.search(r'\btotal\b.*\d', block_text, re.I):
                continue
            # get date from first line
            first_line = block_lines[0]
            dmatch = DATE_AT_START.match(first_line)
            date_raw = dmatch.group(1) if dmatch else ""
            date_norm = self._parse_date(date_raw) or ""

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

            elif section in ("CHECKS", "ELECTRONIC_WITHDRAWALS", "FEES"):
                signed_amount = -abs(amt_val)

            else:
                # fallback ONLY if section is unknown
                if amount_raw.strip().startswith("(") or "-" in amount_raw:
                    signed_amount = -abs(amt_val)
                else:
                    signed_amount = abs(amt_val)




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

        meta = {"parsed_from": "chase_sectioned_fallback", "transactions_extracted": len(txs)}
        return txs, meta

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

# ----------------------------
# LLM enhancer (unchanged)
# ----------------------------
class LLMEnhancer:
    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 1200):
        self.model = model
        self.max_tokens = max_tokens

    def enhance(self, transactions: List[Transaction], raw_text: str) -> List[Transaction]:
        if openai is None:
            logger.info("openai package not installed — skipping LLM enhancement.")
            return transactions
        key = None
        try:
            key = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None
        except Exception:
            key = None
        if not key:
            logger.info("No OPENAI_API_KEY found — skipping LLM enhancement.")
            return transactions
        openai.api_key = key
        rows = []
        for i, t in enumerate(transactions[:200]):
            rows.append({
                "idx": i,
                "date": t.date,
                "vendor": t.vendor,
                "amount": t.amount,
                "direction": "in" if t.amount > 0 else "out",
                "description": t.description
            })
        prompt = (
            "You are a precise financial data cleaner. You will receive a JSON array of parsed transactions.\n"
            "Return a JSON array with exactly the same number of elements. Each element must contain:\n"
            " idx (int), date (YYYY-MM-DD or original), vendor (short), amount (number positive), direction ('in'/'out'), description (string)\n"
            "Return ONLY a JSON array (no explanation).\n\nINPUT:\n" + json.dumps(rows, ensure_ascii=False)
        )
        try:
            resp = openai.ChatCompletion.create(model=self.model, temperature=0,
                                                messages=[{"role": "user", "content": prompt}],
                                                max_tokens=self.max_tokens)
            content = resp.choices[0].message["content"]
            parsed = json.loads(content)
            enhanced = []
            for obj in parsed:
                idx = int(obj.get("idx"))
                amt = float(obj.get("amount", 0.0))
                direction = obj.get("direction", "in")
                amt_signed = abs(amt) if direction == "in" else -abs(amt)
                date_out = obj.get("date") or transactions[idx].date
                vendor_out = obj.get("vendor") or transactions[idx].vendor
                desc_out = obj.get("description") or transactions[idx].description
                enhanced.append(Transaction(
                    date=date_out,
                    transaction_type='deposit' if amt_signed > 0 else 'withdrawal',
                    vendor=str(vendor_out).title() if vendor_out else "UNKNOWN",
                    amount=amt_signed,
                    description=str(desc_out),
                    raw_line=transactions[idx].raw_line,
                    needs_review=False
                ))
            if len(enhanced) != len(rows):
                logger.warning("LLM returned different count — skipping enhancement.")
                return transactions
            final = enhanced + transactions[200:]
            return final
        except Exception as e:
            logger.exception("LLM enhancement error: %s", e)
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
            {'Category': 'Total Expenses', 'Amount ($)': -total_expenses},
            {'Category': 'NET INCOME', 'Amount ($)': net}
        ])

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Bank Statement Analyzer (Hybrid)", layout="wide")
st.title("💼 Bank Statement Analyzer — Improved Parser (Chase-first, Universal fallback)")

st.markdown(
    "Upload a bank statement (PDF / CSV / DOCX). The app uses a robust deterministic parser optimized for Chase-style "
    "statements (including ATM & Daily Ending Balance protections). If that fails, a conservative universal parser attempts extraction."
)
def filter_atm_withdrawals(transactions: List[Transaction]) -> List[Transaction]:
    return [
        t for t in transactions
        if t.section == "ATM" and t.transaction_type == "withdrawal"
    ]



with st.sidebar:
    st.header("Settings")
    sort_by = st.selectbox("Sort vendor summaries by", ["Subtotal (desc)", "Transaction Count (desc)"])

uploaded = st.file_uploader("Upload statement (PDF, CSV, DOCX)", type=["pdf", "csv", "doc", "docx"])

if uploaded:
    st.info(f"File: {uploaded.name} — {uploaded.size/1024:.1f} KB")
    currency = st.selectbox("Currency", ["PKR", "USD", "EUR", "GBP", "AED", "CAD", "AUD"], index=1)

    if st.button("Process Statement"):
        with st.spinner("Parsing & processing..."):
            file_bytes = uploaded.read()
            dp = DocumentParser()
            lines, ok, unreadable = dp.parse_document(file_bytes, uploaded.name)
            if not ok or len(lines) < 1:
                st.error("Could not read text from file.")
                if unreadable:
                    st.warning(f"Unreadable pages: {unreadable}")
                st.stop()

            # First try Chase-optimized fallback
            fallback = FallbackStatementParser()
            transactions, meta = fallback.parse_statement(lines)

            # If nothing extracted or too few rows, try UniversalParser conservative fallback
            if not transactions or len(transactions) < 3:
                up = UniversalParser()
                u_txs = up.parse(lines)
                if u_txs:
                    # prefer universal only if it returns something meaningful
                    transactions = u_txs
                    meta = {"parsed_from": "universal_fallback", "transactions_extracted": len(transactions)}

            parsed_from = meta.get("parsed_from", "fallback")

            if not transactions:
                st.error("No transactions extracted.")
                st.stop()
            # categorize & dedupe
            categorizer = RuleEngineCategorizer()
            transactions = categorizer.apply(transactions)

            # stats & reports
            rg = ReportGenerator()
            stats = rg.generate_summary_statistics(transactions)
            deposits_df = rg.generate_deposits_summary(transactions)
            withdrawals_df = rg.generate_withdrawals_summary(transactions)
            pl_df = rg.generate_pl_report(transactions)
            sc_mapper = ScheduleCMapper()
            schedule_c_df = sc_mapper.map_transactions(transactions)
            st.session_state.schedule_c_df = schedule_c_df

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
            st.session_state.transactions = transactions
            st.session_state.stats = stats
            st.session_state.deposit_df = deposits_df
            st.session_state.withdrawal_df = withdrawals_df
            st.session_state.pl_df = pl_df
            st.session_state.currency = currency
            st.session_state.parsed_from = parsed_from

            st.success(f"Processed {len(transactions)} transactions ({parsed_from}).")

# Dashboard (same UI as before)
if "transactions" in st.session_state and st.session_state.transactions:
    transactions: List[Transaction] = st.session_state.transactions
    stats = st.session_state.stats
    cur = st.session_state.currency

    st.header("📊 Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Deposits", f"{cur} {stats['Total Deposit Amount']:,.2f}", f"{stats['Total Deposits']} tx")
    c2.metric("Total Withdrawals", f"{cur} {stats['Total Withdrawal Amount']:,.2f}", f"{stats['Total Withdrawals']} tx")
    c3.metric("Net Income", f"{cur} {stats['Net Income']:,.2f}")
    c4.metric("Transactions", stats['Total Transactions'])

    computed_deposits = sum(t.amount for t in transactions if t.amount > 0)
    computed_withdrawals = sum(-t.amount for t in transactions if t.amount < 0)
    if abs(computed_deposits - stats['Total Deposit Amount']) > 0.001 or abs(computed_withdrawals - stats['Total Withdrawal Amount']) > 0.001:
        st.warning("Reconciliation mismatch: using computed sums as source of truth.")
        stats['Total Deposit Amount'] = float(computed_deposits)
        stats['Total Withdrawal Amount'] = float(computed_withdrawals)
        stats['Net Income'] = float(computed_deposits - computed_withdrawals)

    tab1, tab2, tab3, tab4 , tab5 , tab6 = st.tabs(["💰 Deposits","💸 Withdrawals","📈 P&L","📋 All Transactions" , "📄 Schedule C" , "🏧 ATM Withdrawals"])
    rg = ReportGenerator()

    with tab1:
        st.subheader("Deposits Summary (by Source/Vendor)")
        df = st.session_state.deposit_df
        if df is None or df.empty:
            st.info("No deposits found.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
            deps = [t for t in transactions if t.amount > 0]
            grouped = {}
            for t in deps:
                key = t.vendor or "UNKNOWN"
                grouped.setdefault(key, []).append(t)
            for vendor, items in sorted(grouped.items(), key=lambda x:(-len(x[1]), x[0])):
                subtotal = sum(i.amount for i in items)
                cnt = len(items)
                with st.expander(f"{vendor}"):
                    details = pd.DataFrame([{
                        "Date": it.date or "",
                        "Amount": f"{cur} {it.amount:,.2f}",
                        "Description": it.description,
                        "Needs Review": "⚠ Yes" if it.needs_review else "✅ No"
                    } for it in items])
                    st.dataframe(details, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Withdrawals Summary (by Vendor)")
        df = st.session_state.withdrawal_df
        if df is None or df.empty:
            st.info("No withdrawals found.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
            wds = [t for t in transactions if t.amount < 0]
            grouped = {}
            for t in wds:
                key = t.vendor or "UNKNOWN"
                grouped.setdefault(key, []).append(t)
            for vendor, items in sorted(grouped.items(), key=lambda x:(-len(x[1]), x[0])):
                subtotal = sum(abs(i.amount) for i in items)
                cnt = len(items)
                with st.expander(f"{vendor}"):
                    details = pd.DataFrame([{
                        "Date": it.date or "",
                        "Amount": f"{cur} {abs(it.amount):,.2f}",
                        "Description": it.description,
                        "Needs Review": "⚠ Yes" if it.needs_review else "✅ No"
                    } for it in items])
                    st.dataframe(details, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Profit & Loss")
        st.dataframe(st.session_state.pl_df, use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("All Transactions")
        all_df = pd.DataFrame([{
            "Date": t.date or "",
            "Type": t.transaction_type,
            "Vendor": t.vendor,
            "Amount": f"{cur} {t.amount:,.2f}",
            "Description": t.description
        } for t in transactions])
        st.dataframe(all_df, use_container_width=True, hide_index=True)
    with tab5:
        st.subheader("📄 Schedule C (IRS View)")
        st.dataframe(
            st.session_state.schedule_c_df,
            use_container_width=True,
            hide_index=True
        )
    with tab6:
        st.subheader("ATM Withdrawals")

        atm_txs = filter_atm_withdrawals(transactions)
        

        if not atm_txs:
            st.info("No ATM withdrawals found.")
        else:
            # Summary table
            atm_df = pd.DataFrame([{
                "Date": t.date or "",
                "Vendor": t.vendor,
                "Amount": abs(t.amount),
                "Description": t.description,
                "Needs Review": t.needs_review
            } for t in atm_txs])

            total_atm = sum(abs(t.amount) for t in atm_txs)

            st.metric("Total ATM Withdrawals", f"{cur} {total_atm:,.2f}", f"{len(atm_txs)} tx")

            st.dataframe(
                atm_df.assign(
                    Amount=lambda x: x["Amount"].map(lambda v: f"{cur} {v:,.2f}"),
                    **{"Needs Review": atm_df["Needs Review"].map(lambda x: "⚠ Yes" if x else "✅ No")}
                ),
                use_container_width=True,
                hide_index=True
            )

            # Group by ATM vendor (optional, nice touch)
            grouped = {}
            for t in atm_txs:
                key = t.vendor or "UNKNOWN ATM"
                grouped.setdefault(key, []).append(t)

            for vendor, items in sorted(grouped.items(), key=lambda x: -len(x[1])):
                subtotal = sum(abs(i.amount) for i in items)
                cnt = len(items)
                with st.expander(f"{vendor} — {cnt} tx — {cur} {subtotal:,.2f}"):
                    details = pd.DataFrame([{
                        "Date": it.date,
                        "Amount": f"{cur} {abs(it.amount):,.2f}",
                        "Description": it.description
                    } for it in items])
                    st.dataframe(details, use_container_width=True, hide_index=True)
    
    # Downloads
    st.header("📥 Download")
    c1,c2,c3 ,c4 = st.columns(4)
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
        sc_csv = st.session_state.schedule_c_df.to_csv(index=False)
        st.download_button(
            "⬇ Schedule C CSV",
            sc_csv,
            "schedule_c.csv",
            mime="text/csv"
        )

    st.success("✅ Report generated. Verify totals against your bank statement.")
