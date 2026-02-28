import pytesseract
# bank_data_analysis.py
# Hybrid Bank Statement Analyzer (deterministic + optional LLM)
# Paste/replace your old file with this and run: streamlit run bank_data_analysis.py

import io
import re
import json
import logging
import phonenumbers
from commonregex import CommonRegex
import tempfile
import platform
import shutil
import os
from dataclasses import dataclass, asdict
from datetime import datetime, date
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from calendar import monthrange

import pdfplumber
import pandas as pd
import streamlit as st
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
try:
    import docx
except ImportError:
    docx = None

from bank_statement_parser import (
    parse_bank_statement, 
    detect_bank, 
    TransactionType, 
    TransactionCategory
)
from document_parser import DocumentParser
from schedule_c_categorizer import ScheduleCCategorizer
from account_code_mapper import AccountCodeMapper

# (Imports moved to top level)
MULTI_BANK_PARSER_AVAILABLE = True

# Platform-aware Tesseract configuration
if platform.system() == "Windows":
    # Only set tesseract path on Windows
    possible_tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for tess_path in possible_tesseract_paths:
        if Path(tess_path).exists():
            pytesseract.pytesseract.tesseract_cmd = tess_path
            break
# On macOS/Linux, tesseract should be in PATH (installed via brew/apt)
# No need to set tesseract_cmd explicitly

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
DATE_AT_START = re.compile(
    r'^[^\w\s]{0,4}\s*('
    r'(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01])(?:[/-]\d{2,4})?|'
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-/ ](?:0?[1-9]|[12][0-9]|3[01])(?:,?\s*[-/ ]\d{2,4})?'
    r')\s*(?![0-9]{5,})', re.I
)
AMOUNT_RE = re.compile(
    r'([+\-]?\(?\s*\$?\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})?\)?)'
)

def _mask_non_amount_entities(line: str) -> str:
    """
    Identify and hide phone numbers and dates to prevent them from being
    misidentified as amounts.
    """
    if not line:
        return ""
        
    masked_line = line
    entities_to_mask = []
    
    # 1. Identify Phone Numbers (Robust)
    try:
        import phonenumbers
        for match in phonenumbers.PhoneNumberMatcher(line, "US"):
            phone_str = line[match.start:match.end]
            entities_to_mask.append(phone_str)
    except Exception:
        pass
            
    # 2. Identify Dates (Regex based to avoid CommonRegex false positives on amounts)
    # Matches patterns like 11/24/25, 11-24-25, 11 24 25, 2024-11-20
    date_patterns = [
        r'(?<!\d)(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})(?!\d)',  # 11/24/25 or 11/24/2025
        r'(?<!\d)(?:\d{4}-\d{2}-\d{2})(?!\d)',              # 2024-11-20
        r'(?<!\d)(?:\d{1,2}\s+\d{1,2}\s+\d{2,4})(?!\d)',    # 11 24 25
        r'(?<!\w)(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:,?\s+\d{2,4})?(?!\w)' # Jan 15, 2025
    ]
    
    for pattern in date_patterns:
        for match in re.finditer(pattern, line, re.IGNORECASE):
            entities_to_mask.append(match.group(0))
    
    # 3. Mask identified entities (preserving length/offsets)
    # Sort by length descending to avoid partial replacements of longer strings
    entities_to_mask.sort(key=len, reverse=True)
    for entity in entities_to_mask:
        # Only mask if it's actually in masked_line (could have Been masked by a longer pattern)
        if entity in masked_line:
            masked_line = masked_line.replace(entity, " " * len(entity))
            
    return masked_line
# Matches full dates with year to avoid partial phone number matches
DATE_FULL_RE = re.compile(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b')
# matches MM/DD or MM-DD but NOT phone parts
DATE_SHORT_RE = re.compile(r'\b\d{1,2}[/-]\d{1,2}\b(?!\d|[-]\d)')
MULTI_DATE_AMT_RE = re.compile(r'(\d{1,2}[/-]\d{1,2}|[+\-]?\(?\s*\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})?\)?)')
CHECK_ROW_RE = re.compile(r'^\s*(\d{2,6})\b') 
FEE_KEYWORDS_RE = re.compile(
    r'(?:^|\s)(monthly\s+service\s+fee|service\s+fee|maintenance\s+fee|bank\s+fees?|account\s+fee|overdraft\s+fee)(?:\s|$)',
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
    
    # Handle parentheses (negative)
    if (s.startswith("(") and s.endswith(")")) or s.endswith("-"):
        negative = True
        s = s.replace('(', '').replace(')', '').replace('-', '')
    elif s.startswith("-"):
        negative = True
        
    s = re.sub(r'[A-Za-z\$£€₹]', '', s)  # drop currency letters
    s = s.replace(',', '').replace(' ', '')
    s = s.replace('¢', '').replace('\u00a2', '').replace('#', '').replace('*', '').replace('+', '')
    
    # Final cleanup: strictly numbers and dots
    s = re.sub(r'[^0-9\.]', '', s)
    
    if not s or not any(c.isdigit() for c in s):
        return None
        
    # Strict check: must have a decimal point followed by two digits, or not be a suspected zip/date
    if not re.search(r'\d+\.\d{2}', s):
        # If it's exactly 5 digits or doesn't have a decimal, it's suspicious
        if len(s) == 5 or '.' not in s:
            return None
            
    parts = s.split('.')
    if len(parts) > 2:
        s = "".join(parts[:-1]) + "." + parts[-1]
    try:
        val = float(s)
        return -abs(val) if negative else abs(val)
    except Exception:
        return None

def _short_vendor(v: str) -> str:
    if not v:
        return "UNKNOWN"
    
    v2 = v
    # Pattern to extract just the vendor name from "Orig CO Name:VENDOR Orig ID:..."
    m = re.search(r'Orig CO Name:\s*(.*?)\s*(?:Orig ID|Desc Date|Entry|CO\b)', v, flags=re.I)
    if m:
        v2 = m.group(1).strip()
    else:
        # Fallback to existing cleaning
        v2 = re.sub(r'\b(id|ref|code|num|number)[:\s]*\d+', '', v2, flags=re.I)
        v2 = re.sub(r'\b\d{5,}\b', '', v2)  # Remove long numeric IDs
        v2 = re.sub(r'(?i)\b(orig|co|name|entry|descr|desc)\b', '', v2)
    
    # Further cleanup of trailing noise
    v2 = re.sub(r'(?i)(?:Orig ID|Desc Date|CO Entry).*', '', v2)
    
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
# DocumentParser now imported from document_parser.py
    
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

# ----------------------------
# Multi-Bank Parser Integration
# ----------------------------
def parse_with_multi_bank_parser(text_lines, filename, manual_year=None, include_opening_balance=False, extract_check_memos=False):
    """
    Uses the new multi-bank parser to parse bank statements and converts to old Transaction format.
    
    Args:
        text_lines: List of text lines extracted from PDF
        filename: Name of the file
        manual_year: Optional manual year override
        include_opening_balance: Whether to include opening balance
        extract_check_memos: Whether to extract check memos
        
    Returns:
        tuple: (list of Transaction objects, dict of metadata)
    """
    if not MULTI_BANK_PARSER_AVAILABLE:
        return None, None
        
    try:
        # Convert lines to text
        text = '\n'.join(text_lines)
        
        # Parse using the new multi-bank parser
        # (Imports moved to top level)
        parsed_result = parse_bank_statement(text, manual_year=manual_year)
        
        if not parsed_result:
            logger.warning(f"Multi-bank parser returned None for {filename}")
            return None, None
            
        # Convert new Transaction format to old format
        old_transactions = []
        for new_tx in parsed_result.transactions:
            # Bridging to the older internal 'tx_type' based on amount sign
            # This is more robust than string-checking the Enum
            if new_tx.amount > 0:
                tx_type = "deposit"
            else:
                tx_type = "withdrawal"
            
            amount = new_tx.amount
            
            # Format date as string (old format expects string)
            if hasattr(new_tx.date, 'strftime'):
                date_str = new_tx.date.strftime("%m/%d/%Y")
            else:
                date_str = str(new_tx.date)
            
            # Extract cleaner vendor from description using the dedicated short_vendor function
            vendor = _short_vendor(new_tx.description) if new_tx.description else "UNKNOWN"
            
            # Map category
            category_map = {
                TransactionCategory.INCOME: "Income",
                TransactionCategory.EXPENSE: "Expense", 
                TransactionCategory.TRANSFER: "Transfer",
                TransactionCategory.UNCATEGORIZED: None
            }
            category = category_map.get(new_tx.category)
            
            # Create old-style Transaction
            old_tx = Transaction(
                date=date_str,
                transaction_type=tx_type,
                vendor=vendor,
                amount=amount,
                description=new_tx.description or "",
                raw_line=f"{date_str} {new_tx.description} {new_tx.amount}",
                section=None,
                category=category,
                needs_review=False,
                source="BANK"
            )
            old_transactions.append(old_tx)
        
        # Create metadata - mapping from new ParsedStatement properties
        per = getattr(parsed_result, "statement_period", None)
        from_d = per.from_date if per else None
        to_d = per.to_date if per else None
        
        meta = {
            "bank": parsed_result.bank_name,
            "statement_period": f"{from_d} to {to_d}" if from_d and to_d else None,
            "opening_balance": getattr(parsed_result, "beginning_balance", getattr(parsed_result, "opening_balance", 0.0)),
            "ending_balance": getattr(parsed_result, "ending_balance", getattr(parsed_result, "closing_balance", 0.0)),
            "parsed_from": "multi_bank_parser",
            "transactions_extracted": len(old_transactions),
            "errors": parsed_result.errors,
            "needs_review": parsed_result.needs_review
        }
        
        # Log success and return
        if not old_transactions and parsed_result.bank_name == "unknown":
            logger.warning(f"No transactions extracted from {filename} by multi-bank parser (Bank: Unknown)")
            return None, None
            
        logger.info(f"Successfully parsed {filename} with {parsed_result.bank_name} parser (Tx: {len(old_transactions)})")
        return old_transactions, meta
        
    except Exception as e:
        logger.error(f"Error using multi-bank parser for {filename}: {e}", exc_info=True)
        return None, None

class FallbackStatementParser:
    def __init__(self, include_opening_balance: bool = False, extract_check_memos: bool = False):
        self.include_opening_balance = include_opening_balance
        self.extract_check_memos = extract_check_memos
        self.opening_balance: Optional[float] = None
        self.statement_start_date: Optional[str] = None
        self.statement_year: Optional[int] = None

    # List of (section_key, regex) tuples — order matters: first match wins per line
    SECTION_PATTERNS = [
        # Chase
        ("DEPOSITS",              re.compile(r'\bdeposits\s+and\s+additions\b', re.I)),
        ("CHECKS",                re.compile(r'\bchecks\s+paid\b', re.I)),
        ("ATM",                   re.compile(r"ATM\s*&\s*DEBIT\s*CARD\s*WITHDRAWALS", re.I)),
        ("ELECTRONIC_WITHDRAWALS",re.compile(r'\belectronic\s+withdrawals?\b', re.I)),
        ("FEES",                  re.compile(r'^(monthly\s+service\s+fee|service\s+fee|bank\s+fees?|fees\s+charged|service\s+charges?)\s*$', re.I)),

        # Bank of America
        ("DEPOSITS",              re.compile(r'\bdeposits?\s+and\s+(other\s+)?(?:credits?|additions)\b', re.I)),
        ("WITHDRAWALS",           re.compile(r'\bwithdrawals?\s+and\s+other\s+(?:debits?|charges?)\b', re.I)),
        # Fifth Third
        ("WITHDRAWALS",           re.compile(r'\bwithdrawals?\s*/\s*debits?\b', re.I)),
        ("DEPOSITS",              re.compile(r'\bdeposits?\s*/\s*credits?\b', re.I)),
        # US Bank
        ("DEPOSITS",              re.compile(r'^Other\s+Deposits?\b', re.I)),
        ("WITHDRAWALS",           re.compile(r'^Other\s+Withdrawals?\b', re.I)),
        # BMO
        ("DEPOSITS",              re.compile(r'\b(?:Deposits?|Credits?)\b', re.I)),
        ("WITHDRAWALS",           re.compile(r'\b(?:Withdrawals?|Debits?|Charges?)\b', re.I)),

        # AMEX
        ("DEPOSITS",              re.compile(r'^\s*Payments\s*/\s*Credits\s*$', re.I)),
        # Match "New Charges" or "Detailed Transactions" or just "Amount" column header
        ("WITHDRAWALS",           re.compile(r'^\s*(New\s+Charges|Detailed\s+Transactions|Amount|Fees)\s*$', re.I)),
    ]
    def _get_section_pattern(self, key: str):
        """Return the first compiled regex for the given section key."""
        for sec_name, pat in self.SECTION_PATTERNS:
            if sec_name == key:
                return pat
        return None

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
        # Search through more lines (first 100 lines) to find year
        search_lines = min(100, len(lines))
        
        for line in lines[:search_lines]:
            line_clean = line.strip()
            
            # Pattern 1: Statement period with year (e.g., "Statement Period: 12/01/2024 - 12/31/2024")
            m = re.search(r'(statement period|period|statement date|dates?).*?(\d{1,2}[/-]\d{1,2}[/-](\d{4}))', line, re.I)
            if m:
                year = int(m.group(3))
                if 2000 <= year <= datetime.now().year + 1:
                    return year
            
            # Pattern 2: Date range with year (e.g., "12/01/2024 - 12/31/2024" or "December 1, 2024 - December 31, 2024")
            m = re.search(r'(\d{1,2}[/-]\d{1,2}[/-](\d{4}))\s*[-–to]+\s*\d{1,2}[/-]\d{1,2}[/-]\d{4}', line, re.I)
            if m:
                year = int(m.group(2))
                if 2000 <= year <= datetime.now().year + 1:
                    return year
            
            # Pattern 3: Any full date with 4-digit year (e.g., "12/17/2024")
            m = re.search(r'\b\d{1,2}[/-]\d{1,2}[/-](\d{4})\b', line)
            if m:
                year = int(m.group(1))
                if 2000 <= year <= datetime.now().year + 1:
                    return year
            
            # Pattern 4: YYYY-MM-DD format
            m = re.search(r'\b(\d{4})-\d{1,2}-\d{1,2}\b', line)
            if m:
                year = int(m.group(1))
                if 2000 <= year <= datetime.now().year + 1:
                    return year
            
            # Pattern 5: Month name with year (e.g., "December 2024", "Dec 2024")
            m = re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[,\s]+(\d{4})\b', line, re.I)
            if m:
                year = int(m.group(2))
                if 2000 <= year <= datetime.now().year + 1:
                    return year
            
            # Pattern 6: Year explicitly mentioned (e.g., "For the year 2024")
            m = re.search(r'\b(for the year|year|fiscal year)\s+(\d{4})\b', line, re.I)
            if m:
                year = int(m.group(2))
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
        if re.match(r'^(daily ending balance|daily ending|statement period|opening balance|ending balance|closing balance|page\s+\d+|minimum\s+payment|new\s+balance|previous\s+balance|payment\s+due|statement\s+date|total\s+for|automatic\s+payment)', low):
            return True
        # New bank section TOTAL lines (avoid pre-filtering header signals)
        if re.match(r'^(total\s+deposits?|total\s+withdrawals?|total\s+service\s+fees?|other\s+deposits?|other\s+withdrawals?|account\s+summary|balance\s+summary)', low):
            return True
        # AMEX specific summary lines
        if any(k in low for k in [
            "minimum payment due", "pay in full portion", "pay over time portion", 
            "total balance", "total new charges",
            "account total", "days in billing period", "late payment warning",
            "important notices", "preset spending limit", "amount of $", "your account current",
            "minimum payment warning", "if you make only", "for example", "will pay off",
            "estimated total", "member summary", "charges by category", "summary of account"
        ]):
            return True
        # If the line contains multiple date+amount pairs (daily ending tables) skip
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
    def parse_statement(self, lines: List[str], manual_year: Optional[int] = None) -> Tuple[List[Transaction], Dict[str, Any]]:
        txs: List[Transaction] = []
        
        # Extract statement year - use manual year if provided, otherwise auto-detect
        if manual_year:
            self.statement_year = manual_year
        else:
            self.statement_year = self._extract_statement_year(lines)
        
        global _STATEMENT_YEAR
        _STATEMENT_YEAR = self.statement_year
        
        # 1. Pre-clean
        for ln in lines:
            self._extract_opening_balance(ln)
        
        # Identify and exclude the "Account Summary" block from transaction parsing
        raw_text = "\n".join(lines)
        summary_m = re.search(r'(?:Account\s+Summary|Statement\s+Summary).*?(?=Monthly\s+Activity|Detailed\s+Transactions|New\s+Transactions|Member\s+Summary|Account\s+Activity|Amount|$)', raw_text, re.I | re.DOTALL)
        exclude_range = (summary_m.start(), summary_m.end()) if summary_m else (0, 0)
        
        cleaned = []
        char_count = 0
        for ln in lines:
            line_len = len(ln) + 1
            if not (exclude_range[0] <= char_count < exclude_range[1]):
                ln_stripped = ln.strip()
                if ln_stripped and not self._is_summary_line(ln_stripped):
                    cleaned.append(ln_stripped)
            char_count += line_len
        
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
            for sec_name, pat in self.SECTION_PATTERNS:
                if pat.search(ln):
                    # Section headers must be relatively short to avoid summary noise
                    if len(ln.strip()) < 50:
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
            _checks_pat = self._get_section_pattern("CHECKS")
            if _checks_pat and _checks_pat.search(ln):
                in_checks = True
                continue  # Skip the header line itself
            # Check if we're exiting the checks section
            if in_checks and any(
                self._get_section_pattern(s) and self._get_section_pattern(s).search(ln)
                for s in ["ATM", "FEES", "ELECTRONIC_WITHDRAWALS"]
            ):
                in_checks = False
                continue
            # If we're in checks section, add the line
            if in_checks:
                # If extract_check_memos is enabled, we keep lines even if they don't start with a number
                # as they might be memos for the previous check line.
                if self.extract_check_memos:
                    check_lines.append(ln)
                else:
                    # Original behavior: Only add lines that look like check transactions (start with check number)
                    if re.match(r'^\d{3,6}\s', ln):
                        check_lines.append(ln)

        check_txs = self._parse_checks_section(check_lines)
        txs.extend(check_txs)
        # ---- END CHECK OVERRIDE ----

        for block_lines, section in blocks:
            if section == "CHECKS":
                continue
            block_text = " ".join(block_lines)
            # skip if looks like a total or summary block
            if section != "FEES" and (
                re.search(r'\btotal\b.*\d', block_text, re.I) or
                self._is_summary_line(block_text)
            ):
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
        # Deduplication logic (merging duplicates from multi-page summaries/details)
        final_txs = []
        seen = set()
        for tx in txs:
            # Create a identifying key: (date, absolute amount, vendor_prefix, desc_prefix)
            # We include more vendor and description to avoid merging distinct Zelle payments
            v_norm = (tx.vendor or "").strip().lower()[:20]
            d_norm = (tx.description or "").strip().lower()[:15]
            key = (tx.date, abs(tx.amount), v_norm, d_norm)
            if key not in seen:
                seen.add(key)
                final_txs.append(tx)
            else:
                # If we already saw it but this one has a longer vendor name, prefer it
                for i, existing in enumerate(final_txs):
                    v_e_norm = (existing.vendor or "").strip().lower()[:20]
                    d_e_norm = (existing.description or "").strip().lower()[:15]
                    if (existing.date == tx.date and 
                        abs(existing.amount) == abs(tx.amount) and 
                        v_e_norm == v_norm and
                        d_e_norm == d_norm):
                        if len(tx.vendor or "") > len(existing.vendor or ""):
                            final_txs[i] = tx
                        break
        
        meta = {"parsed_from": "chase_sectioned_fallback", "transactions_extracted": len(final_txs)}
        return final_txs, meta
    def _parse_checks_section(self, lines: List[str]) -> List[Transaction]:
        txs = []
        current_check = None

        for ln in lines:
            ln_stripped = ln.strip()
            if not ln_stripped:
                continue

            # Check if this is a primary check line (Starts with 3-6 digit number)
            # Pattern: CHECKNO [OPTIONAL TEXT] [OPTIONAL DATE] AMOUNT
            # Date pattern: \d{1,2}/\d{1,2}
            # Amount pattern: \d{1,3}(?:,\d{3})*\.\d{2}
            
            # This regex tries to capture:
            # 1. Check number
            # 2. Middle text (potential memo + date)
            # 3. Final amount
            m = re.match(r'^(\d{3,6})\s+(.*?)(\d{1,3}(?:,\d{3})*\.\d{2})$', ln_stripped)
            
            if m:
                # If we have a previous check, finalize it
                if current_check:
                    txs.append(self._finalize_check_tx(current_check))
                
                check_no = m.group(1)
                middle_text = m.group(2).strip()
                amt_raw = m.group(3)
                
                # Try to extract date from middle_text
                date_paid = ""
                memo = middle_text
                date_m = re.search(r'(\d{1,2}/\d{1,2})', middle_text)
                if date_m:
                    date_paid = date_m.group(1)
                    # Clean memo: remove the date and some separators like '^', '*'
                    memo = middle_text.replace(date_paid, "").replace("^", "").replace("*", "").strip()
                else:
                    memo = middle_text.replace("^", "").replace("*", "").strip()

                current_check = {
                    "check_no": check_no,
                    "date": date_paid,
                    "amount": self._parse_amount(amt_raw),
                    "memo": memo,
                    "raw_lines": [ln]
                }
            elif current_check and self.extract_check_memos:
                # This could be a continuation of the memo
                # Clean it up: remove common stray indicators
                cleaned_extra = ln_stripped.replace("^", "").replace("*", "").strip()
                if cleaned_extra:
                    if current_check["memo"]:
                        current_check["memo"] += " " + cleaned_extra
                    else:
                        current_check["memo"] = cleaned_extra
                current_check["raw_lines"].append(ln)

        # Finalize last check
        if current_check:
            txs.append(self._finalize_check_tx(current_check))

        return txs

    def _finalize_check_tx(self, check_data: Dict[str, Any]) -> Transaction:
        # Normalize date if we found one
        date_norm = ""
        if check_data["date"]:
            date_norm = self._parse_date(check_data["date"]) or ""

        # Build description: ONLY use the memo text if available, otherwise empty as requested
        desc = check_data["memo"] if check_data["memo"] else ""

        return Transaction(
            date=date_norm,
            transaction_type="withdrawal",
            vendor=f"Check #{check_data['check_no']}",
            amount=-abs(check_data["amount"]) if check_data["amount"] else 0.0,
            description=desc,
            raw_line="\n".join(check_data["raw_lines"]),
            section="CHECKS",
            needs_review=False
        )



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
        r'([+\-]?\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)'
    )

    IGNORE_WORDS = ["opening balance", "closing balance", "balance", "running balance"]

    def parse(self, lines: List[str]) -> List[Transaction]:
        txs = []

        # -------- DETECT FORMAT --------
        # SadaPay detection: check for specific transaction markers
        is_sadapay = any("transf" in ln.lower() or "cr/" in ln.lower() or "dr/" in ln.lower() for ln in lines)
        
        # Tabular detection: Only trigger if we see multi-column headers AND it's NOT a typical blocky statement
        # (Avoid Page 1 summaries like AMEX "Member Summary")
        is_tabular = False
        text_joined = "\n".join(lines[:200]).lower()
        if ("debit" in text_joined or "credit" in text_joined) and ("balance" in text_joined):
             # Gating: if it looks like AMEX Page 1, don't use tabular for everything
             if "member summary" in text_joined or "charges by category" in text_joined:
                 is_tabular = False
             else:
                 is_tabular = True

        if is_sadapay:
            txs = self._parse_sadapay(lines)
        elif is_tabular:
            txs = self._parse_tabular(lines)
        else:
            txs = self._parse_simple(lines)

        # Final Deduplication Logic (consistent with FallbackStatementParser)
        final_txs = []
        seen = set()
        for tx in txs:
            v_norm = (tx.vendor or "").strip().lower()[:20]
            d_norm = (tx.description or "").strip().lower()[:15]
            key = (tx.date, abs(tx.amount), v_norm, d_norm)
            if key not in seen:
                seen.add(key)
                final_txs.append(tx)
            else:
                for i, existing in enumerate(final_txs):
                    v_e_norm = (existing.vendor or "").strip().lower()[:20]
                    d_e_norm = (existing.description or "").strip().lower()[:15]
                    if (existing.date == tx.date and abs(existing.amount) == abs(tx.amount) and v_e_norm == v_norm and d_e_norm == d_norm):
                        if len(tx.vendor or "") > len(existing.vendor or ""):
                            final_txs[i] = tx
                        break
        return final_txs

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

                # Parse amount (after masking phone/dates)
                masked_ln = _mask_non_amount_entities(ln)
                amt_m = self.AMOUNT_RE.search(masked_ln)
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

            # Amounts (consistent with global AMOUNT_RE, allow optional decimal to be robust to poor OCR)
            masked_ln = _mask_non_amount_entities(ln)
            nums = re.findall(r'([+\-]?\(?\s*\$?\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})?\)?)', masked_ln)
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
        fb_tester = FallbackStatementParser()
        for ln in lines:
            low = ln.lower().strip()

            if any(w in low for w in self.IGNORE_WORDS) or fb_tester._is_summary_line(ln):
                continue

            # DATE
            d = self.DATE_RE.search(ln)
            if not d:
                continue
            date_raw = d.group(1).replace(",", "")
            date_norm = _normalize_date_token(date_raw)

            # AMOUNT (after masking phone/dates)
            masked_ln = _mask_non_amount_entities(ln)
            m = self.AMOUNT_RE.findall(masked_ln)
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
    # (Imports moved to top level)

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

    # Sort rules by priority (lower number = higher priority, default = 999)
    # This allows more specific rules to be processed first
    sorted_rules = sorted(business_rules, key=lambda r: r.get("priority", 999))

    # Apply rules to both transactions and filtered_transactions (if present)
    for key in ("transactions", "filtered_transactions"):
        if key not in st.session_state:
            continue
        for tx in st.session_state[key]:
            text = f"{tx.vendor or ''} {tx.description or ''}"
            text = _norm(text)
            
            # Use absolute value of amount for comparison (withdrawals are negative)
            tx_amount = abs(getattr(tx, "amount", 0.0))

            matched = False
            for rule in sorted_rules:
                # 1. Check primary keyword (required)
                kw = _norm(rule.get("keyword", ""))
                if not kw or kw not in text:
                    continue
                
                # 2. Check amount range filters (optional)
                min_amt = rule.get("min_amount")
                max_amt = rule.get("max_amount")
                
                if min_amt is not None and tx_amount < min_amt:
                    continue  # Amount too small for this rule
                    
                if max_amt is not None and tx_amount > max_amt:
                    continue  # Amount too large for this rule
                
                # 3. Check exclusion keywords (optional) - skip if any match
                exclude_kws = rule.get("exclude_keywords", [])
                if exclude_kws and any(_norm(excl) in text for excl in exclude_kws):
                    continue  # Excluded by keyword
                
                # 4. Check additional keywords (optional) - ALL must match
                additional_kws = rule.get("additional_keywords", [])
                if additional_kws and not all(_norm(kw_add) in text for kw_add in additional_kws):
                    continue  # Not all additional keywords match
                
                # 5. All conditions met - apply the rule
                tx.account_code = rule.get("account_code")
                tx._mapped_by_rule = True
                matched = True
                break  # First matching rule wins (priority-based)

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

def extract_chase_summary(raw_text):
    """
    Extract pre-formatted summary from Chase bank statements.
    
    Chase statements contain accurate summaries between *start*summary and *end*summary markers.
    This avoids parsing errors from treating account numbers as transaction amounts.
    
    Returns: dict with summary data or None if not found
    """
    # (Imports moved to top level)
    
    # Look for Chase summary section
    summary_match = re.search(r'\*start\*summary.*?\*end\*summary', raw_text, re.DOTALL | re.IGNORECASE)
    if not summary_match:
        return None
    
    summary_section = summary_match.group(0)
    
    # Initialize result structure
    result = {
        "Beginning Balance": {"count": "", "amount": 0.0},
        "Deposits and Additions": {"count": 0, "amount": 0.0},
        "ATM & Debit Card Withdrawals": {"count": 0, "amount": 0.0},
        "Electronic Withdrawals": {"count": 0, "amount": 0.0},
        "Checks Paid": {"count": 0, "amount": 0.0},
        "Other Withdrawals": {"count": 0, "amount": 0.0},
        "Fees": {"count": 0, "amount": 0.0},
        "Ending Balance": {"count": "", "amount": 0.0}
    }
    
    # Parse each line in the summary
    lines = summary_section.split('\n')
    for line in lines:
        # Skip markers and headers
        if any(skip in line for skip in ['*start*', '*end*', 'CHECKING SUMMARY', 'INSTANCES AMOUNT', 'Chase Business']):
            continue
        if not line.strip():
            continue
        
        # Try to extract: Category [instances] [amount]
        # Pattern handles: "Category $amount" or "Category instances amount" or "Category instances -amount"
        match = re.search(r'^(.+?)\s+(\d+)?\s*([\$\-]?[\d,]+\.?\d*)$', line.strip())
        if match:
            category_raw = match.group(1).strip()
            instances_str = match.group(2) if match.group(2) else ""
            amount_str = match.group(3).strip().replace('$', '').replace(',', '')
            
            # Map category names (Chase format → our format)
            category_mapping = {
                "beginning balance": "Beginning Balance",
                "deposits and additions": "Deposits and Additions",
                "checks paid": "Checks Paid",
                "atm & debit card withdrawals": "ATM & Debit Card Withdrawals",
                "atm and debit card withdrawals": "ATM & Debit Card Withdrawals",
                "electronic withdrawals": "Electronic Withdrawals",
                "other withdrawals": "Other Withdrawals",
                "fees": "Fees",
                "ending balance": "Ending Balance"
            }
            
            category_key = category_mapping.get(category_raw.lower())
            if category_key:
                try:
                    amount = float(amount_str)
                    instances = int(instances_str) if instances_str else ""
                    result[category_key] = {"count": instances, "amount": amount}
                except ValueError:
                    continue
    
    return result


def extract_universal_bank_summary(raw_text: str) -> Optional[Dict]:
    """
    Extract summary data from non-Chase bank statements.
    Supports: Bank of America, BMO, Fifth Third, US Bank.
    Returns a dict formatted like extract_chase_summary(), or None if not detected.
    """
    # (Imports moved to top level)

    def clean_amount(s: str) -> float:
        """Parse a dollar amount string to float, handling negatives and trailing dashes."""
        s = s.strip().replace('$', '').replace(',', '').replace(' ', '')
        if s.endswith('-'):
            s = '-' + s[:-1]
        try:
            return float(s)
        except ValueError:
            return 0.0

    # Base result structure (matches Chase format for compatibility)
    result = {
        "Beginning Balance": {"count": "", "amount": 0.0},
        "Deposits and Additions": {"count": 0, "amount": 0.0},
        "ATM & Debit Card Withdrawals": {"count": 0, "amount": 0.0},
        "Electronic Withdrawals": {"count": 0, "amount": 0.0},
        "Checks Paid": {"count": 0, "amount": 0.0},
        "Other Withdrawals": {"count": 0, "amount": 0.0},
        "Fees": {"count": 0, "amount": 0.0},
        "Ending Balance": {"count": "", "amount": 0.0},
    }

    lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
    amt_re = re.compile(r'[\$]?\s*([+\-]?[\d,]+\.?\d{0,2})\-?$')

    # ── Bank of America ──────────────────────────────────────────────────────
    # Sections: "Deposits and other credits", "Withdrawals and other debits", "Service fees"
    # Each section ends with a "Total ..." line we can use.
    if re.search(r'Bank\s*of\s*America|BKOFAMERICA', raw_text, re.I):
        # Beginning balance
        m = re.search(r'Beginning\s*balance\s+on\s+\w+\s+\d+\s+\$?\s*([\d,]+\.?\d{0,2})', raw_text, re.I)
        if m:
            result["Beginning Balance"]["amount"] = clean_amount(m.group(1))
        # Ending balance
        m = re.search(r'Ending\s*balance\s+on\s+\w+\s+\d+,?\s+\d+\s+\$?\s*([\d,]+\.?\d{0,2})', raw_text, re.I)
        if m:
            result["Ending Balance"]["amount"] = clean_amount(m.group(1))
        # Total deposits
        m = re.search(r'Total\s+deposits?\s+and\s+other\s+credits?\s+\$?\s*([\d,]+\.?\d{0,2})', raw_text, re.I)
        if m:
            result["Deposits and Additions"]["amount"] = clean_amount(m.group(1))
        # Total withdrawals
        m = re.search(r'Total\s+withdrawals?\s+and\s+other\s+debits?\s+[-\$]?\s*([\d,]+\.?\d{0,2})', raw_text, re.I)
        if m:
            result["Other Withdrawals"]["amount"] = -abs(clean_amount(m.group(1)))
        # Total fees
        m = re.search(r'Total\s+(?:service\s+)?fees?\s+[-\$]?\s*([\d,]+\.?\d{0,2})', raw_text, re.I)
        if m:
            result["Fees"]["amount"] = -abs(clean_amount(m.group(1)))
        return result

    # ── BMO Bank ─────────────────────────────────────────────────────────────
    # "Account Summary" block with "DEPOSIT AMOUNT ... WITHDRAWAL AMOUNT"
    if re.search(r'\bBMO\b', raw_text, re.I):
        # We use re.DOTALL and tighter value patterns because BMO summaries are often multi-line in OCR
        # Beginning balance
        m = re.search(r'BEGINNING\s+BALANCE\s+AS.*?\$\s*([\d, ]+\.\d{2})', raw_text, re.I | re.DOTALL)
        if m: result["Beginning Balance"]["amount"] = clean_amount(m.group(1))
        
        # Ending balance
        m = re.search(r'ENDING\s+BALANCE\s+AS.*?\$\s*([\d, ]+\.\d{2})', raw_text, re.I | re.DOTALL)
        if m: result["Ending Balance"]["amount"] = clean_amount(m.group(1))
        
        # Total deposits
        m = re.search(r'DEPOSIT\s+AMOUNT.*?\$\s*([\d, ]+\.\d{2})', raw_text, re.I | re.DOTALL)
        if m: result["Deposits and Additions"]["amount"] = clean_amount(m.group(1))
        
        # Total withdrawals
        m = re.search(r'WITHDRAWAL\s+AMOUNT.*?\$\s*([\d, ]+\.\d{2})', raw_text, re.I | re.DOTALL)
        if m: result["Other Withdrawals"]["amount"] = -abs(clean_amount(m.group(1)))
        
        # Fallback to general BMO patterns if above fails
        if result["Ending Balance"]["amount"] == 0.0:
            m = re.search(r'BMO\s+ELITE\s+BUSINESS\s+CKG.*?\$\s*([\d, ]+\.\d{2})', raw_text, re.I | re.DOTALL)
            if not m:
                m = re.search(r'(?:BALANCE|ENDING\s+BALANCE).*?\$\s*([\d, ]+\.\d{2})', raw_text, re.I | re.DOTALL)
            if m: result["Ending Balance"]["amount"] = clean_amount(m.group(1))
        return result

    # ── Fifth Third Bank ─────────────────────────────────────────────────────
    # "Beginning Balance $X.XX  Number of Days in Period 30"  on one line (OCR merged)
    # "Ending Balance $X.XX"
    if re.search(r'Fifth\s*Third|53\.com', raw_text, re.I):
        m = re.search(r'Beginning\s*Balance\s*\$?\s*([\d,]+\.?\d{0,2})', raw_text, re.I)
        if m:
            result["Beginning Balance"]["amount"] = clean_amount(m.group(1))
        m = re.search(r'Ending\s*Balance\s*\$?\s*([\d,]+\.?\d{0,2})', raw_text, re.I)
        if m:
            result["Ending Balance"]["amount"] = clean_amount(m.group(1))
        # Try to find total deposits and withdrawals
        m = re.search(r'Total\s+(?:Deposits?|Credits?)\s*\$?\s*([\d,]+\.?\d{0,2})', raw_text, re.I)
        if m:
            result["Deposits and Additions"]["amount"] = clean_amount(m.group(1))
        m = re.search(r'Total\s+(?:Withdrawals?|Debits?)\s*\$?\s*([\d,]+\.?\d{0,2})', raw_text, re.I)
        if m:
            result["Other Withdrawals"]["amount"] = -abs(clean_amount(m.group(1)))
        return result

    # ── US Bank ──────────────────────────────────────────────────────────────
    # Account Summary table: rows like "Beginning Balance on Nov 3  $  26,427.22"
    if re.search(r'U\.?S\.?\s*Bank|usbank\.com', raw_text, re.I):
        m = re.search(r'Beginning\s*Balance\s+(?:on\s+\w+\s+\d+)?\s*\$?\s*([\d,]+\.?\d{0,2})', raw_text, re.I)
        if m:
            result["Beginning Balance"]["amount"] = clean_amount(m.group(1))
        m = re.search(r'Ending\s*Balance\s+(?:on\s+\w+\s+\d+,?\s+\d+)?\s*\$?\s*([\d,]+\.?\d{0,2})', raw_text, re.I)
        if m:
            result["Ending Balance"]["amount"] = clean_amount(m.group(1))
        # Other Deposits row: "Other Deposits  N  amount"
        m = re.search(r'Other\s+Deposits?\s+(\d+)\s+([\d,]+\.?\d{0,2})', raw_text, re.I)
        if m:
            result["Deposits and Additions"]["count"] = int(m.group(1))
            result["Deposits and Additions"]["amount"] = clean_amount(m.group(2))
        # Other Withdrawals row: "Other Withdrawals  N  amount-"
        m = re.search(r'Other\s+Withdrawals?\s+(\d+)\s+([\d,]+\.?\d{0,2})-?', raw_text, re.I)
        if m:
            result["Other Withdrawals"]["count"] = int(m.group(1))
            result["Other Withdrawals"]["amount"] = -abs(clean_amount(m.group(2)))
        return result

    return None  # Unknown bank — fall back to transaction-computed summary


# ----------------------------
# Credit Card Summary Extraction & Rendering
# ----------------------------

def extract_cc_summary(raw_text: str) -> Dict[str, Any]:
    """
    Extract summary fields from credit card statements with extreme robustness.
    Handles noisy OCR (duplicated chars), split numbers across lines, and fuzzy labels.
    """
    # (Imports moved to top level)
    
    # Initialize result with defaults
    result = {
        "Account Number": "N/A",
        "Previous Balance": 0.0,
        "Payment, Credits": 0.0,
        "Purchases": 0.0,
        "Cash Advances": 0.0,
        "Balance Transfers": 0.0,
        "Fees Charged": 0.0,
        "Interest Charged": 0.0,
        "New Balance": 0.0,
        "Opening/Closing Date": "N/A",
        "Revolving Credit Amount": 0.0,
        "Available Credit": 0.0,
        "Cash Access Line": 0.0,
        "Available for Cash": 0.0,
        "Past Due Amount": 0.0,
        "Balance over the Credit Access Line": 0.0
    }
    
    def denoise_text(text: str) -> str:
        """Collapse duplicated characters in uppercase blocks (OCR artifact)"""
        def collapse(match):
            s = match.group(0)
            new_s = ""
            i = 0
            while i < len(s):
                new_s += s[i]
                if i + 1 < len(s) and s[i] == s[i+1]: i += 2
                else: i += 1
            return new_s
        return re.sub(r'[A-Z]{4,}', collapse, text)

    def clean_amt(s: str) -> float:
        if not s: return 0.0
        s = s.strip().replace('$', '').replace(',', '').replace(' ', '').replace('+', '')
        if '(' in s and ')' in s: s = '-' + s.replace('(', '').replace(')', '')
        if s.endswith('-'): s = '-' + s[:-1]
        # Remove any non-numeric/sign/dot chars sneaked in by OCR
        s = re.sub(r'[^0-9.\-]', '', s)
        try: return float(s)
        except ValueError: return 0.0

    # 1. Prepare Text (Denoise & Join split lines)
    text = denoise_text(raw_text)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    joined_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines):
            nxt = lines[i+1]
            # Handle split numbers: 1,234. + 56 OR 1,23 + 4.56
            if (re.search(r'[\d,]+\.\d?$', line) and re.match(r'^\d{1,2}$', nxt)) or \
               (re.search(r'[\d,]+$', line) and re.match(r'^\.\d{2}$', nxt)):
                line += nxt
                i += 1
        joined_lines.append(line)
        i += 1
    
    processed_text = '\n'.join(joined_lines)

    # 2. Identify Section Boundaries (Fuzzy)
    summary_start = 0
    for hp in [r'ACCOUNT\s*SUMMARY', r'SUMMARY\s*OF\s*ACCOUNT', r'ACCOUNT\s*AT\s*A\s*GLANCE']:
        m = re.search(hp, processed_text, re.I)
        if m:
            summary_start = m.start()
            break
            
    summary_text = processed_text[summary_start:]
    m_end = re.search(r'ACCOUNT\s*ACTIVITY|TRANSACTION\s*DETAIL|ACTIVITY\s*DETAIL', summary_text[100:], re.I)
    if m_end: summary_text = summary_text[:100+m_end.start()]
    else: summary_text = summary_text[:3000]

    # 3. Field Matching (Fuzzy Labels)
    p_val = r'[\$]?\s*([+\-]?\(?\s*[\d,]+\.\d{2}|[+\-]?\(?\s*[\d,]{1,9}\)?)'
    
    field_patterns = {
        "Previous Balance": r'Previ?ous\s*Bala?nce',
        "Payment, Credits": r'Pay?ments?(?:,\s*Credi?ts?|/Credi?ts?|/Credi?t)',
        "Purchases": r'(?:Purch?ases?|New\s*Charges?)',
        "Cash Advances": r'Cash\s*Adva?nces?(?!\s+Line)',
        "Balance Transfers": r'Bala?nce\s*Transf?er?s?',
        "Fees Charged": r'Fees(?:\s*Char?ged)?',
        "Interest Charged": r'Inte?re?st(?:\s*Char?ged)?',
        "New Balance": r'New\s*Bala?nce',
        "Revolving Credit Amount": r'Revolv?ing\s*Credi?t\s*Amou?nt|Credi?t\s*Limit',
        "Available Credit": r'Avai?labl?e\s*Credi?t',
        "Cash Access Line": r'Cash\s*Acce?ss\s*Line',
        "Available for Cash": r'Avai?labl?e\s*for\s*Cash',
        "Past Due Amount": r'Past\s*Due\s*Amou?nt',
    }

    # First pass: try to find each label and its value on the same line or immediate proximity
    for key, f_pat in field_patterns.items():
        # A) Same line
        same_line_pat = f'{f_pat}[^\\n]{{0,40}}?{p_val}'
        m = re.search(same_line_pat, summary_text, re.I)
        if m:
            result[key] = clean_amt(m.group(1))
            continue
            
        # B) Multi-line (stricter than before)
        other_labels = '|'.join([p for k, p in field_patterns.items() if k != key])
        # Allow up to 100 chars but don't jump over another label
        multi_line_pat = f'{f_pat}(?:(?!(?:{other_labels})).){{0,100}}?{p_val}'
        m = re.search(multi_line_pat, summary_text, re.I | re.DOTALL)
        if m:
            result[key] = clean_amt(m.group(1))

    # 4. Block Mapping Fallback (For dissociated layouts like AMEX "Account Total")
    # If key fields are still 0, look for specialized "Total" block
    missing_count = sum(1 for k in ["Previous Balance", "New Balance", "Payment, Credits"] if result[k] == 0.0)
    if missing_count >= 1:
        total_m = re.search(r'(?:Account\s*Total|Total\s*Summary)', summary_text, re.I)
        if total_m:
            sub = summary_text[total_m.start():]
            # Try to find all values in this sub-block
            all_vals = []
            for m in re.finditer(p_val, sub):
                all_vals.append((m.start(), clean_amt(m.group(1))))
            
            # AMEX order is usually: Prev, Pay, New Charges, Fees, Interest, New Balance
            # We map labels to values by order if both are found in the block
            labels_found = []
            for key, f_pat in field_patterns.items():
                m = re.search(f_pat, sub, re.I)
                if m: labels_found.append((m.start(), key))
            labels_found.sort() # sort by appearance in text
            
            if labels_found and len(all_vals) >= len(labels_found):
                # We assume the first value appearing after ALL labels starts the value block
                # (Or they are interleaved but missed by regex)
                # Let's try 1-to-1 mapping of those that are still 0
                val_idx = 0
                # Find common summary values: must have decimal or $ sign to avoid "30 days" noise
                first_label_pos = labels_found[0][0]
                dollar_box = [v for pos, v in all_vals if pos > first_label_pos and ('.' in str(v) or '$' in sub[pos-2:pos+1])]
                
                for i, (pos, key) in enumerate(labels_found):
                    if i < len(dollar_box):
                        if result[key] == 0.0:
                            result[key] = dollar_box[i]

    # Special Case: Account Number
    acc_match = re.search(r'Account\s*Number[:\s]+([\d\s]{10,25})', processed_text, re.I)
    if acc_match: result["Account Number"] = acc_match.group(1).strip()

    # Dates
    date_pat = r'(?:Statement\s*Period|Opening/Closing\s*Date)\s*[:]?\s*(\d{1,2}/\d{1,2}/\d{2,4}\s*[-–to]+\s*\d{1,2}/\d{1,2}/\d{2,4})'
    date_match = re.search(date_pat, processed_text, re.I)
    if date_match: result["Opening/Closing Date"] = date_match.group(1).strip()
    
    return result

def render_cc_summary(cc_summary: Dict[str, Any]):
    """
    Render a unified credit card summary table in the UI.
    Uses exactly the same design system as the bank statement tables.
    """
    # (Imports moved to top level)
    
    def f(val):
        if isinstance(val, (float, int)):
            if val > 0 and val == cc_summary.get("Purchases"):
                return f"+${val:,.2f}"
            if val < 0:
                return f"-${abs(val):,.2f}"
            return f"${val:,.2f}"
        return val

    # Create a DataFrame to use the existing render_chase_table function
    items = []
    
    # Priority rows
    main_rows = [
        "Previous Balance", "Payment, Credits", "Purchases", "Cash Advances", 
        "Balance Transfers", "Fees Charged", "Interest Charged", "New Balance"
    ]
    for label in main_rows:
        items.append({"Type": label, "INSTANCES": "", "AMOUNT": f(cc_summary.get(label, 0.0))})
    
    # Metadata rows
    meta_rows = [
        "Opening/Closing Date", "Revolving Credit Amount", "Available Credit",
        "Cash Access Line", "Available for Cash", "Past Due Amount",
        "Balance over the Credit Access Line"
    ]
    for label in meta_rows:
        items.append({"Type": label, "INSTANCES": "", "AMOUNT": f(cc_summary.get(label, 0.0)) if label != "Opening/Closing Date" else cc_summary.get(label, "N/A")})

    df = pd.DataFrame(items)
    
    # Account Number Metadata
    st.markdown(f'<div style="font-family: Arial, sans-serif; font-weight: 800; color: #004a99; margin-bottom: -15px; font-size: 16px;">Account Number: {cc_summary["Account Number"]}</div>', unsafe_allow_html=True)
    
    # Use the existing function for total consistency
    render_chase_table("ACCOUNT SUMMARY", df)

def render_chase_table(title, df):
    """Render a dataframe in the Chase bank statement style"""
    # (Imports moved to top level)
    
    # Build HTML table
    header_html = f'<div class="chase-header-container"><div class="chase-header-box">{title}</div><div class="chase-header-line"></div></div>'
    
    rows_html = ""
    for _, row in df.iterrows():
        is_ending_balance = str(row.get('Type', '')).strip().lower() == "ending balance"
        row_class = "chase-row-separator-top" if is_ending_balance else ""
        rows_html += f'<tr class="{row_class}">'
        for i, (col, val) in enumerate(row.items()):
            alignment_class = "chase-text-right" if i > 0 else ""
            rows_html += f'<td class="{alignment_class}">{val}</td>'
        rows_html += "</tr>"
        
    table_headers = "".join([f'<th class="{"chase-text-right" if i > 0 else ""}">{col}</th>' for i, col in enumerate(df.columns)])
    
    table_html = f'<div class="chase-container">{header_html}<table class="chase-table"><thead><tr>{table_headers}</tr></thead><tbody>{rows_html}</tbody></table></div>'
    
    st.markdown(table_html, unsafe_allow_html=True)

def render_chase_header(title):
    """Render just the Chase-style boxed header with line"""
    # (Imports moved to top level)
    st.markdown(f'<div class="chase-header-container" style="margin-top: 30px; margin-bottom: 20px;"><div class="chase-header-box">{title}</div><div class="chase-header-line"></div></div>', unsafe_allow_html=True)

# Global Injector for Chase Styles
def inject_chase_styles():
    # (Imports moved to top level)
    st.markdown("""
<style>
.chase-container {
    background-color: transparent;
    padding: 0px;
    margin-bottom: 20px;
    width: 100%;
}
.chase-header-container {
    display: flex;
    align-items: flex-end;
    margin-bottom: 15px;
}
.chase-header-box {
    border: 2px solid #004a99;
    padding: 6px 20px;
    font-weight: 800;
    font-size: 22px;
    text-transform: uppercase;
    color: #004a99 !important;
    white-space: nowrap;
    font-family: Arial, sans-serif;
}
.chase-header-line {
    flex-grow: 1;
    border-bottom: 2px solid #004a99;
    margin-bottom: 0px;
    margin-left: 0px;
}
.chase-table {
    width: auto;
    min-width: 600px;
    max-width: 100%;
    border-collapse: collapse;
    font-family: Arial, sans-serif;
    color: inherit;
}
.chase-table th {
    text-align: left;
    padding: 12px 20px 12px 5px;
    border-bottom: 2px solid #004a99;
    font-size: 14px;
    text-transform: uppercase;
    font-weight: 800;
    color: #004a99 !important;
}
.chase-table td {
    padding: 10px 20px 10px 5px;
    border-bottom: none;
    font-size: 15px;
    color: inherit;
}
.chase-row-separator-top td {
    border-top: 2px solid #004a99;
}
.chase-text-right {
    text-align: right !important;
}
</style>
""", unsafe_allow_html=True)

def get_sub_summary(transactions, opening_balance=0.0, raw_text="", bank_name=""):
    opening_balance = float(opening_balance or 0.0)  # guard against None

    # Try to extract Chase pre-formatted summary (most accurate)
    if raw_text:
        chase_summary = extract_chase_summary(raw_text)
        if chase_summary:
            df = pd.DataFrame([
                {"Type": k, "INSTANCES": v["count"], "AMOUNT": f"${v['amount']:,.2f}"}
                for k, v in chase_summary.items()
            ])
            return df

    # Try universal bank summary (BoA, BMO, Fifth Third, US Bank) ONLY if transactions are empty
    # If we have transactions, we use the fallback logic below which calculates from the list
    if not transactions and raw_text:
        universal_summary = extract_universal_bank_summary(raw_text)
        if universal_summary:
            df = pd.DataFrame([
                {"Type": k, "INSTANCES": v["count"], "AMOUNT": f"${v['amount']:,.2f}"}
                for k, v in universal_summary.items()
            ])
            return df

    # Credit Card Summary for AMEX
    if "amex" in str(bank_name).lower():
        data = {
            "Previous Balance": {"count": "", "amount": opening_balance},
            "Payments and Credits": {"count": 0, "amount": 0.0},
            "New Charges": {"count": 0, "amount": 0.0},
            "Fees": {"count": 0, "amount": 0.0},
            "New Balance": {"count": "", "amount": 0.0}
        }
        for t in transactions:
            amt = getattr(t, "amount", 0.0)
            cat = getattr(t, "category", "")
            if amt > 0: # Payment/Credit
                data["Payments and Credits"]["amount"] += amt
                data["Payments and Credits"]["count"] += 1
            else: # Charge/Fee
                if cat == "fee":
                    data["Fees"]["amount"] -= abs(amt)
                    data["Fees"]["count"] += 1
                else:
                    data["New Charges"]["amount"] -= abs(amt)
                    data["New Charges"]["count"] += 1
        
        closing = opening_balance + data["Payments and Credits"]["amount"] + data["New Charges"]["amount"] + data["Fees"]["amount"]
        data["New Balance"]["amount"] = closing
        
        return pd.DataFrame([
            {"Type": k, "INSTANCES": v["count"], "AMOUNT": f"${v['amount']:,.2f}"}
            for k, v in data.items()
        ])

    # Fallback: Calculate summary from transactions (for Checking accounts)
    # Initialize data structure to track count (instances) and amount
    data = {
        "Beginning Balance": {"count": "", "amount": opening_balance},
        "Deposits and Additions": {"count": 0, "amount": 0.0},
        "ATM & Debit Card Withdrawals": {"count": 0, "amount": 0.0},
        "Electronic Withdrawals": {"count": 0, "amount": 0.0},
        "Checks Paid": {"count": 0, "amount": 0.0},
        "Other Withdrawals": {"count": 0, "amount": 0.0},
        "Fees": {"count": 0, "amount": 0.0},
        "Ending Balance": {"count": "", "amount": 0.0}
    }

    if not transactions:
        # Return empty structure if no transactions
        df = pd.DataFrame([
            {"Type": k, "INSTANCES": v["count"], "AMOUNT": f"${v['amount']:,.2f}"}
            for k, v in data.items()
        ])
        return df

    # Track if we've skipped the opening balance deposit
    skipped_opening = False
    
    total_count = 0

    for t in transactions:
        # 💳 Skip credit card transactions for the bank summary table
        if getattr(t, "source", "") == "CREDIT_CARD":
            continue
            
        amt = getattr(t, "amount", 0.0)
        desc = (getattr(t, "description", "") or "").lower()
        vendor = (getattr(t, "vendor", "") or "").lower()

        if amt > 0:
            # If opening_balance is set and this is the first deposit matching it, skip it
            # to avoid double-counting (it's already shown as "Beginning Balance" row)
            if (opening_balance or 0.0) > 0 and not skipped_opening and abs(amt - (opening_balance or 0.0)) < 0.01:
                skipped_opening = True
                continue
            
            data["Deposits and Additions"]["amount"] += amt
            data["Deposits and Additions"]["count"] += 1
            total_count += 1
        else:
            abs_amt = abs(amt)
            # 1. Checks (high priority)
            if (getattr(t, "section", "") or "").upper() == "CHECKS" or "check" in vendor or "check" in desc:
                data["Checks Paid"]["amount"] -= abs_amt
                data["Checks Paid"]["count"] += 1
            
            # 2. ATM & Debit Card
            elif "atm" in desc or "debit card" in desc or "withdrawal" in desc:
                data["ATM & Debit Card Withdrawals"]["amount"] -= abs_amt
                data["ATM & Debit Card Withdrawals"]["count"] += 1
                
            # 3. Fees
            elif "fee" in desc or "service charge" in desc or "maintenance" in desc or "overdraft" in desc:
                data["Fees"]["amount"] -= abs_amt
                data["Fees"]["count"] += 1
                
            # 4. Electronic Withdrawals (Zelle, ACH, etc.)
            elif "zelle" in desc or "ach" in desc or "electronic" in desc or "vrs" in desc:
                data["Electronic Withdrawals"]["amount"] -= abs_amt
                data["Electronic Withdrawals"]["count"] += 1
            
            # 5. Other Withdrawals (Default for anything else)
            else:
                data["Other Withdrawals"]["amount"] -= abs_amt
                data["Other Withdrawals"]["count"] += 1
            
            total_count += 1

    # Calculation for Ending Balance:
    # Beginning Balance + Deposits - (Checks + ATM + Electronic + Fees)
    # Note: Withdrawals are already negative in the logic above, so we sum everything
    
    # Actually, simpler: Opening + Sum(All processed transaction amounts)
    # But let's follow the table logic to be precise with display values
    
    closing_bal = (
        data["Beginning Balance"]["amount"] +
        data["Deposits and Additions"]["amount"] +
        data["Checks Paid"]["amount"] +
        data["ATM & Debit Card Withdrawals"]["amount"] +
        data["Electronic Withdrawals"]["amount"] +
        data["Other Withdrawals"]["amount"] +
        data["Fees"]["amount"]
    )
    
    data["Ending Balance"]["amount"] = closing_bal
    data["Ending Balance"]["count"] = total_count

    # Create DataFrame with exact column names requested
    rows = []
    for label, values in data.items():
        rows.append({
            "Type": label,
            "INSTANCES": values["count"],
            "AMOUNT": f"${values['amount']:,.2f}"
            # Formatting Note: 
            # - Positive for Beginning, Deposits, Ending
            # - Negative for Checks, Withdrawals, Fees (handled by subtraction logic above)
        })

    df = pd.DataFrame(rows)
    # Reorder columns to match screenshot: [Type (Index-like), INSTANCES, AMOUNT]
    # But Streamlit displays index if we don't hide it. Let's make Type the first column.
    return df[["Type", "INSTANCES", "AMOUNT"]]

# ----------------------------
# Streamlit UI
def main():
    global _STATEMENT_YEAR
    # ----------------------------
    st.set_page_config(page_title="Bank Statement Analyzer (Hybrid)", layout="wide")
    inject_chase_styles()

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
        # extra Box
        # st.markdown("<div style='background: white; padding: 32px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.06);'>", unsafe_allow_html=True)

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
                    if st.button("❌ Yes, Delete", key="confirm_delete_yes", width='stretch'):
                        if delete_business(user_id, business_to_delete):
                            st.session_state.confirm_delete = None
                            st.success(f"Profile '{business_to_delete}' deleted successfully")
                            st.rerun()
                with col_b:
                    if st.button("Cancel", key="confirm_delete_no", width='stretch'):
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
                    if st.button("✅ Create", key="confirm_create", width='stretch'):
                        if new_business:
                            create_business(user_id, new_business)
                            st.session_state.active_business = new_business
                            st.session_state.show_create_form = False
                            st.rerun()
                        else:
                            st.error("Please enter a name")
                with col_b:
                    if st.button("❌ Cancel", key="cancel_create", width='stretch'):
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
    uploaded = st.file_uploader("Upload statement (PDF, CSV, DOCX)", type=["pdf", "csv", "doc", "docx"], accept_multiple_files=True)
    credit_card_files = st.file_uploader(
        "Upload Credit Card Statement(s) (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        key="credit_card_multi"
    )

    # Initialize custom rules in session state (load from file)

    if uploaded or credit_card_files:
        if uploaded:
            if isinstance(uploaded, list):
                st.info(f"Bank Statements: {len(uploaded)} files uploaded")
            else:
                st.info(f"Bank Statement: {uploaded.name} — {uploaded.size/1024:.1f} KB")
        if credit_card_files:
            st.info(f"Credit Card Statements: {len(credit_card_files)} files uploaded")

        # Year selection for uploaded files
        st.markdown("---")
        st.subheader("📅 Statement Year Selection")
        st.markdown("**Optional:** Manually specify the year for your statement(s) if automatic detection fails")

        # Generate year options (last 10 years)
        current_year = datetime.now().year
        year_options = ["Auto-detect"] + [str(y) for y in range(current_year, current_year - 10, -1)]

        # Store selected years in a dictionary
        selected_years = {}

        if uploaded:
            uploaded_list = uploaded if isinstance(uploaded, list) else [uploaded]
            st.markdown("**Bank Statement(s):**")
            for idx, file in enumerate(uploaded_list):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📄 {file.name}")
                with col2:
                    year_choice = st.selectbox(
                        "Year",
                        year_options,
                        key=f"year_bank_{idx}_{file.name}",
                        label_visibility="collapsed"
                    )
                    if year_choice != "Auto-detect":
                        selected_years[file.name] = int(year_choice)

        if credit_card_files:
            st.markdown("**Credit Card Statement(s):**")
            for idx, file in enumerate(credit_card_files):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"💳 {file.name}")
                with col2:
                    year_choice = st.selectbox(
                        "Year",
                        year_options,
                        key=f"year_cc_{idx}_{file.name}",
                        label_visibility="collapsed"
                    )
                    if year_choice != "Auto-detect":
                        selected_years[file.name] = int(year_choice)

        st.markdown("---")

        currency = st.selectbox("Currency", ["PKR", "USD", "EUR", "GBP", "AED", "CAD", "AUD"], index=1)

        # Store selected years in session state for use during parsing
        st.session_state.selected_years = selected_years

        include_opening_balance = False
        if uploaded:
            include_opening_balance = st.checkbox(
                "Include Opening Balance",
                value=False,
                help="Adds opening balance as a deposit before transactions",
                key="include_opening_balance"
            )

        # Check memos is now always ENABLED by request
        extract_check_memos = True

        if st.button("Process Statement"):
            with st.spinner("Parsing & processing..."):
                all_txs = []
                meta = {}
                all_raw_text = ""  # Store raw text for Chase summary extraction
                st.session_state.statement_summaries = [] # Store individual summaries
                st.session_state.cc_summaries = [] # Store list of credit card summaries

                # Process main bank statements if uploaded
                if uploaded:
                    uploaded_list = uploaded if isinstance(uploaded, list) else [uploaded]
                    dp = DocumentParser()
                    for up_file in uploaded_list:
                        file_bytes = up_file.read()
                        lines, ok, unreadable = dp.parse_document(file_bytes, up_file.name)
                        if not ok or len(lines) < 1:
                            st.error(f"Could not read text from bank statement file: {up_file.name}")
                            if unreadable:
                                st.warning(f"Unreadable pages in {up_file.name}: {unreadable}")
                        else:
                            # Store raw text for Chase summary extraction
                            file_raw_text = "\n".join(lines)
                            all_raw_text += file_raw_text + "\n"

                            # Check if user manually selected year for this file
                            manual_year = selected_years.get(up_file.name)
                            if manual_year:
                                st.info(f"✅ Using manually selected year for {up_file.name}: **{manual_year}**")

                            # Try new multi-bank parser first
                            bank_txs, b_meta = parse_with_multi_bank_parser(
                                lines,  # Pass extracted text lines, not bytes
                                up_file.name, 
                                manual_year=manual_year,
                                include_opening_balance=include_opening_balance,
                                extract_check_memos=extract_check_memos
                            )

                            # Fall back to FallbackStatementParser if multi-bank parser fails
                            if bank_txs is None or b_meta is None:
                                st.info(f"Using legacy parser for {up_file.name}")
                                fb_parser = FallbackStatementParser(
                                    include_opening_balance=include_opening_balance,
                                    extract_check_memos=extract_check_memos
                                )
                                bank_txs, b_meta = fb_parser.parse_statement(lines, manual_year=manual_year)
                            else:
                                st.success(f"✅ Detected bank: {b_meta.get('bank', 'Unknown')}")
                                st.write(f"Processed {len(bank_txs)} transactions (multi_bank_parser).")

                            all_txs.extend(bank_txs)

                            # Get opening balance from either parser
                            opening_bal = b_meta.get('opening_balance', 0.0) or 0.0
                            if include_opening_balance:
                                opening_bal = opening_bal  # respected from meta

                            # Generate summary for THIS file
                            file_summary_df = get_sub_summary(
                                transactions=bank_txs,
                                opening_balance=opening_bal,
                                raw_text=file_raw_text,
                                bank_name=b_meta.get("bank", "")
                            )
                            st.session_state.statement_summaries.append({
                                "filename": up_file.name,
                                "df": file_summary_df
                            })

                            if not meta:
                                meta = b_meta
                                # Always persist beginning_balance so UI can show it in deposits
                                st.session_state.opening_balance = opening_bal
                    st.session_state.meta = meta
                    st.session_state.raw_text = all_raw_text  # Save for summary extraction
                # Process credit card statements if uploaded
                if credit_card_files:
                    dp_cc = DocumentParser()
                    for cc_file in credit_card_files:
                        cc_file_bytes = cc_file.read()
                        cc_lines, ok, unreadable = dp_cc.parse_document(cc_file_bytes, cc_file.name)

                        if not ok or len(cc_lines) < 1:
                            st.error(f"Could not read text from credit card statement file: {cc_file.name}")
                            if unreadable:
                                st.warning(f"Unreadable pages in {cc_file.name}: {unreadable}")
                        else:
                            # Store raw text for extraction
                            cc_raw_text = "\n".join(cc_lines)
                            extracted_summary = extract_cc_summary(cc_raw_text)

                            st.session_state.cc_summaries.append({
                                "filename": cc_file.name,
                                "summary": extracted_summary
                            })

                            # Check if user manually selected year for this credit card file
                            cc_manual_year = selected_years.get(cc_file.name)
                            if cc_manual_year:
                                st.info(f"✅ Using manually selected year for {cc_file.name}: **{cc_manual_year}**")
                            elif not uploaded:
                                # If no bank statement was uploaded and no manual year, try to extract year from CC statement
                                fb_temp = FallbackStatementParser()
                                extracted_year = fb_temp._extract_statement_year(cc_lines)
                                if extracted_year:
                                    _STATEMENT_YEAR = extracted_year

                        cc_txs = CreditCardParser().parse(cc_lines)

                        # FORCE credit card as withdrawals
                        for tx in cc_txs:
                            tx.transaction_type = "withdrawal"
                            tx.amount = -abs(tx.amount)
                            tx.source = "CREDIT_CARD"

                        all_txs.extend(cc_txs)

                valid_dates = [
                    _md_key(tx.date)
                    for tx in all_txs
                    if _md_key(tx.date) is not None
                ]

                # If nothing extracted or too few rows, try UniversalParser conservative fallback
                # AMEX can sometimes be tricky for sectioned parser if layout is weird
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
                # (Imports moved to top level)
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

                # Show detected statement year if available
                if _STATEMENT_YEAR:
                    st.info(f"📅 Detected statement year: **{_STATEMENT_YEAR}** (automatically extracted from statement)")
                else:
                    st.warning("⚠️ Could not detect year from statement - using current year for dates without year")

                st.session_state.all_transactions = transactions
                st.session_state.filtered_transactions = transactions 

                # Reset all filter settings when processing new statement
                st.session_state.filter_active = False
                st.session_state.filter_locked = False
                st.session_state.filter_reset_count = 0  # Reset counter for new statement
                if 'filter_start_date' in st.session_state:
                    del st.session_state.filter_start_date
                if 'filter_end_date' in st.session_state:
                    del st.session_state.filter_end_date
                if 'filter_min_amount' in st.session_state:
                    del st.session_state.filter_min_amount
                if 'filter_max_amount' in st.session_state:
                    del st.session_state.filter_max_amount
                if 'filter_search_text' in st.session_state:
                    del st.session_state.filter_search_text
                if 'filter_account_index' in st.session_state:
                    del st.session_state.filter_account_index
                if 'filter_stats' in st.session_state:
                    del st.session_state.filter_stats
    # =========================================================================
    # PROFESSIONAL COMPREHENSIVE FILTER SYSTEM
    # =========================================================================
    # (Imports moved to top level)

    all_transactions = st.session_state.get("all_transactions", [])

    if all_transactions:
        # Always show sub-summary if transactions exist
        st.markdown("---")
        opening_balance_val = st.session_state.get("opening_balance", 0.0) or 0.0
        st.markdown("### Sub-summary / Transaction Breakdown")
        with st.expander("📊 Statement Summaries / Transaction Breakdowns", expanded=True):
            # Render Credit Card Summaries if available
            cc_summaries = st.session_state.get("cc_summaries", [])
            if cc_summaries:
                for cc_entry in cc_summaries:
                    st.markdown(f"**Credit Card Summary: {cc_entry['filename']}**")
                    render_cc_summary(cc_entry['summary'])

            summaries = st.session_state.get("statement_summaries", [])
            if summaries:
                for entry in summaries:
                    # Adaptive title  — pick bank from meta if available
                    bank_meta = st.session_state.get("meta", {})
                    bank_name_display = str(bank_meta.get("bank", "BANK")).upper()
                    render_chase_table(f"STATEMENT SUMMARY: {entry['filename']}", entry['df'])
            elif not cc_summaries:
                # Fallback for combined summary (only if no CC summary shown yet)
                bank_meta = st.session_state.get("meta", {})
                bank_name_display = str(bank_meta.get("bank", "BANK")).upper()
                is_credit_card = "amex" in bank_name_display.lower() or "credit" in bank_name_display.lower()
                summary_title = "CREDIT CARD SUMMARY" if is_credit_card else f"{bank_name_display} STATEMENT SUMMARY"
                sub_summary_df = get_sub_summary(
                    transactions=all_transactions,
                    opening_balance=opening_balance_val,
                    raw_text=st.session_state.get("raw_text", ""),
                    bank_name=bank_name_display
                )
                render_chase_table(summary_title, sub_summary_df)

        st.markdown("---")
        st.header("🔍 Advanced Transaction Filter")
        st.markdown("**Professional filtering system** - All filters work together to give you precise control")

        # Extract all valid dates with full date information
        parsed_dates = []
        for tx in all_transactions:
            if tx.date:
                try:
                    parsed_dates.append(datetime.strptime(tx.date, "%Y-%m-%d").date())
                except:
                    pass

        if not parsed_dates:
            st.warning("⚠️ No valid dates found in transactions.")
        else:
            # Get actual min and max dates from ALL uploaded statements (bank + credit card)
            min_date = min(parsed_dates)
            max_date = max(parsed_dates)

            # Calculate first day of starting month and last day of ending month
            first_day_of_start_month = min_date.replace(day=1)
            last_day_of_end_month = max_date.replace(day=monthrange(max_date.year, max_date.month)[1])

            # Store coverage dates in session state for reference
            st.session_state.statement_coverage_start = min_date
            st.session_state.statement_coverage_end = max_date

            # Display coverage information prominently
            st.info(
                f"📊 **Statement Coverage:** {min_date.strftime('%b %d, %Y')} → {max_date.strftime('%b %d, %Y')} "
                f"({(max_date - min_date).days} days, {len(all_transactions)} total transactions)"
            )

            # Initialize filter state if not present
            if 'filter_active' not in st.session_state:
                st.session_state.filter_active = False
                st.session_state.filter_locked = False

            # Initialize reset counter for forcing widget refresh
            if 'filter_reset_count' not in st.session_state:
                st.session_state.filter_reset_count = 0

            # Show lock status
            lock_col1, lock_col2 = st.columns([3, 1])
            with lock_col1:
                if st.session_state.get('filter_locked', False):
                    st.warning("🔒 **Filter is LOCKED** - Settings preserved for printing/exporting")
            with lock_col2:
                if st.session_state.get('filter_locked', False):
                    if st.button("🔓 Unlock Filter"):
                        st.session_state.filter_locked = False
                        st.success("Filter unlocked! You can now adjust settings.")
                        st.rerun()
                else:
                    if st.button("🔒 Lock Filter"):
                        st.session_state.filter_locked = True
                        st.success("Filter locked! Settings preserved for printing/exporting.")
                        st.rerun()

            # Disable filter controls if locked
            filter_disabled = st.session_state.get('filter_locked', False)

            # ===== DATE FILTER =====
            st.subheader("📅 Date Range")
            col_date1, col_date2 = st.columns(2)

            # Use reset counter in widget keys to force recreation on reset
            reset_suffix = f"_{st.session_state.filter_reset_count}"

            with col_date1:
                start_md = st.date_input(
                    "Start Date",
                    value=first_day_of_start_month,
                    min_value=first_day_of_start_month,
                    max_value=last_day_of_end_month,
                    key=f"filter_start_md{reset_suffix}",
                    disabled=filter_disabled
                )

            with col_date2:
                end_md = st.date_input(
                    "End Date",
                    value=last_day_of_end_month,
                    min_value=first_day_of_start_month,
                    max_value=last_day_of_end_month,
                    key=f"filter_end_md{reset_suffix}",
                    disabled=filter_disabled
                )

            # ===== AMOUNT FILTER =====
            st.subheader("💰 Amount Range")
            col_amt1, col_amt2 = st.columns(2)

            # Get min/max amounts from transactions
            all_amounts = [abs(tx.amount) for tx in all_transactions if tx.amount]
            min_amount_possible = min(all_amounts) if all_amounts else 0.0
            max_amount_possible = max(all_amounts) if all_amounts else 10000.0

            with col_amt1:
                min_amount_filter = st.number_input(
                    "Minimum Amount ($)",
                    min_value=0.0,
                    max_value=max_amount_possible,
                    value=st.session_state.get('filter_min_amount', 0.0),
                    step=10.0,
                    key="filter_min_amt",
                    disabled=filter_disabled
                )

            with col_amt2:
                max_amount_filter = st.number_input(
                    "Maximum Amount ($)",
                    min_value=0.0,
                    max_value=max_amount_possible * 2,  # Allow some overhead
                    value=st.session_state.get('filter_max_amount', max_amount_possible),
                    step=10.0,
                    key="filter_max_amt",
                    disabled=filter_disabled
                )

            # ===== VENDOR/KEYWORD SEARCH =====
            st.subheader("🔎 Search by Vendor or Keyword")
            search_text = st.text_input(
                "Enter vendor name or keyword (searches in vendor, description, and transaction details)",
                value=st.session_state.get('filter_search_text', ""),
                placeholder="e.g., Amazon, Stripe, Check, etc.",
                key="filter_search",
                disabled=filter_disabled
            )

            # ===== ACCOUNT CODE FILTER =====
            st.subheader("📊 Account Code Filter")

            # Load all account codes
            # (Imports moved to top level)
            all_account_options = {"All Account Codes": "ALL"}
            account_file = Path(__file__).parent / 'account_keywords.json'
            try:
                with open(account_file, 'r') as f:
                    data = json.load(f)
                    for acc_code, acc_details in sorted(data.items()):
                        all_account_options[f"{acc_code} · {acc_details['name']}"] = acc_code
            except Exception as e:
                st.warning(f"⚠️ Could not load account codes: {e}")

            account_code_filter = st.selectbox(
                "Filter by Account Code:",
                options=list(all_account_options.keys()),
                index=st.session_state.get('filter_account_index', 0),
                key="filter_account_code",
                disabled=filter_disabled
            )

            # ===== APPLY FILTER BUTTON =====
            st.markdown("---")
            col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 2])

            with col_btn1:
                if st.button("✅ Apply Filter", type="primary", disabled=filter_disabled, width='stretch'):
                    # Store filter settings in session state
                    st.session_state.filter_start_date = start_md
                    st.session_state.filter_end_date = end_md
                    st.session_state.filter_min_amount = min_amount_filter
                    st.session_state.filter_max_amount = max_amount_filter
                    st.session_state.filter_search_text = search_text
                    st.session_state.filter_account_index = list(all_account_options.keys()).index(account_code_filter)

                    # Apply all filters
                    filtered = []
                    filter_stats = {
                        'date_filtered': 0,
                        'amount_filtered': 0,
                        'search_filtered': 0,
                        'account_filtered': 0
                    }

                    for tx in all_transactions:
                        # DATE FILTER
                        if tx.date:
                            try:
                                tx_date = datetime.strptime(tx.date, "%Y-%m-%d").date()
                                if not (start_md <= tx_date <= end_md):
                                    filter_stats['date_filtered'] += 1
                                    continue
                            except:
                                continue

                        # AMOUNT FILTER
                        tx_abs_amount = abs(tx.amount)
                        if tx_abs_amount < min_amount_filter or tx_abs_amount > max_amount_filter:
                            filter_stats['amount_filtered'] += 1
                            continue

                        # SEARCH FILTER
                        if search_text.strip():
                            search_lower = search_text.lower().strip()
                            searchable_text = f"{tx.vendor} {tx.description} {tx.raw_line}".lower()
                            if search_lower not in searchable_text:
                                filter_stats['search_filtered'] += 1
                                continue

                        # ACCOUNT CODE FILTER
                        if account_code_filter != "All Account Codes":
                            selected_account_code = all_account_options[account_code_filter]
                            tx_account_code = getattr(tx, 'account_code', None)
                            if tx_account_code != selected_account_code:
                                filter_stats['account_filtered'] += 1
                                continue

                        # Transaction passed all filters
                        filtered.append(tx)

                    st.session_state.filtered_transactions = filtered
                    st.session_state.filter_active = True
                    st.session_state.filter_stats = filter_stats

                    # Calculate totals for success message
                    filtered_active_msg = [t for t in filtered if not getattr(t, "is_excluded", False)]
                    msg_deposits = sum(t.amount for t in filtered_active_msg if t.amount > 0)
                    msg_withdrawals = sum(abs(t.amount) for t in filtered_active_msg if t.amount < 0)
                    msg_deposits_count = len([t for t in filtered_active_msg if t.amount > 0])
                    msg_withdrawals_count = len([t for t in filtered_active_msg if t.amount < 0])

                    # Success message with details
                    total_filtered_out = len(all_transactions) - len(filtered)
                    st.success(
                        f"✅ **Filter Applied Successfully!**\n\n"
                        f"📊 Showing **{len(filtered)}** out of **{len(all_transactions)}** transactions\n\n"
                        f"🗓️ Date Range: {start_md.strftime('%b %d, %Y')} → {end_md.strftime('%b %d, %Y')}\n\n"
                        f"---\n\n"
                        f"💰 Deposits: **${msg_deposits:,.2f}** ({msg_deposits_count} tx)\n\n"
                        f"💸 Withdrawals: **${msg_withdrawals:,.2f}** ({msg_withdrawals_count} tx)\n\n"
                        f"📊 Net Income: **${msg_deposits - msg_withdrawals:,.2f}**"
                    )

                    # Show filter breakdown
                    if total_filtered_out > 0:
                        with st.expander("📋 Filter Breakdown"):
                            if filter_stats['date_filtered'] > 0:
                                st.write(f"• Date filter removed: {filter_stats['date_filtered']} transactions")
                            if filter_stats['amount_filtered'] > 0:
                                st.write(f"• Amount filter removed: {filter_stats['amount_filtered']} transactions")
                            if filter_stats['search_filtered'] > 0:
                                st.write(f"• Search filter removed: {filter_stats['search_filtered']} transactions")
                            if filter_stats['account_filtered'] > 0:
                                st.write(f"• Account code filter removed: {filter_stats['account_filtered']} transactions")

                    st.rerun()

            with col_btn2:
                if st.button("🔄 Reset Filter", disabled=filter_disabled, width='stretch'):
                    st.session_state.filtered_transactions = all_transactions
                    st.session_state.filter_active = False
                    # Increment reset counter to force widget recreation with default values
                    st.session_state.filter_reset_count += 1
                    # Clear filter state keys
                    if 'filter_start_date' in st.session_state:
                        del st.session_state.filter_start_date
                    if 'filter_end_date' in st.session_state:
                        del st.session_state.filter_end_date
                    if 'filter_min_amount' in st.session_state:
                        del st.session_state.filter_min_amount
                    if 'filter_max_amount' in st.session_state:
                        del st.session_state.filter_max_amount
                    if 'filter_search_text' in st.session_state:
                        del st.session_state.filter_search_text
                    if 'filter_account_index' in st.session_state:
                        del st.session_state.filter_account_index
                    if 'filter_stats' in st.session_state:
                        del st.session_state.filter_stats
                    st.info("🔄 Filter reset. Showing all transactions.")
                    st.rerun()

            with col_btn3:
                if st.session_state.get('filter_active', False):
                    st.metric("Filtered", f"{len(st.session_state.get('filtered_transactions', []))} tx")
                else:
                    st.metric("Total", f"{len(all_transactions)} tx")

            # Show active filter summary
            if st.session_state.get('filter_active', False):
                st.markdown("---")

                # Calculate totals for the summary
                filtered_txs_sum = st.session_state.get('filtered_transactions', [])
                filtered_active_sum = [t for t in filtered_txs_sum if not getattr(t, "is_excluded", False)]
                summary_deposits = sum(t.amount for t in filtered_active_sum if t.amount > 0)
                summary_withdrawals = sum(abs(t.amount) for t in filtered_active_sum if t.amount < 0)
                summary_deposits_count = len([t for t in filtered_active_sum if t.amount > 0])
                summary_withdrawals_count = len([t for t in filtered_active_sum if t.amount < 0])

                st.info(
                    f"🔍 **Active Filters:**\n\n"
                    f"📅 Dates: {st.session_state.filter_start_date.strftime('%b %d, %Y')} - {st.session_state.filter_end_date.strftime('%b %d, %Y')}\n\n"
                    f" Amount: ${st.session_state.filter_min_amount:,.2f} - ${st.session_state.filter_max_amount:,.2f}" +
                    (f"\n\n🔎 Search: '{st.session_state.filter_search_text}'" if st.session_state.filter_search_text else "") +
                    (f"\n\n📊 Account: {account_code_filter}" if account_code_filter != "All Account Codes" else "") +
                    f"\n\n---\n\n"
                    f"**📈 Filtered Results:**\n\n"
                    f"💰 Total Deposits: ${summary_deposits:,.2f} ({summary_deposits_count} transactions)\n\n"
                    f"💸 Total Withdrawals: ${summary_withdrawals:,.2f} ({summary_withdrawals_count} transactions)\n\n"
                    f"📊 Net Income: ${summary_deposits - summary_withdrawals:,.2f}"
                )


    # Dashboard (same UI as before)
    if "transactions" in st.session_state and st.session_state.transactions:
        # ============================================
        # FILTER STATUS BANNER (Always visible at top)
        # ============================================
        if st.session_state.get('filter_active', False):
            # Calculate filtered totals for banner
            filtered_txs = st.session_state.get('filtered_transactions', [])
            filtered_active = [t for t in filtered_txs if not getattr(t, "is_excluded", False)]
            filtered_deposits = sum(t.amount for t in filtered_active if t.amount > 0)
            filtered_withdrawals = sum(abs(t.amount) for t in filtered_active if t.amount < 0)

            filter_banner_cols = st.columns([4, 1])
            with filter_banner_cols[0]:
                lock_text = " | 🔒 LOCKED" if st.session_state.get('filter_locked', False) else ""
                banner_html = f"""
                    <div style="background-color: #4CAF50; color: white; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                        <h3 style="margin: 0; color: white;">🔍 FILTER ACTIVE</h3>
                        <p style="margin: 5px 0 0 0; color: white;">
                            Showing <strong>{len(st.session_state.get('filtered_transactions', []))}</strong> of <strong>{len(st.session_state.get('all_transactions', []))}</strong> transactions | 
                            {st.session_state.get('filter_start_date', date.today()).strftime('%b %d, %Y')} to {st.session_state.get('filter_end_date', date.today()).strftime('%b %d, %Y')}{lock_text}
                        </p>
                        <p style="margin: 5px 0 0 0; color: white; font-size: 0.9em;">
                            💰 Deposits: <strong>${filtered_deposits:,.2f}</strong> | 
                            💸 Withdrawals: <strong>${filtered_withdrawals:,.2f}</strong> | 
                            📊 Net: <strong>${filtered_deposits - filtered_withdrawals:,.2f}</strong>
                        </p>
                    </div>
                """
                st.markdown(banner_html, unsafe_allow_html=True)
            with filter_banner_cols[1]:
                if st.button("📊 View All", key="view_all_banner"):
                    st.session_state.filter_active = False
                    st.session_state.filtered_transactions = st.session_state.all_transactions
                    st.rerun()
        else:
            # Calculate totals for all transactions
            all_txs = st.session_state.get('all_transactions', [])
            all_active = [t for t in all_txs if not getattr(t, "is_excluded", False)]
            all_deposits = sum(t.amount for t in all_active if t.amount > 0)
            all_withdrawals = sum(abs(t.amount) for t in all_active if t.amount < 0)

            banner_html = f"""
                <div style="background-color: #2196F3; color: white; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                    <h3 style="margin: 0; color: white;">📊 ALL TRANSACTIONS</h3>
                    <p style="margin: 5px 0 0 0; color: white;">
                        Showing all <strong>{len(all_txs)}</strong> transactions from uploaded statements
                    </p>
                    <p style="margin: 5px 0 0 0; color: white; font-size: 0.9em;">
                        💰 Deposits: <strong>${all_deposits:,.2f}</strong> | 
                        💸 Withdrawals: <strong>${all_withdrawals:,.2f}</strong> | 
                        📊 Net: <strong>${all_deposits - all_withdrawals:,.2f}</strong>
                    </p>
                </div>
            """
            st.markdown(banner_html, unsafe_allow_html=True)

        transactions: List[Transaction] = get_active_transactions()


        rg = ReportGenerator()
        stats = rg.generate_summary_statistics(transactions)
        cur = st.session_state.currency

        # Get the opening balance setting and value
        include_ob = st.session_state.get('include_opening_balance', False)
        ob_val = st.session_state.get('opening_balance', 0.0) or 0.0

        # ALWAYS calculate from filtered transactions to respect date filter
        computed_deposits = sum(t.amount for t in transactions if t.amount > 0 and not is_tx_excluded(t))
        computed_withdrawals = sum(-t.amount for t in transactions if t.amount < 0 and not is_tx_excluded(t))

        # Include opening balance in deposits total if available
        deposits_total_display = computed_deposits + ob_val

        # Update stats with filtered transaction data
        stats['Total Deposit Amount'] = float(deposits_total_display)
        stats['Total Withdrawal Amount'] = float(computed_withdrawals)
        stats['Net Income'] = float(deposits_total_display - computed_withdrawals)

        # Count transactions from filtered set
        stats['Total Deposits'] = len([t for t in transactions if t.amount > 0 and not is_tx_excluded(t)])
        stats['Total Withdrawals'] = len([t for t in transactions if t.amount < 0 and not is_tx_excluded(t)])
        stats['Total Transactions'] = len(transactions)

        # Now display metrics with the filtered stats
        render_chase_header("SUMMARY")

        # Show filter status in header if active
        if st.session_state.get('filter_active', False):
            st.caption(f"📊 Statistics based on filtered data ({len(transactions)} transactions)")
        else:
            st.caption(f"📊 Statistics based on all transactions ({len(transactions)} transactions)")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Deposits",
            f"{cur} {stats['Total Deposit Amount']:,.2f}",
            f"+{stats['Total Deposits']} tx"
        )
        c2.metric(
            "Total Withdrawals",
            f"{cur} {stats['Total Withdrawal Amount']:,.2f}",
            f"+{stats['Total Withdrawals']} tx"
        )
        c3.metric(
            "Net Income",
            f"{cur} {stats['Net Income']:,.2f}"
        )
        c4.metric(
            "Transactions",
            stats['Total Transactions']
        )





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

        def clean_vendor_display(name):
            n = name.split('*')[0].split('.')[0].strip().upper()
            # Common name maps
            if "AMAZON" in n: return "AMAZON"
            if "CHASE" in n: return "CHASE"
            if "STRIPE" in n: return "STRIPE"
            if "UPWORK" in n: return "UPWORK"
            if "FIDELITY" in n: return "FIDELITY"
            return n or name

        rg = ReportGenerator()

        if selected_tab == 0:  # Deposits tab
            st.subheader("All Deposits Summary (by Source/Vendor)")
            # Regenerate deposits summary with filtered transactions
            df = rg.generate_deposits_summary(transactions)
            if df is None or df.empty:
                st.info("No deposits found.")
            else:
                st.dataframe(df, width='stretch', hide_index=True)
                st.subheader("👉 Customer Transaction Details")

                # Show opening balance at the top if available
                if ob_val and ob_val > 0:
                    with st.expander(f"🏛️ Opening Balance — {cur} {ob_val:,.2f}", expanded=False):
                        st.markdown(
                            f"| Field | Value |\n"
                            f"|---|---|\n"
                            f"| **Opening Balance** | {cur} {ob_val:,.2f} |\n"
                            f"| **Source** | Bank Statement Beginning Balance |"
                        )

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
                    display_vendor = clean_vendor_display(vendor)
                    with st.expander(f"💰 {display_vendor}  ·  {cur} {subtotal:,.2f}  ({cnt} tx)"):
                        details = pd.DataFrame([{
                            "Date": f"~~{it.date}~~" if is_tx_excluded(it) else it.date,
                            "Amount": f"{cur} {it.amount:,.2f}",
                            "Description": f"~~{it.description}~~" if is_tx_excluded(it) else it.description,
                            "Needs Review": "⚠ Yes" if it.needs_review else "✅ No",
                            "Status": "🚫 Excluded" if is_tx_excluded(it) else "✅ Active"
                        } for it in items])
                        st.dataframe(details, width='stretch', hide_index=True)

        elif selected_tab == 1:  # Withdrawals tab
            st.subheader("All Withdrawals Summary (by Vendor)")
            # Regenerate withdrawals summary with filtered transactions
            df = rg.generate_withdrawals_summary(transactions)
            if df is None or df.empty:
                st.info("No withdrawals found.")
            else:
                st.dataframe(df, width='stretch', hide_index=True)
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
                    display_vendor = clean_vendor_display(vendor)
                    with st.expander(f"💸 {display_vendor}  ·  {cur} {subtotal:,.2f}  ({cnt} tx)"):
                        details = pd.DataFrame([{
                            "Date": f"~~{it.date}~~" if is_tx_excluded(it) else it.date,
                            "Amount": f"{cur} {abs(it.amount):,.2f}",
                            "Description": f"~~{it.description}~~" if is_tx_excluded(it) else it.description,
                            "Needs Review": "⚠ Yes" if it.needs_review else "✅ No",
                            "Status": "🚫 Excluded" if is_tx_excluded(it) else "✅ Active"
                        } for it in items])
                        st.dataframe(details, width='stretch', hide_index=True)

        elif selected_tab == 2:  # P&L tab
            st.subheader("Profit & Loss")
            # Regenerate P&L with filtered transactions
            filtered_pl_df = rg.generate_pl_report(get_active_transactions())
            st.dataframe(filtered_pl_df, width='stretch', hide_index=True)

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
            st.dataframe(all_df, width='stretch', hide_index=True)

        elif SHOW_SCHEDULE_C and selected_tab == 4:  # Schedule C tab
                st.subheader("📄 Schedule C")

                schedule_c_df = st.session_state.get("schedule_c_df")

                if schedule_c_df is None or schedule_c_df.empty:
                    st.info("No Schedule C data available.")
                else:
                    categorized_transactions = st.session_state.get("categorized_transactions", [])
                    transactions = st.session_state.get("transactions", [])

                    # from datetime import datetime
                    # import pandas as pd
                    # import re

                    cur = "USD"

                    # ===============================
                    # IRS SCHEDULE C VIEW
                    # ===============================
                    # from schedule_c_categorizer import ScheduleCCategorizer
                    sc_categorizer = ScheduleCCategorizer()

                    # ---- Generate Schedule C report
                    schedule_c_text = sc_categorizer.generate_schedule_c_report(categorized_transactions)
                    st.subheader("📄 IRS Schedule C Report")
                    st.code(schedule_c_text)

                    st.subheader("🧾 IRS Schedule C Summary")
                    st.dataframe(schedule_c_df, width='stretch', hide_index=True)

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

                            st.dataframe(df, width='stretch', hide_index=True)

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
                # from datetime import datetime
                # import pandas as pd
                # from schedule_c_categorizer import ScheduleCCategorizer

                sc_categorizer = ScheduleCCategorizer()
                cur = "USD"

                # ---- Robust period detection from date filter or min → max date
                # Check if user has applied a date filter
                if st.session_state.get('filter_active', False) and 'filter_start_date' in st.session_state and 'filter_end_date' in st.session_state:
                    # Use the EXACT filtered dates (don't expand to full months when filter is active)
                    start = st.session_state.filter_start_date
                    end = st.session_state.filter_end_date
                    # Format: "Jan 15, 2025 - Mar 20, 2025" (exact dates from filter)
                    period_input = f"{start.strftime('%b %d, %Y')} - {end.strftime('%b %d, %Y')}"
                elif "filter_start_md" in st.session_state and "filter_end_md" in st.session_state:
                    # Use the date picker values (when filter UI is present but not applied)
                    start = st.session_state.filter_start_md
                    end = st.session_state.filter_end_md
                    # Get first day of starting month and last day of ending month
                    # from calendar import monthrange
                    first_day_of_month = start.replace(day=1)
                    last_day_of_month = end.replace(day=monthrange(end.year, end.month)[1])
                    # Format: "Jan 01, 2025 - Dec 31, 2025"
                    period_input = f"{first_day_of_month.strftime('%b %d, %Y')} - {last_day_of_month.strftime('%b %d, %Y')}"
                else:
                    # Fall back to detecting from transactions
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
                        # Get first day of starting month and last day of ending month
                        # (Imports moved to top level)
                        first_day_of_month = start.replace(day=1)
                        last_day_of_month = end.replace(day=monthrange(end.year, end.month)[1])
                        period_input = f"{first_day_of_month.strftime('%b %d, %Y')} - {last_day_of_month.strftime('%b %d, %Y')}"
                    else:
                        period_input = datetime.now().strftime("%B %Y")

                # Get business name from active profile
                business_name = st.session_state.get("active_business", "")

                # ---- Filter out excluded transactions for P&L statement generation
                # from account_code_mapper import AccountCodeMapper

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
                        # (Imports moved to top level)
                        # account_file = Path(__file__).parent / 'account_keywords.json'
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
                    st.dataframe(summary_df, width='stretch', hide_index=True)

                    # Group by account code
                    st.subheader("💼 By Account Code")
                    grouped = {}
                    for tx, cat in categorized_transactions:
                        # Include all transactions in grouping

                        # Check if account_code is already set on transaction (from manual reassignment)
                        if hasattr(tx, 'account_code') and tx.account_code:
                            account_code = tx.account_code
                            # Get the account name from JSON
                            # (Imports moved to top level)
                            # account_file = Path(__file__).parent / 'account_keywords.json'
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
                    # import json
                    # from pathlib import Path
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
            st.subheader("⚙️ Automatic Transaction Sorting & Categorization")
            st.markdown("""
            **Set up automatic filtering rules ONCE** - they'll automatically sort transactions into the right categories for all future uploads.
            Configure your sorting parameters below and the system will remember them.
            """)

            # Load all account codes for dropdown
            # (Imports moved to top level)
            all_account_options = {}
            account_file = Path(__file__).parent / 'account_keywords.json'
            try:
                with open(account_file, 'r') as f:
                    data = json.load(f)
                    for acc_code, acc_details in data.items():
                        all_account_options[f"{acc_code} · {acc_details['name']}"] = acc_code
            except Exception as e:
                st.error(f"Error loading account codes: {e}")

            # ==========================================
            # MAIN RULE INPUT FORM
            # ==========================================
            st.subheader("➕ Setup New Rule")

            col_adv1, col_adv2 = st.columns([2, 2])

            with col_adv1:
                rule_keyword_adv = st.text_input(
                    "Primary Keyword *",
                    placeholder="e.g., Check, Stripe",
                    key="keyword_adv"
                )

            with col_adv2:
                rule_account_adv = st.selectbox(
                    "Target Account Code *",
                    options=list(all_account_options.keys()),
                        key="account_adv"
                    )

                st.markdown("**Filtering Parameters:**")

                col_adv_amt1, col_adv_amt2 = st.columns(2)
                with col_adv_amt1:
                    min_amount_adv = st.number_input(
                        "Min Amount", min_value=0.0, value=0.0, step=10.0, key="min_adv"
                    )
                with col_adv_amt2:
                    max_amount_adv = st.number_input(
                        "Max Amount", min_value=0.0, value=0.0, step=10.0, key="max_adv"
                    )

                st.markdown("**Pattern Matching:**")

                col_adv_pri, col_adv_add = st.columns(2)
                with col_adv_pri:
                    priority_adv = st.number_input(
                        "Priority (1=highest, 999=lowest)",
                        min_value=1, max_value=999, value=999, step=1, key="priority_adv"
                    )

                with col_adv_add:
                    pass  # Placeholder for alignment

                additional_keywords_adv = st.text_input(
                    "Additional Keywords (ALL must match, comma-separated)",
                    placeholder="e.g., rent, payment",
                    key="additional_adv"
                )

                exclude_keywords_adv = st.text_input(
                    "Exclusion Keywords (skip if ANY match, comma-separated)",
                    placeholder="e.g., refund, dispute",
                    key="exclude_adv"
                )

            # ==========================================
            # SUBMIT RULE
            # ==========================================
            st.markdown("---")

            if st.button("Create Rule", key="add_rule_btn", type="primary"):
                # Use advanced form inputs
                keyword = rule_keyword_adv.strip()
                account = rule_account_adv
                min_amt = min_amount_adv
                max_amt = max_amount_adv
                add_kws = additional_keywords_adv
                excl_kws = exclude_keywords_adv
                priority = priority_adv

                if keyword:
                        account_code = all_account_options[account]

                        # Build the new rule
                        new_rule = {
                            'keyword': keyword,
                            'account_code': account_code,
                            'account_display': account
                        }

                        # Add optional fields only if they have meaningful values
                        if min_amt > 0:
                            new_rule['min_amount'] = min_amt
                        if max_amt > 0:
                            new_rule['max_amount'] = max_amt
                        if priority != 999:
                            new_rule['priority'] = priority
                        if add_kws.strip():
                            new_rule['additional_keywords'] = [kw.strip() for kw in add_kws.split(',') if kw.strip()]
                        if excl_kws.strip():
                            new_rule['exclude_keywords'] = [kw.strip() for kw in excl_kws.split(',') if kw.strip()]

                        st.session_state.custom_rules.append(new_rule)
                        save_business_rules(
                            st.session_state.user["id"],
                            st.session_state.active_business,
                            st.session_state.custom_rules
                        )

                        # Reapply rules to existing transactions
                        reapply_custom_rules()

                        # Create success message with rule details
                        msg = f"✅ Rule created: '{keyword}' → {account}"
                        if min_amt > 0 or max_amt > 0:
                            amount_filter = []
                            if min_amt > 0:
                                amount_filter.append(f"${min_amt:.2f}+")
                            if max_amt > 0:
                                amount_filter.append(f"up to ${max_amt:.2f}")
                            msg += f" ({', '.join(amount_filter)})"

                        st.success(msg)
                        st.rerun()
                else:
                    st.error("❌ Please enter a keyword to search for")

            # Display existing rules
            st.subheader("📋 Active Rules")
            if st.session_state.custom_rules:
                # Sort rules by priority for display
                sorted_rules = sorted(st.session_state.custom_rules, key=lambda r: r.get('priority', 999))
                st.info(f"Total rules: {len(st.session_state.custom_rules)} (sorted by priority)")

                for idx, rule in enumerate(sorted_rules):
                    # Find original index for deletion
                    orig_idx = st.session_state.custom_rules.index(rule)

                    # Build display label
                    label_parts = [f"🔍 {rule['keyword']}"]

                    # Add amount range if present
                    if rule.get('min_amount') or rule.get('max_amount'):
                        amount_range = []
                        if rule.get('min_amount'):
                            amount_range.append(f"≥ ${rule['min_amount']:.0f}")
                        if rule.get('max_amount'):
                            amount_range.append(f"≤ ${rule['max_amount']:.0f}")
                        label_parts.append(f"[{' & '.join(amount_range)}]")

                    # Add priority if not default
                    if rule.get('priority', 999) != 999:
                        label_parts.append(f"[Priority: {rule['priority']}]")

                    label = " ".join(label_parts)

                    with st.expander(label):
                        col1, col2 = st.columns([4, 1])

                        with col1:
                            st.markdown(f"**→ {rule['account_display']}**")

                            # Show filters if present
                            filters = []
                            if rule.get('min_amount'):
                                filters.append(f"Min Amount: ${rule['min_amount']:.2f}")
                            if rule.get('max_amount'):
                                filters.append(f"Max Amount: ${rule['max_amount']:.2f}")
                            if rule.get('priority', 999) != 999:
                                filters.append(f"Priority: {rule['priority']}")
                            if rule.get('additional_keywords'):
                                filters.append(f"Additional Keywords: {', '.join(rule['additional_keywords'])}")
                            if rule.get('exclude_keywords'):
                                filters.append(f"Exclusion Keywords: {', '.join(rule['exclude_keywords'])}")

                            if filters:
                                st.markdown("**Filters:**")
                                for f in filters:
                                    st.markdown(f"- {f}")

                        with col2:
                            if st.button("🗑️ Delete", key=f"delete_rule_{orig_idx}"):
                                st.session_state.custom_rules.pop(orig_idx)

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
            st.markdown("**💡 Enhanced Features:**")
            st.markdown("- **Amount Filters**: Categorize same transaction types differently based on amount")
            st.markdown("- **Priority Control**: Lower priority numbers are processed first (1 is highest)")
            st.markdown("- **Pattern Matching**: Require multiple keywords (additional) or exclude specific ones")
            st.markdown("- **Auto-Apply**: Rules are applied immediately to all current and future transactions")
            st.markdown("- **Persistent**: Rules are saved automatically and persist across sessions")

            st.markdown("---")
            st.markdown("**📖 Example Use Cases:**")
            st.markdown("1. **ATM by Amount**: Large ATMs (≥$200) → Temp Help, Small ATMs (<$200) → Other Expenses")
            st.markdown("2. **Check by Amount**: Checks ≥$1000 → Salaries, <$1000 → Supplies")
            st.markdown("3. **Pattern Match**: 'payment' + 'rent' keywords → Rent category")
            st.markdown("4. **Exclusions**: 'stripe' transactions except 'refund' → Sales")

        # Downloads
        st.markdown("---")
        st.header("📥 Download & Print Reports")

        # Show filter status in download section
        if st.session_state.get('filter_active', False):
            filter_info = f"""
            **🔍 Active Filter Applied to Exports:**
            - Date Range: {st.session_state.get('filter_start_date', 'N/A').strftime('%b %d, %Y')} - {st.session_state.get('filter_end_date', 'N/A').strftime('%b %d, %Y')}
            - Showing {len(st.session_state.get('filtered_transactions', []))} of {len(st.session_state.get('all_transactions', []))} transactions
            """
            st.info(filter_info)
            if st.session_state.get('filter_locked', False):
                st.success("🔒 Filter is LOCKED - All exports and prints will use these filtered results")

        # Print button
        st.markdown("""
            <style>
            @media print {
                .stButton, .stFileUploader, .stSelectbox, .stTextInput, .stNumberInput, .stDateInput {
                    display: none !important;
                }
                .filter-status {
                    border: 2px solid #4CAF50;
                    padding: 10px;
                    margin: 10px 0;
                    background-color: #f0f0f0;
                    page-break-inside: avoid;
                }
            }
            </style>
        """, unsafe_allow_html=True)

        if st.button("🖨️ Print Current View", type="secondary", width='stretch'):
            st.markdown('<script>window.print();</script>', unsafe_allow_html=True)
            st.info("💡 Print dialog should open. The current filter settings will be preserved in the printout.")

        st.markdown("---")
        st.subheader("Download Files")

        c1,c2,c3,c4 = st.columns(4)

        # Get the appropriate transactions for export (filtered if active)
        export_transactions = get_active_transactions()
        rg_export = ReportGenerator()

        with c1:
            # Regenerate deposits with current filter
            export_deposits_df = rg_export.generate_deposits_summary(export_transactions)
            dep_csv = export_deposits_df.to_csv(index=False) if (export_deposits_df is not None and not export_deposits_df.empty) else ""
            filename_suffix = "_filtered" if st.session_state.get('filter_active', False) else ""
            st.download_button("⬇ Deposits CSV", dep_csv, f"deposits{filename_suffix}.csv", mime="text/csv")

        with c2:
            # Regenerate withdrawals with current filter
            export_withdrawals_df = rg_export.generate_withdrawals_summary(export_transactions)
            wd_csv = export_withdrawals_df.to_csv(index=False) if (export_withdrawals_df is not None and not export_withdrawals_df.empty) else ""
            filename_suffix = "_filtered" if st.session_state.get('filter_active', False) else ""
            st.download_button("⬇ Withdrawals CSV", wd_csv, f"withdrawals{filename_suffix}.csv", mime="text/csv")

        with c3:
            # Regenerate P&L with current filter
            export_pl_df = rg_export.generate_pl_report(export_transactions)
            pnl_csv = export_pl_df.to_csv(index=False) if (export_pl_df is not None and not export_pl_df.empty) else ""
            filename_suffix = "_filtered" if st.session_state.get('filter_active', False) else ""
            st.download_button("⬇ P&L CSV", pnl_csv, f"pnl{filename_suffix}.csv", mime="text/csv")

        with c4:
            pl_statement_text = st.session_state.get("pl_statement_text", "")
            filename_suffix = "_filtered" if st.session_state.get('filter_active', False) else ""
            st.download_button("⬇ P&L Statement", pl_statement_text, f"pl_statement{filename_suffix}.txt", mime="text/plain")

        st.success("✅ Report generated. Verify totals against your bank statement.")

if __name__ == "__main__":
    main()
