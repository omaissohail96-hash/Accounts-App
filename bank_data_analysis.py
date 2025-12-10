# bank_data_analysis.py
# Option B — Rewritten & optimized single-file Bank Statement Analyzer (Deterministic + optional LLM)
# Run: streamlit run bank_data_analysis.py

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

# Optional OpenAI usage (LLM enhancement). If you don't want LLM, leave secrets empty.
try:
    import openai
except Exception:
    openai = None

# Logging
logger = logging.getLogger("bank_analyzer")
logging.basicConfig(level=logging.INFO)

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
# Date recognition (many formats)
DATE_RE = re.compile(r'^(\d{1,2}/\d{1,2})')


AMOUNT_RE = re.compile(r'([+\-]?\(?\s*\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})\)?)')

SUMMARY_KEYWORDS = [
    "summary", "daily ending", "fees section", "beginning balance",
    "ending balance", "total deposits", "total withdrawals",
    "complete checking", "page", "instance", "amount", "balance",
    "checking summary", "deposits and additions", "checks paid", "atm & debit"
]

def _normalize_date_token(token: str) -> str:
    if not token:
        return ""
    t = token.strip().replace(",", "")
    formats = [
        "%d/%m/%Y", "%d/%m/%y",
        "%m/%d/%Y", "%m/%d/%y",
        "%Y-%m-%d",
        "%d-%m-%Y", "%d-%m-%y",
        "%d %b %Y", "%d %b %y",
        "%d %B %Y", "%d %B %y",
        "%b %d %Y", "%b %d, %Y", "%B %d %Y",
        "%m/%d", "%d/%m"
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(t, fmt)
            # If format had no year (mm/dd), assume current year
            if "%Y" not in fmt and "%y" not in fmt:
                dt = dt.replace(year=datetime.now().year)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue
    # heuristics: "241203" style? not handling here — return raw
    return t

def _clean_amount_token(token: str) -> Optional[float]:
    if not token:
        return None
    s = str(token).strip()
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
    if s.startswith("-"):
        negative = True
    # Remove currency letters & symbols, but not dot or minus
    s = re.sub(r'[A-Za-z\$£€₹]', '', s)
    # Remove commas and spaces used as thousand separators
    s = s.replace(',', '').replace(' ', '')
    s = re.sub(r'[^0-9\.\-]', '', s)
    if not re.search(r'\d', s):
        return None
    # if multiple dots, keep last as decimal separator
    parts = s.split('.')
    if len(parts) > 2:
        s = "".join(parts[:-1]) + "." + parts[-1]
    try:
        val = float(s)
    except Exception:
        return None
    return -abs(val) if negative else abs(val)

def _vendor_cleanup(raw: str) -> str:
    if not raw:
        return "UNKNOWN"
    v = raw.strip()
    v = re.sub(r'Orig Co Name[:\s]*', '', v, flags=re.I)
    v = re.sub(r'Ind Name[:\s]*', '', v, flags=re.I)
    # remove trace, id, trace#, sec: etc
    v = re.sub(r'trace#?:?\s*\S+', '', v, flags=re.I)
    v = re.sub(r'orig id[:\s]*\S+', '', v, flags=re.I)
    v = re.sub(r'descr:?', '', v, flags=re.I)
    v = re.sub(r'\b(sec|sec:|ccd|web|ccd|ccd:|entry)\b', '', v, flags=re.I)
    v = re.sub(r'[^A-Za-z0-9\-\&\.\s]', ' ', v)
    v = re.sub(r'\s{2,}', ' ', v).strip()
    if not v:
        return "UNKNOWN"
    # Common normalizations
    v = v.title()
    v = re.sub(r'\bShopifypmt\b', 'Shopify', v, flags=re.I)
    v = re.sub(r'\bShopifypmnt\b', 'Shopify', v, flags=re.I)
    v = re.sub(r'\bTiktok\b', 'TikTok', v, flags=re.I)
    v = re.sub(r'\bAmazoncom\b', 'Amazon', v, flags=re.I)
    v = re.sub(r'\bEbay\b', 'Ebay', v, flags=re.I)
    # short cutoff
    if len(v) > 60:
        v = v[:60] + "..."
    return v

# ----------------------------
# Document parsing
# ----------------------------
class DocumentParser:
    def parse_pdf_text_lines(self, file_bytes: bytes) -> Tuple[List[str], List[int]]:
        unreadable_pages = []
        lines = []
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    txt = page.extract_text() or ""
                    txt = txt.replace('\xa0', ' ')
                    if txt.strip():
                        page_lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
                        lines.extend(page_lines)
                    else:
                        unreadable_pages.append(page.page_number)
            return lines, unreadable_pages
        except Exception as e:
            logger.exception("PDF read failed: %s", e)
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
# Robust fallback parser (date-segmented)
# ----------------------------
class FallbackStatementParser:
    """
    Parser tuned for Chase-style statements that contain explicit section headings:
      - Deposits and Additions
      - Checks Paid
      - ATM & Debit Card Withdrawals
      - Electronic Withdrawals
      - Fees
    It strips summary blocks and totals, segments transactions by date, and uses section context
    to determine deposit vs withdrawal.
    """

    DATE_RE = re.compile(r'^(\d{1,2}/\d{1,2})\b')
    AMOUNT_RE = re.compile(r'([+\-]?\(?\s*\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})\)?)')

    # Section header tokens (targeting the exact sections you listed + small variants)
    SECTION_PATTERNS = {
        "DEPOSITS": re.compile(r'\bdeposits\s+and\s+additions\b', re.I),
        "CHECKS": re.compile(r'\bchecks\s+paid\b', re.I),
        "ATM": re.compile(r'\batm\b.*\bdebit\b|\batm\s*&\s*debit\b|\batm\s+withdrawal\b|\batm\s*&\s*debit\s*card\s*withdrawals?', re.I),
        "ELECTRONIC_WITHDRAWALS": re.compile(r'\belectronic\s+withdrawals?\b', re.I),
        "FEES": re.compile(r'\bfees?\b', re.I),
        "DAILY_BALANCE": re.compile(r'\bdaily\s+ending\s+balance\b|\bdaily ending balance\b|\bending\s+balance\b.*\bdaily\b', re.I),
    }

    SUMMARY_BLACKLIST = re.compile(
        r'\b(total deposits|total withdrawals|beginning balance|ending balance|closing balance|statement|page of|deposits and additions summary|total)\b',
        re.I
    )

    def _extract_vendor(self, block_text: str, date_raw: str, amount_token: str, section: str = "") -> str:
        # Handle specific sections first
        if section == "CHECKS":
            # Extract check number from format: "157 ^ 12/31 $1,800.00"
            check_match = re.match(r'^(\d+)\s*\^?\s*\d{1,2}/\d{1,2}', block_text)
            if check_match:
                return f"Check #{check_match.group(1)}"
            return "Check"
        
        if section == "ATM":
            # Extract ATM location or just return "ATM Withdrawal"
            if "ATM Withdrawal" in block_text:
                # Try to extract location
                location_match = re.search(r'ATM Withdrawal.*?(\d{4}\s+[A-Z].*?)(?:Card|\d{2}/\d{2}|$)', block_text)
                if location_match:
                    return f"ATM Withdrawal"
            return "ATM Withdrawal"
        
        if section == "FEES":
            # Extract fee description
            fee_match = re.search(r'\d{1,2}/\d{1,2}\s+(.+?)(?:\$|\d+\.\d{2}|$)', block_text)
            if fee_match:
                fee_desc = fee_match.group(1).strip()
                return fee_desc if fee_desc else "Bank Fee"
            return "Bank Fee"
        
        # Regular electronic withdrawal patterns
        m = re.search(r'Orig Co Name[:\s]*([A-Za-z0-9\-\&\.\s\\\/\*]+?)(?:\s+Orig ID|\s+Descr|Trace#|Eed:|Descr:|Ind Name|Trn:|$)', block_text, re.I)
        if m:
            v = m.group(1).strip()
        else:
            m2 = re.search(r'Ind Name[:\s]*([A-Za-z0-9\-\&\.\s\\\/\*]+?)(?:Trace#|Trn:|$)', block_text, re.I)
            if m2:
                v = m2.group(1).strip()
            else:
                # fallback: take words after date in first line
                first_line = block_text.splitlines()[0] if '\n' in block_text else block_text
                after_date = re.sub(self.DATE_RE, '', first_line, count=1).strip()
                after_date = re.sub(r'[\d\|\-:\/\.\(\)\[\],]', ' ', after_date)
                after_date = re.sub(r'\s{2,}', ' ', after_date).strip()
                v = " ".join(after_date.split()[:6]) if after_date else "UNKNOWN"

        # cleanup
        v = re.sub(r'\b(sec|sec:|ccd|web|descr|trace#|trace)\b', ' ', v, flags=re.I)
        v = re.sub(r'[^A-Za-z0-9\-\&\.\s]', ' ', v)
        v = re.sub(r'\s{2,}', ' ', v).strip()
        if not v:
            return "UNKNOWN"
        # shorten overly long vendors
        if len(v) > 60:
            v = v[:60].strip() + "..."
        return v.title()

    def _is_summary_line(self, ln: str) -> bool:
        if not ln or not ln.strip():
            return True

        low = ln.lower().strip()

        # Skip pure section headers and summaries
        SUMMARY_PHRASES = [
            "summary", 
            "daily ending", 
            "beginning balance",
            "ending balance",
            "closing balance",
            "statement period",
            "page ",
            "balance november",
            "balance december",
            "throughdecember",
            "through december",
        ]

        for phrase in SUMMARY_PHRASES:
            if phrase in low:
                return True

        # Skip "Total ..." lines
        if re.match(r'^total\b', low):
            return True
            
        # Skip lines that are ONLY balance information (no vendor)
        # Pattern: date followed by amount(s) but contains "balance" or "end daily"
        if re.search(r'\bbalance\b|\bend\s+daily\b|\bdaily\s+ending\b', low):
            return True

        return False


    def parse_statement(self, lines: List[str]) -> Tuple[List[Transaction], Dict[str, Any]]:
        txs: List[Transaction] = []

        # 1) Pre-clean: remove ALL lines between *start*daily ending balance and *end*daily ending balance
        cleaned = []
        skip_until_end_marker = None
        
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            
            # Check if we're entering a section to skip
            if re.search(r'^\*start\*.*daily.*ending.*balance', s, re.I):
                skip_until_end_marker = 'daily ending balance'
                continue
            
            # Check if we're exiting the skip section
            if skip_until_end_marker:
                if re.search(rf'^\*end\*.*{skip_until_end_marker}', s, re.I):
                    skip_until_end_marker = None
                continue
            
            # Skip section markers (they're just metadata)
            if re.match(r'^\*start\*|^\*end\*', s):
                continue
            
            if self._is_summary_line(s):
                continue
            cleaned.append(s)

        if not cleaned:
            return [], {"parsed_from": "fallback", "transactions_extracted": 0}

        # 2) Walk lines, detect section context and build date-started blocks
        blocks: List[Tuple[List[str], str]] = []
        current_section = "UNKNOWN"
        current_block: Optional[List[str]] = None

        for ln in cleaned:
            # update section if matches
            section_matched = False
            for sec_name, pattern in self.SECTION_PATTERNS.items():
                if pattern.search(ln):
                    # finish any open block BEFORE changing section (with OLD section tag)
                    if current_block:
                        blocks.append((current_block, current_section))
                        current_block = None
                    # now update to new section
                    current_section = sec_name
                    section_matched = True
                    break
            
            if section_matched:
                continue
            
            # no section match — treat as potential transaction content
            # Check for date at start OR check number format (for CHECKS section)
            has_date = self.DATE_RE.search(ln)
            # Pattern for checks: number, optional ^, then date
            is_check_line = current_section == "CHECKS" and re.search(r'^\d+\s*\^?\s*\d{1,2}/\d{1,2}', ln)
            # ATM lines may start with date OR contain "ATM Withdrawal"
            is_atm_line = current_section == "ATM" and (has_date or "atm withdrawal" in ln.lower())
            # Fees may not start with date; allow date or fee keyword
            is_fee_line = current_section == "FEES" and (has_date or "fee" in ln.lower())
            
            if has_date or is_check_line or is_atm_line or is_fee_line:
                # start a new block: finish previous
                if current_block:
                    blocks.append((current_block, current_section))
                current_block = [ln]
            else:
                # continuation line
                if current_block is not None:
                    current_block.append(ln)
                else:
                    # stray non-date lines outside a block — ignore
                    continue

        # finalize last block
        if current_block:
            blocks.append((current_block, current_section))

        # 3) Parse each block: pick last amount, use section to determine direction
        for block_lines, section in blocks:
            block_text = " ".join(block_lines)
            block_lower = block_text.lower()

            # If section detection missed, infer from content to avoid misclassification
            if section == "UNKNOWN":
                if "atm withdrawal" in block_lower:
                    section = "ATM"
                elif block_lower.startswith("check") or re.match(r'^\d+\s*\^?\s*\d{1,2}/\d{1,2}', block_text):
                    section = "CHECKS"
                elif "fee" in block_lower:
                    section = "FEES"
            
            # Skip entire DAILY_BALANCE section (contains multiple dates/amounts per line)
            if section == "DAILY_BALANCE":
                continue
            
            # CRITICAL FIRST FILTER: Detect daily balance lines by their distinctive pattern
            # Daily balance format: "12/02 $11,805.75 12/11 23,765.22 12/20 26,876.38"
            # Key insight: Daily balance has multiple DATE-AMOUNT pairs with NO descriptive text
            
            simple_date_pattern = re.compile(r'\b(\d{1,2}/\d{1,2})\b')
            all_dates = simple_date_pattern.findall(block_text)
            
            # HARD GUARD: if multiple dates and no vendor keywords, drop as daily balance (all sections except checks/atm/fees)
            if section not in ("CHECKS", "ATM", "FEES") and len(all_dates) >= 2:
                has_vendor_keywords_any = bool(re.search(
                    r'\b(orig|co name|ind name|descr|trace|payment|transfer|wise|irs|online|sec:|ccd|web|atm withdrawal|check|fee)\b',
                    block_lower
                ))
                if not has_vendor_keywords_any:
                    continue

            # Skip checks, ATM, fees - they have special formats
            if section not in ("CHECKS", "ATM", "FEES"):
                if len(all_dates) >= 2:
                    # Has multiple dates - check if it's a daily balance line
                    # Daily balance lines have NO vendor-identifying text
                    has_vendor_keywords = bool(re.search(
                        r'\b(orig|co name|ind name|descr|trace|payment|transfer|wise|irs|online|sec:|ccd|web|atm withdrawal|check|fee)\b',
                        block_lower
                    ))
                    
                    # If multiple dates but no vendor keywords, it's daily balance
                    if not has_vendor_keywords:
                        continue
                
                # Additional safety: Skip blocks with 3+ dates (definitely daily balance)
                if len(all_dates) >= 3:
                    continue
            
            # Skip balance lines and end-of-day summaries
            if re.match(r'^(end|ending|end daily|daily ending|end daily ending balance|through)', block_text.strip(), re.I):
                continue
            if re.search(r'\b(balance|end daily|daily ending|throughdecember|through december)\b', block_lower):
                continue
            # Skip blocks that look like totals or summaries
            if re.search(r'\btotal\b.*\d', block_lower):
                continue
            # Skip if it's ONLY a date and amount (balance reporting pattern)
            if re.match(r'^\d{1,2}/\d{1,2}\s+[\d,\.\s]+$', block_text.strip()):
                continue
            # Skip if contains "november" or "december" without a vendor (balance date ranges)
            if re.search(r'\b(november|december)\b', block_lower) and not re.search(r'\b(orig co name|ind name|descr|trace)\b', block_lower):
                continue
            
            # date from first line
            # Try date at start first, then anywhere in the line (for checks)
            d_match = self.DATE_RE.search(block_lines[0])
            if not d_match:
                # For checks format: "157 ^ 12/31 $1,800.00"
                d_match = re.search(r'\b(\d{1,2}/\d{1,2})\b', block_lines[0])
            
            date_raw = d_match.group(1) if d_match else ""
            try:
                month, day = date_raw.split('/')
                year = datetime.now().year
                date_norm = f"{year}-{int(month):02d}-{int(day):02d}"
            except:
                date_norm = ""


            # amounts: for CHECKS pick the first amount; otherwise pick last
            amounts = self.AMOUNT_RE.findall(block_text)
            if not amounts:
                continue
            if section == "CHECKS":
                amount_raw = amounts[0]
            else:
                amount_raw = amounts[-1]
            amt_val = _clean_amount_token(amount_raw)
            if amt_val is None:
                continue

            # determine direction by section (strict)
            if section == "DEPOSITS":
                direction = "deposit"
                signed_amount = abs(amt_val)
            elif section in ("CHECKS", "ATM", "ELECTRONIC_WITHDRAWALS", "FEES"):
                direction = "withdrawal"
                signed_amount = -abs(amt_val)
            else:
                # unknown section — fallback to heuristics
                low_block = block_text.lower()
                if amount_raw.strip().startswith("(") or "-" in amount_raw:
                    direction = "withdrawal"
                    signed_amount = -abs(amt_val)
                elif any(k in low_block for k in ["deposit", "credit", "received", "payout"]):
                    direction = "deposit"
                    signed_amount = abs(amt_val)
                else:
                    # safer default: withdrawal
                    direction = "withdrawal"
                    signed_amount = -abs(amt_val)

            # Skip this redundant filter - we already filtered above
            
            # vendor extraction
            vendor = self._extract_vendor(block_text, date_raw, amount_raw, section)
            
            # Skip if vendor is empty or too generic (likely a balance line)
            # BUT allow checks, ATM, and fees even if generic
            if section not in ("CHECKS", "ATM", "FEES"):
                if not vendor or vendor == "UNKNOWN" or len(vendor.strip()) < 3:
                    continue
                # Skip if vendor is purely numeric (daily balance fragments like "18.", "511.")
                if re.match(r'^[\d\.,\s]+\.?$', vendor.strip()):
                    continue
            # Skip if vendor contains balance keywords
            if re.search(r'\b(balance|end daily|ending|through)\b', vendor, re.I):
                continue

            # optional: flag very large items for review (adjust threshold as needed)
            needs_review = abs(signed_amount) >= 20000.0

            txs.append(Transaction(
                date=date_norm,
                transaction_type='deposit' if signed_amount > 0 else 'withdrawal',
                vendor=vendor,
                amount=signed_amount,
                description=block_text,
                raw_line=block_text,
                section=section,
                needs_review=needs_review
            ))

        meta = {"parsed_from": "chase_sectioned_fallback", "transactions_extracted": len(txs)}
        return txs, meta


# ----------------------------
# LLM enhancer (unchanged but safe)
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
            logger.info("No OPENAI_API_KEY found in Streamlit secrets — skipping LLM enhancement.")
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
            "  idx (int), date (YYYY-MM-DD or original), vendor (short), amount (number positive), direction ('in'/'out'), description (string)\n"
            "Rules:\n"
            "- Normalize amounts (e.g. '29 083.00' -> 29083.00). Return numeric amount (positive).\n"
            "- Do NOT add or remove rows; keep idx mapping.\n"
            "- If you can canonicalize the date to YYYY-MM-DD do so, otherwise return original date string.\n"
            "- Return ONLY a JSON array (no explanation).\n\n"
            "INPUT:\n" + json.dumps(rows, ensure_ascii=False)
        )

        try:
            resp = openai.ChatCompletion.create(
                model=self.model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens
            )
            content = resp.choices[0].message["content"]
            parsed = json.loads(content)
            enhanced: List[Transaction] = []
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

# ----------------------------
# Categorizer & dedupe
# ----------------------------
class TransactionCategorizer:
    def __init__(self):
        pass

    def process_transactions(self, txs: List[Transaction]) -> List[Transaction]:
        for t in txs:
            if t.amount > 0:
                t.category = "Income"
            else:
                low = (t.description or "").lower()
                if 'uber' in low or 'careem' in low:
                    t.category = "Transport"
                elif 'starbuck' in low or 'pizza' in low or 'restaurant' in low:
                    t.category = "Food"
                elif 'wise' in low:
                    t.category = "Transfer"
                elif 'shopify' in low:
                    t.category = "Sales"
                else:
                    t.category = "Expense"
        return txs

    def detect_duplicates(self, txs: List[Transaction]) -> List[Transaction]:
        seen = {}
        order = []
        for t in txs:
            vendor_norm = re.sub(r'\W+', '', (t.vendor or '').lower())
            key = (t.date, round(t.amount, 2), vendor_norm)
            if key in seen:
                existing = seen[key]
                # prefer existing that is not needs_review
                if existing.needs_review and not t.needs_review:
                    seen[key] = t
            else:
                seen[key] = t
                order.append(key)
        return [seen[k] for k in order]

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
st.title("💼 Bank Statement Analyzer — Rewritten Parser (Chase-first, Robust)")

st.markdown(
    "Upload a bank statement (PDF / CSV / DOCX). The app uses a robust deterministic parser optimized for Chase-style multi-line "
    "statements, with optional LLM enhancement. Totals and subtotals are computed from parsed numeric amounts."
)

with st.sidebar:
    st.header("Settings")
    use_llm = st.checkbox("Enable LLM enhancement (cost)", value=False)
    llm_model = st.selectbox("LLM model", ["gpt-4o-mini"], index=0)
    st.markdown("Put your OpenAI key in `.streamlit/secrets.toml` as: `OPENAI_API_KEY = \"sk-...\"`")
    st.markdown("Sort vendor summaries by:")
    sort_by = st.selectbox("Sort by", ["Subtotal (desc)", "Transaction Count (desc)"])

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

            fallback = FallbackStatementParser()
            transactions, meta = fallback.parse_statement(lines)
            parsed_from = meta.get("parsed_from", "fallback")

            if not transactions:
                st.error("No transactions extracted.")
                st.stop()

            # optional LLM enhancement
            if use_llm and openai is not None:
                try:
                    key = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None
                except Exception:
                    key = None
                if key:
                    openai.api_key = key
                    enhancer = LLMEnhancer(model=llm_model)
                    try:
                        transactions = enhancer.enhance(transactions, raw_text="\n".join([t.raw_line for t in transactions]))
                        parsed_from = parsed_from + "-llm"
                    except Exception as e:
                        logger.exception("LLM enhancement error: %s", e)
                        st.warning("LLM enhancement failed — continuing with deterministic parse.")
                else:
                    st.warning("OPENAI_API_KEY not set in Streamlit secrets — skipping LLM enhancement.")
            else:
                if use_llm and openai is None:
                    st.warning("openai Python package not installed — skipping LLM enhancement.")

            # categorize & dedupe
            cat = TransactionCategorizer()
            transactions = cat.process_transactions(transactions)
            transactions = cat.detect_duplicates(transactions)

            # stats & reports
            rg = ReportGenerator()
            stats = rg.generate_summary_statistics(transactions)
            deposits_df = rg.generate_deposits_summary(transactions)
            withdrawals_df = rg.generate_withdrawals_summary(transactions)
            pl_df = rg.generate_pl_report(transactions)

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

# Dashboard
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

    tab1, tab2, tab3, tab4 = st.tabs(["💰 Deposits","💸 Withdrawals","📈 P&L","📋 All Transactions"])
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
                with st.expander(f"{vendor} — {cnt} tx — {cur} {subtotal:,.2f}"):
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
                with st.expander(f"{vendor} — {cnt} tx — {cur} {subtotal:,.2f}"):
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

    # Downloads
    st.header("📥 Download")
    c1,c2,c3 = st.columns(3)
    with c1:
        dep_csv = st.session_state.deposit_df.to_csv(index=False) if (st.session_state.deposit_df is not None and not st.session_state.deposit_df.empty) else ""
        st.download_button("⬇ Deposits CSV", dep_csv, "deposits.csv", mime="text/csv")
    with c2:
        wd_csv = st.session_state.withdrawal_df.to_csv(index=False) if (st.session_state.withdrawal_df is not None and not st.session_state.withdrawal_df.empty) else ""
        st.download_button("⬇ Withdrawals CSV", wd_csv, "withdrawals.csv", mime="text/csv")
    with c3:
        pnl_csv = st.session_state.pl_df.to_csv(index=False) if (st.session_state.pl_df is not None and not st.session_state.pl_df.empty) else ""
        st.download_button("⬇ P&L CSV", pnl_csv, "pnl.csv", mime="text/csv")

    st.success("✅ Report generated. Verify totals against your bank statement.")

