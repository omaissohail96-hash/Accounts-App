# ----------------------------
# PART 1: Core classes & parsers
# ----------------------------
import io
import re
import tempfile
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Tuple, Dict, Any
import pdfplumber
import pandas as pd

# Optional libs - if present will be used; code works without OCR libs too
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import docx
except Exception:
    docx = None

try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    convert_from_bytes = None
    pytesseract = None
    OCR_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ----------------------------
# Data model
# ----------------------------
@dataclass
class Transaction:
    date: str                      # date string (display)
    transaction_type: str          # 'deposit', 'withdrawal', or 'unknown'
    vendor: str
    category: str = None
    amount: float = 0.0
    description: str = ""
    needs_review: bool = False
    raw_line: str = ""


# ----------------------------
# Document Parser
# ----------------------------
class DocumentParser:
    """
    Extracts text lines from PDF / DOCX / CSV bytes.
    OCR fallback (pytesseract + pdf2image) is available when dependencies are installed.
    """

    def __init__(self, use_ocr_if_needed: bool = True):
        # enable OCR fallback (requires pdf2image, pytesseract, Pillow, and Tesseract binary)
        self.ocr_enabled = bool(use_ocr_if_needed and OCR_AVAILABLE)
        if use_ocr_if_needed and not OCR_AVAILABLE:
            logger.warning(
                "OCR requested but pdf2image/pytesseract/Pillow are not installed. "
                "Install the optional dependencies plus the Tesseract binary to enable scanned PDF support."
            )

    def _clean_lines(self, text: str) -> List[str]:
        if not text:
            return []
        return [ln.strip() for ln in text.splitlines() if ln.strip()]

    def _ocr_page(self, file_bytes: bytes, page_number: int) -> str:
        """Run OCR on a single PDF page."""
        if not self.ocr_enabled or not convert_from_bytes:
            return ""
        try:
            images = convert_from_bytes(
                file_bytes,
                first_page=page_number,
                last_page=page_number,
                dpi=300,
                fmt="png"
            )
            if not images:
                return ""
            text = pytesseract.image_to_string(images[0])
            return text or ""
        except Exception:
            logger.exception("OCR extraction failed for page %s", page_number)
            return ""

    def _ocr_entire_pdf(self, file_bytes: bytes) -> Tuple[List[str], List[int]]:
        """OCR entire PDF when no text could be extracted."""
        if not self.ocr_enabled or not convert_from_bytes:
            return [], []
        try:
            images = convert_from_bytes(file_bytes, dpi=300, fmt="png")
        except Exception:
            logger.exception("Unable to convert PDF to images for OCR fallback")
            return [], []

        lines: List[str] = []
        unreadable: List[int] = []
        for idx, image in enumerate(images, start=1):
            try:
                text = pytesseract.image_to_string(image)
                cleaned = self._clean_lines(text)
                if cleaned:
                    lines.extend(cleaned)
                else:
                    unreadable.append(idx)
            except Exception:
                logger.exception("OCR failed on page %s during full-document fallback", idx)
                unreadable.append(idx)
        return lines, unreadable
    def parse_document(self, file_bytes: bytes, filename: str) -> Tuple[List[str], bool, List[int]]:
        ext = filename.lower().split('.')[-1]
        pages_text = []
        unreadable_pages = []
        is_readable = True
        full_text = ""

        try:
            if ext == "pdf":
                try:
                    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                        for i, page in enumerate(pdf.pages):
                            try:
                                page_text = (page.extract_text() or "").replace("\xa0", " ")
                                if page_text.strip():
                                    pages_text.append(page_text)
                                else:
                                    if self.ocr_enabled:
                                        ocr_text = self._ocr_page(file_bytes, i + 1)
                                        if ocr_text.strip():
                                            pages_text.append(ocr_text)
                                            logger.info("Recovered text on page %s via OCR", i + 1)
                                        else:
                                            unreadable_pages.append(i + 1)
                                            pages_text.append("")
                                    else:
                                        unreadable_pages.append(i + 1)
                                        pages_text.append("")
                            except:
                                unreadable_pages.append(i + 1)
                                pages_text.append("")
                    full_text = "\n".join(pages_text)

                    if self.ocr_enabled and not full_text.strip():
                        logger.info("PDF contained no extractable text; running full-document OCR fallback.")
                        ocr_lines, ocr_unreadable = self._ocr_entire_pdf(file_bytes)
                        if ocr_lines:
                            return ocr_lines, len(ocr_unreadable) == 0, ocr_unreadable
                        else:
                            unreadable_pages = ocr_unreadable or unreadable_pages

                except Exception:
                    logger.exception("PDF parsing via pdfplumber failed")
                    full_text = ""
                    is_readable = False

            elif ext in ('doc', 'docx') and docx:
                try:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.' + ext)
                    tmp.write(file_bytes)
                    tmp.flush()
                    doc = docx.Document(tmp.name)
                    paras = [p.text for p in doc.paragraphs]
                    full_text = "\n".join(paras)
                except Exception:
                    logger.exception("DOCX extraction failed")
                    full_text = ""
                    is_readable = False

            elif ext == 'csv':
    # UNIVERSAL CSV DECODER + LINE GENERATOR
                try:
                    # Decode CSV safely
                    csv_text = file_bytes.decode('utf-8', errors='ignore')
                except Exception:
                    csv_text = str(file_bytes)

                # Split into lines (Streamlit parser expects list of lines)
                raw_lines = csv_text.splitlines()

                # Remove extra spaces and empty lines
                lines = [ln.strip() for ln in raw_lines if ln.strip()]

                # CSV files are always readable since they are text
                is_readable = True
                unreadable_pages = []

                return lines, is_readable, unreadable_pages
            else:
                try:
                    full_text = file_bytes.decode('utf-8', errors='ignore')
                except Exception:
                    full_text = ""
                    is_readable = False

        except Exception:
            logger.exception("Unexpected error in parse_document")
            full_text = ""
            is_readable = False

        if ext == "pdf" and self.ocr_enabled and not full_text.strip():
            ocr_lines, ocr_unreadable = self._ocr_entire_pdf(file_bytes)
            if ocr_lines:
                return ocr_lines, len(ocr_unreadable) == 0, ocr_unreadable
            else:
                unreadable_pages = ocr_unreadable or unreadable_pages

        lines = []
        if full_text and len(full_text.strip()) > 5:
            # basic normalization
            full_text = full_text.replace('\xa0', ' ')
            raw_lines = [ln.strip() for ln in full_text.splitlines()]
            # filter out typical headers/footers heuristically
            header_patterns = [r'generated on:', r'page \d+\s+of', r'iban:', r'account transactions', r'note: this is a system generated']
            for ln in raw_lines:
                if not ln:
                    continue
                low = ln.lower()
                if any(re.search(pat, low) for pat in header_patterns):
                    continue
                lines.append(ln)
        else:
            is_readable = False

        # If insufficient text and OCR desired, user can enable OCR mode (not implemented here)
        # if not is_readable and self.use_ocr_if_needed and OCR_AVAILABLE:
        #     ... run pdf2image + pytesseract per page ...

        if len(lines) < 5:
            is_readable = False

        return lines, is_readable, unreadable_pages


# ----------------------------
# Bank Statement Parser (robust)
# ----------------------------
class BankStatementParser:
    """
    Universal statement parser:
    - Groups multiline entries (date -> optional time -> description lines -> amount line)
    - Supports single-line date+amount lines
    - Has robust _parse_amount that normalizes broken PDF number tokens
    """

    DATE_LINE_PAT = re.compile(r'^\s*(\d{1,2}\s+[A-Za-z]{3}\,?\s+\d{4})\s*$', re.IGNORECASE)
    TIME_LINE_PAT = re.compile(r'^\s*(\d{1,2}:\d{2}\s*(AM|PM|am|pm)?)\s*$', re.IGNORECASE)
    # amount: allow commas and parentheses and spaces. We'll clean in _parse_amount.
    AMOUNT_PAT = re.compile(r'([+\-]?\s*\(?\s*[\d\.,\s]+\s*\)?)')
    SINGLE_LINE_DATE = re.compile(r'(\d{1,2}\s+[A-Za-z]{3}\,?\s+\d{4})')

    def __init__(self):
        pass

    def parse_statement(self, lines: List[str]) -> Tuple[List[Transaction], Dict[str, Any]]:
        transactions: List[Transaction] = []
        metadata = {'rows_scanned': len(lines), 'parsed_from': 'unknown'}
        # Quick CSV-like attempt
        header_text = " ".join(lines[:6]).lower()
        if ('date' in header_text and ('amount' in header_text or 'description' in header_text)) or any(',' in ln for ln in lines[:6]):
            try:
                raw = "\n".join(lines)
                df = pd.read_csv(io.StringIO(raw))
                date_col = self._find_column(
                    df.columns,
                    ['date', 'transaction date', 'posted date', 'posted_at', 'posting date', 'value date']
                )
                amt_col = self._find_column(
                    df.columns,
                    ['amount', 'value', 'debit', 'credit', 'amount in', 'amount out']
                )
                desc_col = self._find_column(
                    df.columns,
                    ['description', 'description_raw', 'details', 'narration', 'particulars', 'memo']
                )
                direction_col = self._find_column(
                    df.columns,
                    ['direction', 'transaction type', 'type', 'credit/debit', 'debit/credit', 'dr/cr']
                )
                credit_col = self._find_column(
                    df.columns,
                    ['credit', 'deposit', 'amount in', 'money in']
                )
                debit_col = self._find_column(
                    df.columns,
                    ['debit', 'withdrawal', 'amount out', 'money out', 'payment']
                )
                vendor_col = self._find_column(
                    df.columns,
                    ['vendor', 'merchant', 'merchant name', 'merchant_name', 'payee', 'counterparty']
                )
                if date_col and (amt_col or credit_col or debit_col):
                    for _, row in df.iterrows():
                        date = str(row[date_col]) if not pd.isna(row[date_col]) else ''
                        desc = str(row[desc_col]) if desc_col and not pd.isna(row[desc_col]) else ''

                        if vendor_col and not desc:
                            vendor_val = str(row[vendor_col]) if not pd.isna(row[vendor_col]) else ''
                            desc = vendor_val or desc
                        else:
                            vendor_val = str(row[vendor_col]) if vendor_col and not pd.isna(row[vendor_col]) else ''

                        if amt_col:
                            raw_amt = row[amt_col]
                            amt = self._parse_amount(str(raw_amt))
                        else:
                            credit_amt = self._parse_amount(row[credit_col]) if credit_col and not pd.isna(row[credit_col]) else 0.0
                            debit_amt = self._parse_amount(row[debit_col]) if debit_col and not pd.isna(row[debit_col]) else 0.0
                            amt = credit_amt - abs(debit_amt)

                        if direction_col and not pd.isna(row[direction_col]):
                            direction_value = str(row[direction_col]).strip().lower()
                            if direction_value in {'out', 'debit', 'withdrawal', 'payment', 'charge', 'fee', 'purchase'}:
                                amt = -abs(amt)
                            elif direction_value in {'in', 'credit', 'deposit', 'add', 'received'}:
                                amt = abs(amt)
                        tx_type = 'deposit' if amt > 0 else 'withdrawal'
                        vendor = vendor_val or self._infer_vendor_from_description(desc)
                        transactions.append(Transaction(date=date, transaction_type=tx_type, vendor=vendor,
                                                        amount=abs(amt), description=desc, needs_review=False, raw_line=str(row.to_dict())))
                    metadata['parsed_from'] = 'csv'
                    metadata['transactions_extracted'] = len(transactions)
                    return transactions, metadata
            except Exception:
                logger.info("CSV heuristic parse failed; continuing with multiline heuristics")

        # Heuristic multiline grouping
        i = 0
        n = len(lines)
        grouped = []
        while i < n:
            ln = lines[i].strip()
            # Date line start
            date_match = self.DATE_LINE_PAT.match(ln)
            if date_match:
                date_text = date_match.group(1).strip()
                j = i + 1
                # optional time line
                if j < n and self.TIME_LINE_PAT.match(lines[j].strip()):
                    j += 1
                # collect description until amount-like line or next date
                desc_parts = []
                amount_line = None
                while j < n:
                    candidate = lines[j].strip()
                    if self.DATE_LINE_PAT.match(candidate):
                        break
                    if re.search(r'[+\-]\s*\d', candidate) or self.AMOUNT_PAT.search(candidate):
                        amount_line = candidate
                        j += 1
                        break
                    desc_parts.append(candidate)
                    j += 1
                # fallback: maybe amount on next line
                if not amount_line and j < n and re.search(r'[+\-]\s*\d', lines[j].strip()):
                    amount_line = lines[j].strip()
                    j += 1

                grouped.append({'date': date_text, 'desc': " ".join(desc_parts).strip(), 'amount': amount_line or '', 'raw': lines[i:j]})
                i = j
                continue

            # single-line date + amount
            if self.SINGLE_LINE_DATE.search(ln) and self.AMOUNT_PAT.search(ln):
                d = self.SINGLE_LINE_DATE.search(ln).group(1)
                grouped.append({'date': d, 'desc': ln, 'amount': ln, 'raw': [ln]})
                i += 1
                continue

            # amount-only line attaching to previous
            if re.match(r'^[+\-]?\s*\(?\s*[\d\.,\s]+\s*\)?$', ln) and grouped:
                if not grouped[-1].get('amount'):
                    grouped[-1]['amount'] = ln
                    grouped[-1]['raw'].append(ln)
                else:
                    grouped.append({'date': '', 'desc': '', 'amount': ln, 'raw': [ln]})
                i += 1
                continue

            i += 1

        # Build transactions from grouped entries
        for rec in grouped:
            date_raw = rec.get('date', '').strip()
            desc_raw = rec.get('desc', '').strip()
            amt_raw = rec.get('amount', '') or ''
            raw_join = " | ".join(rec.get('raw', []))

            if not amt_raw:
                transactions.append(Transaction(date=date_raw or 'N/A', transaction_type='unknown', vendor='UNKNOWN',
                                                amount=0.0, description=desc_raw or raw_join, needs_review=True, raw_line=raw_join))
                continue

            amt = self._parse_amount(amt_raw)
            tx_type = 'deposit' if amt > 0 else 'withdrawal'
            combined = f"{desc_raw} {amt_raw}".upper()
            if '/CR' in combined or ' CR' in combined:
                tx_type = 'deposit'
            if '/DR' in combined or ' DR' in combined:
                tx_type = 'withdrawal'

            vendor = self._infer_vendor_from_description(desc_raw)
            if vendor in ('', 'UNKNOWN'):
                vendor = self._infer_vendor_from_description(amt_raw)

            needs_review = (vendor in ('UNKNOWN', '') or amt == 0)
            transactions.append(Transaction(date=date_raw or 'N/A', transaction_type=tx_type, vendor=vendor or 'UNKNOWN',
                                            amount=abs(amt), description=(desc_raw or amt_raw).strip(), needs_review=needs_review, raw_line=raw_join))

        # Fallback single-line scan if none found
        if not transactions:
            for ln in lines:
                dmatch = self.SINGLE_LINE_DATE.search(ln)
                amatch = self.AMOUNT_PAT.search(ln)
                if dmatch and amatch:
                    date_raw = dmatch.group(1)
                    amt = self._parse_amount(amatch.group(1))
                    tx_type = 'deposit' if amt > 0 else 'withdrawal'
                    vendor = self._infer_vendor_from_description(ln)
                    transactions.append(Transaction(date=date_raw, transaction_type=tx_type, vendor=vendor,
                                                    amount=abs(amt), description=ln, needs_review=False, raw_line=ln))
            metadata['parsed_from'] = 'single-line-fallback'

        metadata.setdefault('parsed_from', 'heuristic-multiline')
        metadata['transactions_extracted'] = len(transactions)
        return transactions, metadata

    def _find_column(self, columns, candidates):
        cols = [c.lower() for c in columns]
        for cand in candidates:
            for i, c in enumerate(cols):
                if cand in c:
                    return columns[i]
        return None

    def _parse_amount(self, s: str) -> float:
            """
            SUPER-ROBUST AMOUNT PARSER
            Fixes:
            - 2 9 0 8 3 . 0 0
            - + 29 . 083 .00
            - ( 1 , 4 0 3 . 7 5 )
            - 29 . 083 . 00
            - 2,90 83 .00
            - PKR 2,903.00 CR/DR
            - Random spaces inside digits
            """

            if not s:
                return 0.0

            s = str(s)

            # Strip non-breaking spaces
            s = s.replace("\xa0", " ")

            # Detect parentheses negative
            negative = "(" in s and ")" in s

            # Remove currency symbols
            s = re.sub(r"[A-Za-z₹$€£¥]", " ", s)

            # Remove CR/DR words
            s = re.sub(r"\bCR\b|\bDR\b", "", s, flags=re.IGNORECASE)

            # Remove all commas
            s = s.replace(",", " ")

            # FIX: join digits separated by spaces OR dots between digits
            # e.g. 2 9 . 0 8 3 . 0 0  → 29083.00
            s = re.sub(r'(?<=\d)[\s\.]+(?=\d)', '', s)

            # Remove remaining spaces
            s = s.replace(" ", "")

            # Find number with optional decimal
            match = re.search(r"([+-]?\d+(?:\.\d+)?)", s)
            if not match:
                return 0.0

            num = match.group(1)

            try:
                val = float(num)
            except:
                val = 0.0

            # Apply negative parentheses
            if negative:
                val = -abs(val)

            return val    


    def _infer_vendor_from_description(self, desc: str) -> str:
        if not desc:
            return 'UNKNOWN'
        d = desc
        # remove likely GUIDs and long hex strings
        d = re.sub(r'[0-9a-fA-F\-]{8,}', ' ', d)
        # remove common transaction tokens
        d = re.sub(r'\b(TRANSF|CR|DR|OGT|ICT|IB|W2W|PUR|POS|PAYMENT|DEBIT|CREDIT|ATM)\b', ' ', d, flags=re.IGNORECASE)
        d = re.sub(r'[\|\-\:\,\/]+', ' ', d)
        d = re.sub(r'\s{2,}', ' ', d).strip()
        vendor = d[:80]
        if re.fullmatch(r'[\d\W]+', vendor):
            return 'UNKNOWN'
        return vendor if vendor else 'UNKNOWN'


# ----------------------------
# Transaction Categorizer
# ----------------------------
class TransactionCategorizer:
    CATEGORY_RULES = {
        'rent': ['rent', 'landlord'],
        'utilities': ['electric', 'water', 'gas', 'utility', 'utilities'],
        'internet': ['internet', 'fiber', 'wifi'],
        'payroll': ['salary', 'payroll', 'salary deposit'],
        'bank fees': ['fee', 'service charge', 'bank fee', 'overdraft'],
        'atm': ['atm', 'cash withdrawal'],
        'shopping': ['amazon', 'walmart', 'store', 'shop', 'shopping'],
        'advertising': ['facebook', 'google ads', 'adwords', 'ads'],
        'transport': ['uber', 'lyft', 'taxi', 'transport'],
        'food': ['restaurant', 'starbucks', 'cafe', 'pizza'],
        'shipping': ['ups', 'fedex', 'dhl', 'post'],
        'insurance': ['insurance', 'insurer'],
        'taxes': ['tax', 'irs', 'federal tax'],
        'interest': ['interest'],
        'unknown': []
    }

    def __init__(self):
        pass

    def process_transactions(self, transactions: List[Transaction]) -> List[Transaction]:
        for t in transactions:
            assigned = self._categorize_by_keywords(t)
            if assigned:
                t.category = assigned
            else:
                t.category = 'Uncategorized'
                if t.vendor in ('UNKNOWN', '') or (not t.description or len(t.description) < 5):
                    t.needs_review = True
        return transactions

    def detect_duplicates(self, transactions: List[Transaction]) -> List[Transaction]:
        seen = {}
        order = []
        for t in transactions:
            key = (self._normalize_date(t.date), round(t.amount, 2), re.sub(r'\W+', '', (t.vendor or '').lower()))
            if key in seen:
                existing = seen[key]
                if existing.needs_review and not t.needs_review:
                    seen[key] = t
            else:
                seen[key] = t
                order.append(key)
        return [seen[k] for k in order]

    def _normalize_date(self, s: str) -> str:
        if not s:
            return ''
        s = s.strip()
        # try a few common formats
        for fmt in ['%d %b, %Y', '%d %b %Y', '%Y-%m-%d', '%d-%b-%Y']:
            try:
                d = datetime.strptime(s.replace(',', ''), fmt)
                return d.strftime('%Y-%m-%d')
            except Exception:
                continue
        return s

    def _categorize_by_keywords(self, t: Transaction) -> str:
        text = f"{t.vendor} {t.description}".lower()
        for cat, kws in self.CATEGORY_RULES.items():
            for kw in kws:
                if kw in text:
                    return cat.title()
        if 'atm' in text or 'cash' in text:
            return 'ATM'
        if t.amount >= 1000 and t.transaction_type == 'deposit':
            return 'Large Deposit'
        return None


# ----------------------------
# Report Generator
# ----------------------------
class ReportGenerator:
    def __init__(self):
        pass

    def generate_summary_statistics(self, transactions: List[Transaction]) -> Dict[str, Any]:
        total_deposits = sum(t.amount for t in transactions if t.transaction_type == 'deposit')
        total_withdrawals = sum(t.amount for t in transactions if t.transaction_type == 'withdrawal')
        total_deposit_count = sum(1 for t in transactions if t.transaction_type == 'deposit')
        total_withdrawal_count = sum(1 for t in transactions if t.transaction_type == 'withdrawal')
        needs_review = sum(1 for t in transactions if t.needs_review)
        net_income = total_deposits - total_withdrawals
        stats = {
            'Total Deposit Amount': total_deposits,
            'Total Withdrawal Amount': total_withdrawals,
            'Total Deposits': total_deposit_count,
            'Total Withdrawals': total_withdrawal_count,
            'Total Transactions': len(transactions),
            'Net Income': net_income,
            'Transactions Needing Review': needs_review
        }
        return stats

    def generate_deposits_summary(self, transactions: List[Transaction]) -> pd.DataFrame:
        deps = [t for t in transactions if t.transaction_type == 'deposit']
        if not deps:
            return pd.DataFrame()
        df = pd.DataFrame([asdict(t) for t in deps])
        grp = df.groupby('vendor').agg({'amount': 'sum', 'raw_line': 'count'}).reset_index()
        grp.columns = ['Source/Vendor', 'Subtotal ($)', 'Transaction Count']
        grp['Subtotal ($)'] = grp['Subtotal ($)'].astype(float)
        total = grp['Subtotal ($)'].sum()
        total_row = pd.DataFrame([{'Source/Vendor': 'TOTAL DEPOSITS', 'Subtotal ($)': total, 'Transaction Count': grp['Transaction Count'].sum()}])
        out = pd.concat([grp, total_row], ignore_index=True)
        return out[['Source/Vendor', 'Transaction Count', 'Subtotal ($)']]

    def generate_withdrawals_summary(self, transactions: List[Transaction]) -> pd.DataFrame:
        wds = [t for t in transactions if t.transaction_type == 'withdrawal']
        if not wds:
            return pd.DataFrame()
        df = pd.DataFrame([asdict(t) for t in wds])
        df['category'] = df['category'].fillna('Uncategorized')
        grp = df.groupby(['category', 'vendor']).agg({'amount': 'sum', 'raw_line': 'count'}).reset_index()
        grp.columns = ['Category', 'Vendor', 'Subtotal ($)', 'Transaction Count']
        total = grp['Subtotal ($)'].sum()
        total_row = pd.DataFrame([{'Category': 'TOTAL WITHDRAWALS', 'Vendor': '', 'Subtotal ($)': total, 'Transaction Count': grp['Transaction Count'].sum()}])
        out = pd.concat([grp, total_row], ignore_index=True)
        return out[['Category', 'Vendor', 'Transaction Count', 'Subtotal ($)']]

    def generate_pl_report(self, transactions: List[Transaction]) -> pd.DataFrame:
        stats = self.generate_summary_statistics(transactions)
        wds = [t for t in transactions if t.transaction_type == 'withdrawal']
        if not wds:
            pl = pd.DataFrame([
                {'Category': 'Total Income', 'Type': 'Income', 'Amount ($)': stats['Total Deposit Amount']},
                {'Category': 'Total Expenses', 'Type': 'Expense', 'Amount ($)': -stats['Total Withdrawal Amount']},
                {'Category': 'NET INCOME', 'Type': 'Net', 'Amount ($)': stats['Net Income']}
            ])
            return pl
        df = pd.DataFrame([asdict(t) for t in wds])
        df['category'] = df['category'].fillna('Uncategorized')
        expense_by_cat = df.groupby('category').agg({'amount': 'sum'}).reset_index()
        expense_by_cat['amount'] = -expense_by_cat['amount']  # negative for display
        expense_by_cat['Type'] = 'Expense'
        expense_by_cat = expense_by_cat.rename(columns={'category': 'Category', 'amount': 'Amount ($)'})
        total_expenses = expense_by_cat['Amount ($)'].sum()
        income_row = pd.DataFrame([{'Category': 'Total Income', 'Type': 'Income', 'Amount ($)': stats['Total Deposit Amount']}])
        total_row = pd.DataFrame([{'Category': 'TOTAL EXPENSES', 'Type': 'Expense', 'Amount ($)': total_expenses}])
        net_row = pd.DataFrame([{'Category': 'NET INCOME', 'Type': 'Net', 'Amount ($)': stats['Net Income']}])
        pl = pd.concat([income_row, expense_by_cat, total_row, net_row], ignore_index=True)
        return pl

    def generate_needs_review_section(self, transactions: List[Transaction]) -> pd.DataFrame:
        review = [t for t in transactions if t.needs_review]
        if not review:
            return pd.DataFrame()
        df = pd.DataFrame([{
            'Date': t.date,
            'Type': t.transaction_type.title(),
            'Vendor': t.vendor,
            'Amount': t.amount,
            'Description': t.description,
            'Reason': 'Missing vendor/unclear description' if (not t.vendor or t.vendor == 'UNKNOWN') else 'Check formatting'
        } for t in review])
        return df

    def export_to_excel(self, transactions: List[Transaction], out_filename="bank_statement_report.xlsx") -> str:
        deposits_df = self.generate_deposits_summary(transactions)
        withdrawals_df = self.generate_withdrawals_summary(transactions)
        pl_df = self.generate_pl_report(transactions)
        needs_df = self.generate_needs_review_section(transactions)
        all_df = pd.DataFrame([asdict(t) for t in transactions])

        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        with pd.ExcelWriter(tmpfile.name, engine='openpyxl') as writer:
            if not deposits_df.empty:
                deposits_df.to_excel(writer, sheet_name='Deposits', index=False)
            if not withdrawals_df.empty:
                withdrawals_df.to_excel(writer, sheet_name='Withdrawals', index=False)
            if not pl_df.empty:
                pl_df.to_excel(writer, sheet_name='P&L', index=False)
            if not needs_df.empty:
                needs_df.to_excel(writer, sheet_name='Needs Review', index=False)
            all_df.to_excel(writer, sheet_name='All Transactions', index=False)
            stats = self.generate_summary_statistics(transactions)
            summary_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        return tmpfile.name
# ----------------------------
# PART 2 – Streamlit App UI
# ----------------------------
import streamlit as st

st.set_page_config(
    page_title="Bank Statement Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Title + Description
st.markdown("""
    <style>
    .main-header { font-size: 2.4rem; font-weight:700; color:#0b5ed7; }
    .sub-header { color:#444; margin-bottom:0.8rem;font-size:1.1rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">💼 Bank Statement Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload ANY bank statement (PDF / CSV / DOCX). System auto-detects, parses, categorizes and generates full report.</p>', unsafe_allow_html=True)


# ----------------------------
# Session State
# ----------------------------
if "transactions" not in st.session_state:
    st.session_state.transactions = []

if "parsing_metadata" not in st.session_state:
    st.session_state.parsing_metadata = {}

if "statistics" not in st.session_state:
    st.session_state.statistics = {}

if "currency" not in st.session_state:
    st.session_state.currency = "PKR"


# ----------------------------
# Centered Upload UI
# ----------------------------
st.markdown("<h4 style='text-align:center;'>📤 Upload Statement</h4>", unsafe_allow_html=True)
u_cols = st.columns([1, 2, 1])

with u_cols[1]:
    uploaded_file = st.file_uploader(
        "Select Bank Statement File",
        type=["pdf", "csv", "doc", "docx"],
        accept_multiple_files=False,
        help="Supported formats: PDF, CSV, Word documents"
    )

# 🔥 Ask Currency BEFORE processing
if uploaded_file:
    st.info(f"📄 File: {uploaded_file.name}")
    st.info(f"📊 Size: {uploaded_file.size / 1024:.1f} KB")

    st.subheader("🌍 Select Currency for this Statement")
    st.session_state.currency = st.selectbox(
        "Currency",
        ["PKR", "USD", "EUR", "GBP", "AED", "CAD", "AUD"],
        index=0
    )

    st.write("")  # spacing
    btn_col = st.columns([1, 1, 1])[1]

    with btn_col:
        if st.button("🔄 Process Statement", type="primary", use_container_width=True):

            with st.spinner("⏳ Extracting & Processing..."):
                try:
                    file_bytes = uploaded_file.read()

                    # Step 1 — Document Parsing
                    doc_parser = DocumentParser(use_ocr_if_needed=True)
                    lines, is_readable, unreadable_pages = doc_parser.parse_document(
                        file_bytes, uploaded_file.name
                    )

                    if not is_readable or len(lines) < 5:
                        st.error("❌ Could not read enough text from this file.")
                        if unreadable_pages:
                            st.warning(f"Unreadable Pages: {unreadable_pages}")
                        st.session_state.transactions = []
                        st.stop()

                    # Step 2 — Bank Statement Parser
                    statement_parser = BankStatementParser()
                    transactions, metadata = statement_parser.parse_statement(lines)

                    if not transactions:
                        st.warning("⚠ No transactions were detected in this file.")
                        st.session_state.transactions = []
                        st.stop()

                    # Step 3 — Categorizer
                    categorizer = TransactionCategorizer()
                    transactions = categorizer.process_transactions(transactions)
                    transactions = categorizer.detect_duplicates(transactions)

                    # Step 4 — Stats
                    report_gen = ReportGenerator()
                    stats = report_gen.generate_summary_statistics(transactions)

                    # Save into session
                    st.session_state.transactions = transactions
                    st.session_state.parsing_metadata = metadata
                    st.session_state.statistics = stats

                    st.success(f"✅ Successfully processed {len(transactions)} transactions!")

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.session_state.transactions = []
                    st.stop()


# ----------------------------
# STOP if no transactions yet
# ----------------------------
if not st.session_state.transactions:
    st.info("👆 Upload a statement to begin analysis.")
    st.stop()


transactions = st.session_state.transactions
metadata = st.session_state.parsing_metadata
statistics = st.session_state.statistics
cur = st.session_state.currency


# ----------------------------
# Warnings (Unreadable Pages)
# ----------------------------
if metadata.get("unreadable_pages"):
    st.warning(
        f"⚠ Some pages were unreadable: {metadata['unreadable_pages']}. "
        "OCR or clearer scan recommended."
    )

# Needs Review Warning
if statistics["Transactions Needing Review"] > 0:
    st.warning(
        f"⚠ {statistics['Transactions Needing Review']} transaction(s) need manual review."
    )


# ----------------------------
# Summary Metrics
# ----------------------------
st.header("📊 Summary Statistics")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Total Deposits",
        f"{cur} {statistics['Total Deposit Amount']:,.2f}",
        f"{statistics['Total Deposits']} transactions"
    )

with c2:
    st.metric(
        "Total Withdrawals",
        f"{cur} {statistics['Total Withdrawal Amount']:,.2f}",
        f"{statistics['Total Withdrawals']} transactions"
    )

with c3:
    st.metric("Net Income", f"{cur} {statistics['Net Income']:,.2f}")

with c4:
    st.metric(
        "Total Transactions",
        statistics["Total Transactions"],
        f"{statistics['Transactions Needing Review']} need review"
    )


# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💰 Deposits Summary",
    "💸 Withdrawals Summary",
    "📈 Profit & Loss",
    "⚠ Needs Review",
    "📋 All Transactions"
])

report_gen = ReportGenerator()


# ----------------------------
# TAB 1 – Deposits
# ----------------------------
with tab1:
    st.subheader("Deposits Summary")

    df = report_gen.generate_deposits_summary(transactions)
    if df.empty:
        st.info("No deposits found.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


# ----------------------------
# TAB 2 – Withdrawals
# ----------------------------
with tab2:
    st.subheader("Withdrawals Summary")

    df = report_gen.generate_withdrawals_summary(transactions)
    if df.empty:
        st.info("No withdrawals found.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


# ----------------------------
# TAB 3 – P&L
# ----------------------------
with tab3:
    st.subheader("Profit & Loss Report")
    pl_df = report_gen.generate_pl_report(transactions)

    if pl_df.empty:
        st.info("No data available.")
    else:
        st.dataframe(pl_df, use_container_width=True, hide_index=True)

        # Expenses Chart
        expenses = pl_df[(pl_df["Type"] == "Expense") & (pl_df["Category"] != "TOTAL EXPENSES")]
        if not expenses.empty:
            st.subheader("Expenses by Category")
            chart_df = expenses[["Category", "Amount ($)"]].copy()
            chart_df["Amount ($)"] = chart_df["Amount ($)"].abs()
            st.bar_chart(chart_df.set_index("Category"))


# ----------------------------
# TAB 4 – Needs Review
# ----------------------------
with tab4:
    st.subheader("Transactions Needing Review")

    review_df = report_gen.generate_needs_review_section(transactions)
    if review_df.empty:
        st.success("No issues found!")
    else:
        st.warning(f"{len(review_df)} transaction(s) need attention.")
        st.dataframe(review_df, use_container_width=True, hide_index=True)


# ----------------------------
# TAB 5 – All Transactions
# ----------------------------
with tab5:
    st.subheader("All Transactions")

    df = pd.DataFrame([{
        "Date": t.date,
        "Type": t.transaction_type.title(),
        "Vendor": t.vendor,
        "Category": t.category,
        "Amount": f"{cur} {t.amount:,.2f}",
        "Description": t.description,
        "Needs Review": "⚠ Yes" if t.needs_review else "✅ No"
    } for t in transactions])

    st.dataframe(df, use_container_width=True, hide_index=True)


# ----------------------------
# Download Section
# ----------------------------
st.header("📥 Download Reports")

d1, d2, d3 = st.columns(3)

with d1:
    dep_csv = report_gen.generate_deposits_summary(transactions).to_csv(index=False)
    st.download_button(
        "⬇ Deposits (CSV)",
        dep_csv.encode("utf-8"),
        "deposits.csv",
        "text/csv"
    )

with d2:
    wd_csv = report_gen.generate_withdrawals_summary(transactions).to_csv(index=False)
    st.download_button(
        "⬇ Withdrawals (CSV)",
        wd_csv.encode("utf-8"),
        "withdrawals.csv",
        "text/csv"
    )

with d3:
    pl_csv = report_gen.generate_pl_report(transactions).to_csv(index=False)
    st.download_button(
        "⬇ Profit & Loss (CSV)",
        pl_csv.encode("utf-8"),
        "pl_report.csv",
        "text/csv"
    )


st.markdown("---")

if st.button("📊 Download FULL Excel Report", type="primary"):
    try:
        excel_file = report_gen.export_to_excel(transactions)
        with open(excel_file, "rb") as f:
            st.download_button(
                "⬇ Download Excel File",
                f.read(),
                "bank_statement_report.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    except Exception as e:
        st.error(f"Error generating Excel: {e}")

st.success("✅ Report ready! Verify results with your bank statement.")
