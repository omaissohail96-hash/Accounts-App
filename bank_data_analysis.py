"""
Bank Statement Analyzer - Robust single-file Streamlit app
Improvements:
 - Robust multiline transaction grouping (date + time + desc + amount)
 - Handles single-line transactions too
 - Better vendor inference and sign detection (CR/DR, + / -)
 - Keeps categorization, duplicate detection, needs-review, export features
"""

import io
import re
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Tuple, Dict, Any

import pandas as pd
import streamlit as st
import logging

# Optional libs
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import docx
except Exception:
    docx = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ----------------------------
# Data model
# ----------------------------
@dataclass
class Transaction:
    date: str  # string display
    transaction_type: str  # 'deposit' or 'withdrawal'
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
    def __init__(self):
        pass

    def parse_document(self, file_bytes: bytes, filename: str) -> Tuple[List[str], bool, List[int]]:
        """
        Extract text lines from uploaded file.
        Returns lines (list), is_readable (bool), unreadable_pages (list)
        """
        ext = filename.lower().split('.')[-1]
        text_pages = []
        unreadable_pages = []
        is_readable = True

        if ext == 'pdf' and PyPDF2:
            try:
                reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                for i, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text() or ""
                        if not page_text.strip():
                            unreadable_pages.append(i + 1)
                        text_pages.append(page_text)
                    except Exception:
                        unreadable_pages.append(i + 1)
                        text_pages.append("")
                text = "\n".join(text_pages)
            except Exception as e:
                logger.exception("PDF parsing failed")
                text = ""
                is_readable = False
        elif ext in ('doc', 'docx') and docx:
            try:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.' + ext)
                tmp.write(file_bytes)
                tmp.flush()
                d = docx.Document(tmp.name)
                paras = [p.text for p in d.paragraphs]
                text = "\n".join(paras)
            except Exception:
                logger.exception("DOCX parse failed")
                text = ""
                is_readable = False
        elif ext == 'csv':
            try:
                text = file_bytes.decode('utf-8', errors='ignore')
            except Exception:
                text = str(file_bytes)
        else:
            try:
                text = file_bytes.decode('utf-8', errors='ignore')
            except Exception:
                text = ""
                is_readable = False

        lines = []
        if text:
            # Normalize common non-breaking spaces and weird unicode
            text = text.replace('\xa0', ' ')
            # split and preserve order; remove empty lines but keep structural short lines
            raw_lines = [ln.strip() for ln in text.splitlines()]
            # Collapse runs of page headers/footers by removing repeated known phrases
            # We'll do a lightweight filter: drop lines that look like repeated headers such as "Generated on:" or address lines repeated many times
            repeated_header_patterns = [
                r'ufone tower', r'generated on:', r'page \d+ of', r'iban:', r'account transactions', r'note: this is a system generated'
            ]
            for ln in raw_lines:
                low = ln.lower()
                if any(re.search(pat, low) for pat in repeated_header_patterns):
                    continue
                if ln.strip() == '':
                    continue
                lines.append(ln)
        else:
            is_readable = False

        # Heuristic: if very few lines, mark unreadable
        if len(lines) < 5:
            is_readable = False

        return lines, is_readable, unreadable_pages
currency = st.selectbox(
    "Select Currency",
    ["PKR", "USD", "EUR", "GBP", "AED", "CAD"],
    index=0
)
st.session_state.currency = currency

# ----------------------------
# Bank Statement Parser (robust)
# ----------------------------
class BankStatementParser:
    # Patterns
    DATE_LINE_PAT = re.compile(r'^\s*(\d{1,2}\s+[A-Za-z]{3}\,?\s+\d{4})\s*$', re.IGNORECASE)  # 1 Oct, 2025
    TIME_LINE_PAT = re.compile(r'^\s*(\d{1,2}:\d{2}\s*(AM|PM|am|pm)?)\s*$', re.IGNORECASE)
    AMOUNT_PAT = re.compile(r'([+\-]?\s*\(?\s*\d{1,3}(?:[,\d{3}])*(?:\.\d{2})\s*\)?)')
    # Single-line date+amount (like "28 Oct, 2025 02:03 PM TRANSF ... - 295.00")
    SINGLE_LINE_PAT_DATE = re.compile(r'(\d{1,2}\s+[A-Za-z]{3}\,?\s+\d{4})')

    def __init__(self):
        pass

    def parse_statement(self, lines: List[str]) -> Tuple[List[Transaction], Dict[str, Any]]:
        """
        Attempt multiple strategies:
         - Group multiline records (date line => optional time => description lines => amount line)
         - Fall back to scanning single lines for date+amount
         - Support CSV-ish input if lines contain commas and headers
        """
        transactions: List[Transaction] = []
        metadata = {'rows_scanned': len(lines), 'parsed_from': 'unknown'}

        # Strategy 1: If looks like CSV table, handle via pandas
        header_candidates = " ".join(lines[:6]).lower()
        if ('date' in header_candidates and ('amount' in header_candidates or 'description' in header_candidates)) or any(',' in ln for ln in lines[:6]):
            try:
                raw = "\n".join(lines)
                df = pd.read_csv(io.StringIO(raw))
                date_col = self._find_column(df.columns, ['date', 'transaction date', 'posted date'])
                amt_col = self._find_column(df.columns, ['amount', 'amount ($)', 'debit', 'credit', 'value'])
                desc_col = self._find_column(df.columns, ['description', 'details', 'narration', 'particulars'])
                if date_col and amt_col:
                    for _, row in df.iterrows():
                        date = str(row[date_col]) if not pd.isna(row[date_col]) else ''
                        desc = str(row[desc_col]) if desc_col and not pd.isna(row[desc_col]) else ''
                        raw_amt = row[amt_col]
                        try:
                            amt = float(raw_amt)
                        except Exception:
                            amt = self._parse_amount(str(raw_amt))
                        tx_type = 'deposit' if amt > 0 else 'withdrawal'
                        vendor = self._infer_vendor_from_description(desc)
                        t = Transaction(date=date, transaction_type=tx_type, vendor=vendor, category=None,
                                        amount=abs(amt), description=desc, needs_review=False, raw_line=str(row.to_dict()))
                        transactions.append(t)
                    metadata['parsed_from'] = 'csv'
                    metadata['transactions_extracted'] = len(transactions)
                    return transactions, metadata
            except Exception:
                logger.info("CSV attempt failed, continuing heuristic parsing")

        # Strategy 2: Group multiline entries
        i = 0
        n = len(lines)
        grouped_records = []
        while i < n:
            ln = lines[i]
            # If this line is a date-only line -> start record
            date_m = self.DATE_LINE_PAT.match(ln)
            if date_m:
                date_text = date_m.group(1).strip()
                rec_lines = [ln]
                j = i + 1
                # optional time
                if j < n and self.TIME_LINE_PAT.match(lines[j]):
                    rec_lines.append(lines[j])
                    j += 1
                # Collect description lines until we hit an amount-like line or next date
                amount_line = None
                desc_lines = []
                while j < n:
                    # If next starts a date, break (new record)
                    if self.DATE_LINE_PAT.match(lines[j]):
                        break
                    # If line contains a clear amount token (+ or - followed by digits) treat as amount
                    if re.search(r'[+\-]\s*\d', lines[j]) or self.AMOUNT_PAT.search(lines[j]):
                        amount_line = lines[j]
                        j += 1
                        break
                    # Another possibility: if line looks like "PUR/..." or "TRANSF ..." it's description
                    desc_lines.append(lines[j])
                    j += 1
                # fallback: maybe amount is on the next next line (some statements put desc then blank then amount)
                if not amount_line and j < n and re.search(r'[+\-]\s*\d', lines[j]):
                    amount_line = lines[j]
                    j += 1

                # Build a record
                record = {
                    'date': date_text,
                    'desc': " ".join(desc_lines).strip(),
                    'amount_line': amount_line or '',
                    'raw_lines': lines[i:j]
                }
                grouped_records.append(record)
                i = j
                continue

            # If line contains a single-line date + amount
            single_date = self.SINGLE_LINE_PAT_DATE.search(ln)
            amount_here = self.AMOUNT_PAT.search(ln)
            if single_date and amount_here:
                grouped_records.append({
                    'date': single_date.group(1),
                    'desc': ln,
                    'amount_line': ln,
                    'raw_lines': [ln]
                })
                i += 1
                continue

            # Otherwise, maybe the file uses a two-line per transaction format (desc then amount)
            # If this line contains an amount alone, try to attach to previous non-amount line
            if re.match(r'^[+\-]?\s*\(?\s*\d', ln) and grouped_records:
                prev = grouped_records[-1]
                if not prev.get('amount_line'):
                    prev['amount_line'] = ln
                    prev['raw_lines'].append(ln)
                else:
                    # new standalone
                    grouped_records.append({'date': '', 'desc': '', 'amount_line': ln, 'raw_lines': [ln]})
                i += 1
                continue

            # else skip or move on
            i += 1

        # From grouped_records, create Transaction objects
        for rec in grouped_records:
            date_raw = rec.get('date', '').strip()
            desc_raw = rec.get('desc', '').strip()
            amt_line = rec.get('amount_line', '')
            raw_join = " | ".join(rec.get('raw_lines', []))
            if not amt_line:
                # if no amount, mark for review but still include
                t = Transaction(date=date_raw or 'N/A', transaction_type='unknown', vendor='UNKNOWN',
                                category=None, amount=0.0, description=desc_raw or raw_join, needs_review=True, raw_line=raw_join)
                transactions.append(t)
                continue

            # parse amount: detect sign by presence of + or - or CR/DR in description
            amt = self._parse_amount(amt_line)
            # determine credit/debit by tokens in description or amount sign
            tx_type = 'deposit' if amt > 0 else 'withdrawal'
            # also consider CR/DR markers in desc or amount_line
            combined_text = f"{desc_raw} {amt_line}".upper()
            if ' CR' in combined_text or '/CR' in combined_text:
                tx_type = 'deposit'
            if ' DR' in combined_text or '/DR' in combined_text:
                tx_type = 'withdrawal'
            # vendor inference
            vendor = self._infer_vendor_from_description(desc_raw)
            # if vendor unknown, try to pull from amount line context
            if vendor in ('', 'UNKNOWN'):
                vendor = self._infer_vendor_from_description(amt_line)

            needs_review = False
            if vendor in ('UNKNOWN', '') or amt == 0:
                needs_review = True

            t = Transaction(
                date=date_raw or 'N/A',
                transaction_type=tx_type,
                vendor=vendor or 'UNKNOWN',
                category=None,
                amount=abs(amt),
                description=(desc_raw or amt_line).strip(),
                needs_review=needs_review,
                raw_line=raw_join
            )
            transactions.append(t)

        # Strategy 3 fallback: If no records found, try single-line scanning
        if not transactions:
            for ln in lines:
                # find date and amount in same line
                dmatch = self.SINGLE_LINE_PAT_DATE.search(ln)
                am = self.AMOUNT_PAT.search(ln)
                if dmatch and am:
                    date_text = dmatch.group(1)
                    amt = self._parse_amount(am.group(1))
                    tx_type = 'deposit' if amt > 0 else 'withdrawal'
                    vendor = self._infer_vendor_from_description(ln)
                    t = Transaction(date=date_text, transaction_type=tx_type, vendor=vendor, category=None,
                                    amount=abs(amt), description=ln, needs_review=False, raw_line=ln)
                    transactions.append(t)
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
        if not s:
            return 0.0

        s = str(s).strip()

        # -------------------------------
        # UNIVERSAL NORMALIZATION FIX
        # Handles broken PDF numbers like:
        # "29 083. 00" → "29083.00"
        # "+ 29. 083 .00" → "+29083.00"
        # "29.083.00" → "29083.00"
        # -------------------------------

        # Remove inner spaces/dots between digits
        cleaned = re.sub(r'(?<=\d)[\s\.]+(?=\d)', '', s)

        # Remove thousand commas
        cleaned = cleaned.replace(",", "")

        # If number is like 2908300 → try forcing decimal if pattern broken
        # But primary: extract normal sign + decimal
        match = re.search(r'([+\-]?)\s*(\d+(?:\.\d{1,2})?)', cleaned)
        if match:
            sign = match.group(1)
            num = match.group(2)
            try:
                val = float(num)
            except:
                val = 0.0
            if sign == '-':
                return -abs(val)
            return val

        # Fallback: digits only
        digits = re.sub(r'[^\d\.]', '', cleaned)
        try:
            return float(digits) if digits else 0.0
        except:
            return 0.0


    def _infer_vendor_from_description(self, desc: str) -> str:
        if not desc:
            return 'UNKNOWN'
        # Remove long GUID-like tokens
        desc = re.sub(r'[0-9a-fA-F\-]{8,}', ' ', desc)
        # Remove words like TRANSF, DR, CR, OGT, ICT, IB, W2W
        desc = re.sub(r'\b(TRANSF|CR|DR|OGT|ICT|IB|W2W|PUR|POS|PAYMENT|DEBIT|CREDIT)\b', ' ', desc, flags=re.IGNORECASE)
        # Remove extra whitespace, commas etc.
        desc = re.sub(r'[\|\-\:\,\/]+', ' ', desc)
        desc = re.sub(r'\s{2,}', ' ', desc).strip()
        if not desc:
            return 'UNKNOWN'
        # If description contains location/city repeated, trim to first 60 chars
        vendor = desc[:60]
        # If vendor is just a time or numeric, mark UNKNOWN
        if re.fullmatch(r'[\d\W]+', vendor):
            return 'UNKNOWN'
        return vendor


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
            key = (self._normalize_date(t.date), round(t.amount, 2), self._normalize_text(t.vendor))
            if key in seen:
                existing = seen[key]
                # prefer non-reviewed entry
                if existing.needs_review and not t.needs_review:
                    seen[key] = t
            else:
                seen[key] = t
                order.append(key)
        result = [seen[k] for k in order]
        return result

    def _normalize_text(self, s: str) -> str:
        return re.sub(r'\W+', '', (s or '').lower())

    def _normalize_date(self, s: str) -> str:
        if not s:
            return ''
        s = s.strip()
        # try formats
        fmts = ['%d %b, %Y', '%d %b %Y', '%Y-%m-%d']
        for f in fmts:
            try:
                d = datetime.strptime(s, f)
                return d.strftime('%Y-%m-%d')
            except Exception:
                continue
        # try common '1 Oct, 2025' -> %d %b, %Y
        try:
            d = datetime.strptime(s.replace(',', ''), '%d %b %Y')
            return d.strftime('%Y-%m-%d')
        except Exception:
            pass
        return s

    def _categorize_by_keywords(self, t: Transaction) -> str:
        text = f"{t.vendor} {t.description}".lower()
        for cat, keywords in self.CATEGORY_RULES.items():
            for kw in keywords:
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
        out = out[['Source/Vendor', 'Transaction Count', 'Subtotal ($)']]
        return out

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
        out = out[['Category', 'Vendor', 'Transaction Count', 'Subtotal ($)']]
        return out

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
# Streamlit UI (main)
# ----------------------------
st.set_page_config(
    page_title="Bank Statement Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight:700; color:#0b5ed7; }
    .sub-header { color:#333; margin-bottom:0.6rem;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">💼 Bank Statement Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload bank statements (PDF / Word / CSV). App will auto-extract, categorize and produce summaries.</p>', unsafe_allow_html=True)

# Session state
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
if 'parsing_metadata' not in st.session_state:
    st.session_state.parsing_metadata = {}
if 'statistics' not in st.session_state:
    st.session_state.statistics = {}

# Centered uploader
st.markdown("<h4 style='text-align:center;'>📤 Upload Statement</h4>", unsafe_allow_html=True)
u_cols = st.columns([1, 2, 1])
with u_cols[1]:
    uploaded_file = st.file_uploader(" ", type=['pdf', 'doc', 'docx', 'csv'], help="Supported: PDF, Word, CSV", accept_multiple_files=False)

# Hidden helper: quick test of uploaded file path (useful for local debugging)
# You can uncomment and set local_path variable to test file without uploading
# local_test_path = "/mnt/data/sadapay_account_statement_2025-10-01_2025-11-24.pdf"
# if local_test_path and not uploaded_file:
#     with open(local_test_path, 'rb') as f:
#         uploaded_file = io.BytesIO(f.read())
#         uploaded_file.name = local_test_path.split('/')[-1]
#         uploaded_file.size = f.tell()

if uploaded_file:
    st.info(f"📄 File: {uploaded_file.name}")
    st.info(f"📊 Size: {uploaded_file.size / 1024:.2f} KB")
    bcols = st.columns([1, 1, 1])
    with bcols[1]:
        if st.button("🔄 Process Statement", type="primary", use_container_width=True):
            with st.spinner("Processing statement..."):
                try:
                    file_bytes = uploaded_file.read()
                    doc_parser = DocumentParser()
                    lines, is_readable, unreadable_pages = doc_parser.parse_document(file_bytes, uploaded_file.name)

                    if not is_readable or len(lines) < 5:
                        st.error("⚠️ Document appears unreadable or has insufficient text content.")
                        if unreadable_pages:
                            st.warning(f"Unreadable pages: {unreadable_pages}. Consider OCR or clearer scan.")
                        st.session_state.transactions = []
                    else:
                        stmt_parser = BankStatementParser()
                        transactions, metadata = stmt_parser.parse_statement(lines)

                        if not transactions:
                            st.warning("⚠️ No transactions found. Try exporting as CSV if possible, or send sample for tuning.")
                            st.session_state.transactions = []
                        else:
                            categorizer = TransactionCategorizer()
                            transactions = categorizer.process_transactions(transactions)
                            transactions = categorizer.detect_duplicates(transactions)
                            report_gen = ReportGenerator()
                            statistics = report_gen.generate_summary_statistics(transactions)

                            st.session_state.transactions = transactions
                            st.session_state.parsing_metadata = {
                                **metadata,
                                'total_lines': len(lines),
                                'unreadable_pages': unreadable_pages,
                                'is_readable': is_readable
                            }
                            st.session_state.statistics = statistics

                            st.success(f"✅ Processed {len(transactions)} transactions successfully!")
                except Exception as e:
                    st.error(f"❌ Error processing document: {e}")
                    logger.exception("Processing error")
                    st.session_state.transactions = []

if not st.session_state.transactions:
    st.info("👆 Please upload a bank statement file to begin analysis.")
    st.stop()

transactions = st.session_state.transactions
metadata = st.session_state.parsing_metadata
statistics = st.session_state.statistics

# Warnings
if metadata.get('unreadable_pages'):
    st.warning(f"⚠️ Some pages ({metadata['unreadable_pages']}) were unreadable. Check 'Needs Review' tab.")

needs_review_count = statistics.get('Transactions Needing Review', 0)
if needs_review_count > 0:
    st.warning(f"⚠️ {needs_review_count} transaction(s) need manual review. See 'Needs Review' tab.")

# Summary
st.header("📊 Summary Statistics")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Deposits", f"${statistics.get('Total Deposit Amount', 0):,.2f}", f"{statistics.get('Total Deposits', 0)} transactions")
with c2:
    st.metric("Total Withdrawals", f"${statistics.get('Total Withdrawal Amount', 0):,.2f}", f"{statistics.get('Total Withdrawals', 0)} transactions")
with c3:
    st.metric("Net Income", f"${statistics.get('Net Income', 0):,.2f}")
with c4:
    st.metric("Total Transactions", statistics.get('Total Transactions', 0), f"{needs_review_count} need review" if needs_review_count > 0 else "All clear")

# Tabs and reports
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💰 Deposits Summary",
    "💸 Withdrawals Summary",
    "📈 P&L Report",
    "⚠️ Needs Review",
    "📋 All Transactions"
])

report_gen = ReportGenerator()

with tab1:
    st.subheader("Deposits Summary")
    deposits_df = report_gen.generate_deposits_summary(transactions)
    if not deposits_df.empty:
        st.dataframe(deposits_df[['Source/Vendor', 'Transaction Count', 'Subtotal ($)']], use_container_width=True, hide_index=True)
        with st.expander("View Details"):
            st.dataframe(deposits_df, use_container_width=True, hide_index=True)
    else:
        st.info("No deposits found.")

with tab2:
    st.subheader("Withdrawals Summary")
    withdrawals_df = report_gen.generate_withdrawals_summary(transactions)
    if not withdrawals_df.empty:
        st.dataframe(withdrawals_df, use_container_width=True, hide_index=True)
        with st.expander("View Details"):
            st.dataframe(withdrawals_df, use_container_width=True, hide_index=True)
    else:
        st.info("No withdrawals found.")

with tab3:
    st.subheader("Profit & Loss Report")
    pl_df = report_gen.generate_pl_report(transactions)
    if not pl_df.empty:
        st.dataframe(pl_df, use_container_width=True, hide_index=True)
        if 'Type' in pl_df.columns:
            expense_df = pl_df[(pl_df['Type'] == 'Expense') & (pl_df['Category'] != 'TOTAL EXPENSES')]
            if not expense_df.empty:
                st.subheader("Expenses by Category")
                chart_df = expense_df[['Category', 'Amount ($)']].copy()
                chart_df['Amount ($)'] = chart_df['Amount ($)'].abs()
                st.bar_chart(chart_df.set_index('Category'))
    else:
        st.info("No P&L data available.")

with tab4:
    st.subheader("⚠️ Transactions Needing Review")
    review_df = report_gen.generate_needs_review_section(transactions)
    if not review_df.empty:
        st.warning(f"Found {len(review_df)} transaction(s) needing review.")
        st.dataframe(review_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ No transactions need review!")

with tab5:
    st.subheader("All Transactions")
    all_trans_df = pd.DataFrame([{
        'Date': t.date or 'N/A',
        'Type': t.transaction_type.title(),
        'Vendor': t.vendor or 'N/A',
        'Category': t.category or 'Uncategorized',
        'Amount': f"${t.amount:,.2f}" if t.amount != 0 else 'N/A',
        'Description': (t.description[:120] + '...') if t.description and len(t.description) > 120 else (t.description or 'N/A'),
        'Needs Review': '⚠️ Yes' if t.needs_review else '✅ No'
    } for t in transactions])

    f1, f2, f3 = st.columns(3)
    with f1:
        type_filter = st.selectbox("Filter by Type", ['All', 'Deposit', 'Withdrawal'], key='type_filter')
    with f2:
        category_filter = st.selectbox("Filter by Category", ['All'] + sorted(all_trans_df['Category'].unique().tolist()), key='category_filter')
    with f3:
        review_filter = st.selectbox("Review Status", ['All', 'Needs Review', 'OK'], key='review_filter')

    filtered_df = all_trans_df.copy()
    if type_filter != 'All':
        filtered_df = filtered_df[filtered_df['Type'] == type_filter]
    if category_filter != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == category_filter]
    if review_filter == 'Needs Review':
        filtered_df = filtered_df[filtered_df['Needs Review'] == '⚠️ Yes']
    elif review_filter == 'OK':
        filtered_df = filtered_df[filtered_df['Needs Review'] == '✅ No']

    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    st.info(f"Showing {len(filtered_df)} of {len(all_trans_df)} transactions")

# Downloads
st.header("📥 Download Reports")
d1, d2, d3 = st.columns(3)

with d1:
    deposits_csv = report_gen.generate_deposits_summary(transactions).to_csv(index=False)
    st.download_button("⬇️ Download Deposits (CSV)", deposits_csv.encode('utf-8'), "deposits_summary.csv", "text/csv", use_container_width=True)

with d2:
    withdrawals_csv = report_gen.generate_withdrawals_summary(transactions).to_csv(index=False)
    st.download_button("⬇️ Download Withdrawals (CSV)", withdrawals_csv.encode('utf-8'), "withdrawals_summary.csv", "text/csv", use_container_width=True)

with d3:
    pl_csv = report_gen.generate_pl_report(transactions).to_csv(index=False)
    st.download_button("⬇️ Download P&L (CSV)", pl_csv.encode('utf-8'), "pl_report.csv", "text/csv", use_container_width=True)

st.markdown("---")
if st.button("📊 Download Complete Report (Excel)", type="primary", use_container_width=True):
    try:
        excel_filename = report_gen.export_to_excel(transactions)
        with open(excel_filename, 'rb') as f:
            st.download_button("⬇️ Download Excel Report", f.read(), "bank_statement_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    except Exception as e:
        st.error(f"Error generating Excel report: {e}")

st.success("✅ Reports generated. Please verify totals match your bank statement.")
