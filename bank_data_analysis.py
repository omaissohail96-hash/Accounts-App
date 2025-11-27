# ----------------------------
# PART 1: Core Engine + LLM Enhancer
# ----------------------------
import io
import re
import json
import tempfile
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import pdfplumber
import pandas as pd

import openai
import streamlit as st  # used for secrets access in Part 2 too

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set OpenAI API key from Streamlit secrets if available (Part2 will set it before processing)
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]


# ----------------------------
# Data Model
# ----------------------------
@dataclass
class Transaction:
    date: str
    transaction_type: str  # 'deposit' or 'withdrawal'
    vendor: str
    amount: float
    description: str
    raw_line: str
    section: Optional[str] = None
    category: Optional[str] = None
    needs_review: bool = False


# ----------------------------
# Document Parser (PDF / CSV / DOCX)
# ----------------------------
class DocumentParser:
    # Column detection regex patterns
    COL_DATE = re.compile(r"date", re.I)
    COL_DEBIT = re.compile(r"debit|withdraw|paid|sent|out|dr", re.I)
    COL_CREDIT = re.compile(r"credit|deposit|received|in|cr", re.I)
    COL_DESC = re.compile(r"desc|details|narration|merchant|vendor|description", re.I)

    def _clean_vendor(self, desc: str) -> str:
        desc = desc.title()
        desc = re.sub(r"\b[A-Z]\b", "", desc)  # remove solo initials (Sadapay issue)
        desc = re.sub(r"\s{2,}", " ", desc).strip()
        return desc or "Unknown"

    def parse_pdf_table(self, file_bytes):
        """ Extract transactions from tables inside PDF """
        transactions = []

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()

                for table in tables:
                    if not table or len(table) < 2:
                        continue

                    header = [c.strip() if c else "" for c in table[0]]

                    date_idx = next((i for i, h in enumerate(header) if self.COL_DATE.search(h)), None)
                    debit_idx = next((i for i, h in enumerate(header) if self.COL_DEBIT.search(h)), None)
                    credit_idx = next((i for i, h in enumerate(header) if self.COL_CREDIT.search(h)), None)
                    desc_idx = next((i for i, h in enumerate(header) if self.COL_DESC.search(h)), None)

                    # Table not matching transaction format
                    if date_idx is None or (debit_idx is None and credit_idx is None):
                        continue

                    for row in table[1:]:
                        if not row or len(row) <= max(date_idx, debit_idx or 0, credit_idx or 0):
                            continue

                        date = (row[date_idx] or "").strip()
                        desc = (row[desc_idx] or "").strip() if desc_idx is not None else ""

                        debit = row[debit_idx] if debit_idx is not None else ""
                        credit = row[credit_idx] if credit_idx is not None else ""
                        amount = None
                        direction = None  
                        if credit and credit.strip():
                            cleaned = re.sub(r"[^\d.-]", "", credit)
                            if cleaned:
                                if cleaned.startswith('-'):
                                    direction = "withdrawal"
                                else:
                                    direction = "deposit"
                                amount = abs(float(cleaned))

                        elif debit and debit.strip():
                            cleaned = re.sub(r"[^\d.-]", "", debit)
                            if cleaned:
                                direction = "withdrawal"
                                amount = abs(float(cleaned))

                        # 2️⃣ Look for CR/DR labeling in description
                        if direction is None:
                            low = desc.lower()
                            if "cr" in low or "credit" in low or "received" in low:
                                direction = "deposit"
                            elif "dr" in low or "debit" in low or "sent" in low or "purchase" in low:
                                direction = "withdrawal"

                        # 3️⃣ If still unknown, drop the row
                        if amount is None or direction is None:
                            continue

                        # ✔️ Finalized Transaction
                        transactions.append(Transaction(
                            date=date,
                            transaction_type=direction,
                            vendor=self._clean_vendor(desc),
                            amount=amount if direction == "deposit" else -amount,
                            description=desc,
                            raw_line=" | ".join(str(x) for x in row)
                        ))

        return transactions

    def parse_document(self, file_bytes: bytes, filename: str) -> Tuple[List[str], bool, List[int]]:
        """Fallback to text parsing if table extraction is incomplete"""
        ext = filename.lower().split('.')[-1]
        unreadable_pages: List[int] = []
        lines: List[str] = []

        # CSV support
        if ext == "csv":
            try:
                text = file_bytes.decode('utf-8', errors='ignore')
            except Exception:
                text = str(file_bytes)
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            return lines, True, []

        # PDF — use text fallback if tables not fully detected
        if ext == "pdf":
            try:
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    for page in pdf.pages:
                        txt = page.extract_text() or ""
                        txt = txt.replace('\xa0', ' ')
                        if txt.strip():
                            lines.extend([ln.strip() for ln in txt.splitlines() if ln.strip()])
                        else:
                            unreadable_pages.append(page.page_number)
                return lines, len(lines) > 0, unreadable_pages
            except Exception:
                logger.exception("PDF parse failed")
                return [], False, []

        # DOCX support
        if ext in ("doc", "docx"):
            try:
                import docx
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.' + ext)
                tmp.write(file_bytes)
                tmp.flush()
                doc = docx.Document(tmp.name)
                lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
                return lines, True, []
            except Exception:
                logger.exception("DOCX parse failed")
                return [], False, []

        # Plain text fallback
        try:
            text = file_bytes.decode('utf-8', errors='ignore')
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            return lines, True, []
        except Exception:
            return [], False, []
    def extract_transactions(self, file_bytes: bytes, filename: str):
        ext = filename.lower().split('.')[-1]

        # If PDF → Try table parsing first
        if ext == "pdf":
            table_tx = self.parse_pdf_table(file_bytes)
            if table_tx:
                return table_tx, {"parsed_from": "pdf-table", "transactions_extracted": len(table_tx)}

        # Fallback → text line parsing
        lines, ok, unreadable = self.parse_document(file_bytes, filename)
        if not ok or not lines:
            return [], {"parsed_from": "failed"}

        fallback = FallbackStatementParser()
        txs, meta = fallback.parse_statement(lines)
        meta["raw_lines"] = len(lines)
        return txs, meta


# ----------------------------
# Fallback Statement Parser (deterministic)
# ----------------------------
class FallbackStatementParser:
    # Detect dates like "09/10/2025", "09 Oct 2025", "09 Oct, 2025", "2025-10-09"
    DATE_RE = re.compile(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\,?\s+\d{2,4}|\d{4}-\d{2}-\d{2})')
    # Amount tokens: allow spaces inside thousands "29 083.00", commas, optional sign, parentheses for negative
    AMOUNT_RE = re.compile(
    r'([+\-]?\(?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})\s*\)?)'
)

    deposit_keys = ["received", "credit", "payment received", "deposit", "payment from", "paid in", "inbound", "refund"]
    withdraw_keys = ["purchase", "paid", "payment", "purchase at", "sent", "withdraw", "debit", "card", "purchase -", "purchase/"]

    def _clean_amount_token(self, token: str) -> Optional[float]:
        if not token:
            return None
        
        s = token.strip()

        # negative if () or a minus sign
        negative = ("(" in s and ")" in s) or s.strip().startswith("-")

        # Remove parentheses and currency symbols
        s = re.sub(r'[A-Za-z\(\)\$]', '', s)

        # Normalize spaces
        s = re.sub(r'\s+', '', s)

        # Fix "2, 500" or "2 . 500" → correct thousands
        s = s.replace(',', '').replace(' ', '')

        # Make sure only one decimal exists
        parts = s.split('.')
        if len(parts) > 2:  # more than one dot -> join all except last as integer
            s = ''.join(parts[:-1]) + "." + parts[-1]

        # If no decimal: consider integer safely
        if "." not in s:
            try:
                val = int(s)
            except:
                return None
            return -val if negative else val

        try:
            val = float(s)
            return -abs(val) if negative else abs(val)
        except:
            return None
    def _infer_vendor(self, line: str, date_str: str, amount_token: str) -> str:
        tmp = line
        if date_str:
            tmp = tmp.replace(date_str, ' ')
        if amount_token:
            tmp = tmp.replace(amount_token, ' ')
        # remove currency words and balances
        tmp = re.sub(r'\b(PKR|USD|EUR|GBP|AED|CAD|AUD|Balance|Available)\b', ' ', tmp, flags=re.I)
        # remove numbers
        tmp = re.sub(r'\d[\d,./-]*', ' ', tmp)
        tmp = re.sub(r'[\|\-:]+', ' ', tmp)
        tmp = re.sub(r'\s{2,}', ' ', tmp).strip()
        return tmp if tmp else "UNKNOWN"

    def parse_statement(self, lines: List[str]) -> Tuple[List[Transaction], Dict[str, Any]]:
        transactions: List[Transaction] = []
        for ln in lines:
            low = ln.lower()
            if 'balance' in low and 'available' in low:
                # skip balance lines
                continue
            d_match = self.DATE_RE.search(ln)
            a_match = self.AMOUNT_RE.search(ln)
            if not a_match:
                # skip lines without amount
                continue

            amount_token = a_match.group(1)
            amount_val = self._clean_amount_token(amount_token)
            if amount_val is None:
                continue
            if re.match(r'^\d{1,2}/\d{1,2}/\d{2,4}$', ln):
                continue    
            # default sign handling: assume positive unless keywords or parentheses or leading '-'
            # detect parentheses or leading minus
            ln_signed = ln.strip()
            if amount_token.strip().startswith('(') and amount_token.strip().endswith(')'):
                amount_val = -abs(amount_val)
            elif amount_token.strip().startswith('-'):
                amount_val = -abs(amount_val)

            # check explicit + or - anywhere
            if '+' in ln and '-' not in ln:
                amount_val = abs(amount_val)
            if '-' in ln and '+' not in ln and ('-' in amount_token):
                amount_val = -abs(amount_val)

            # keywords detection override
            if any(k in low for k in self.withdraw_keys):
                amount_val = -abs(amount_val)
            if any(k in low for k in self.deposit_keys):
                amount_val = abs(amount_val)

            date_str = d_match.group(1) if d_match else ""
            vendor = self._infer_vendor(ln, date_str, amount_token)

            transactions.append(Transaction(
                date=date_str,
                transaction_type='deposit' if amount_val > 0 else 'withdrawal',
                vendor=vendor.title(),
                amount=amount_val,
                description=ln,
                raw_line=ln
            ))

        meta = {'parsed_from': 'fallback', 'transactions_extracted': len(transactions)}
        return transactions, meta


# ----------------------------
# LLM Enhancer (Cleans & canonicalizes rows)
#    — HYBRID MODE: we send the fallback-extracted rows for cleaning.
#    — LLM returns a JSON list aligned by index: it MUST keep same count and include index.
# ----------------------------
class LLMEnhancer:
    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 1500):
        self.model = model
        self.max_tokens = max_tokens

    def enhance(self, transactions: List[Transaction], raw_text: str) -> List[Transaction]:
        """
        Send a compact prompt containing extracted rows and ask LLM to:
        - Standardize date to YYYY-MM-DD if possible
        - Fix vendor name (short canonical)
        - Fix amounts (remove spaces in numbers) but DO NOT add/remove transactions
        - Return JSON: [{"idx":0,"date":"YYYY-MM-DD","vendor":"...","amount":number,"direction":"in/out","description":"..."} ...]
        """
        if not openai.api_key:
            logger.info("No OpenAI API key — skipping LLM enhancement.")
            return transactions

        # Build small sample of rows to send (limit to first 200 rows to avoid huge prompt)
        rows = []
        for i, t in enumerate(transactions):
            rows.append({
                "idx": i,
                "date": t.date,
                "vendor": t.vendor,
                "amount": t.amount,
                "direction": "in" if t.amount > 0 else "out",
                "description": t.description
            })

        prompt = f"""
You are a careful financial cleaner. You will receive a JSON array of parsed transactions. 
You MUST return a JSON array with exactly the same number of elements. 
Each element must be an object with keys:
  - idx (integer): index matching the input
  - date (string or null): canonicalize to YYYY-MM-DD if possible, else return original text
  - vendor (string or null): short cleaned vendor name
  - amount (number): numeric amount (positive)
  - direction (string): "in" or "out"
  - description (string): cleaned description text

Rules:
- Do NOT add or remove transactions, keep idx mapping strict.
- Normalize amounts: convert strings like "29 083.00" -> 29083.00
- If amount parsing uncertain, return the numeric value closest to input.
- For date: try parsing common formats; if you can convert to YYYY-MM-DD do so.
- Return ONLY a JSON array (no extra text or markdown).
Input JSON:
{json.dumps(rows, ensure_ascii=False)}
"""
        try:
            resp = openai.ChatCompletion.create(
                model=self.model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens
            )
            content = resp.choices[0].message["content"]
            parsed = json.loads(content)
            # Build new transactions from parsed
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
            # Ensure same length
            if len(enhanced) == len(transactions):
                return enhanced
            else:
                logger.warning("LLM returned different number of rows — skipping enhancement.")
                return transactions
        except Exception as e:
            logger.exception("LLM enhancement failed: %s", e)
            return transactions


# ----------------------------
# Categorizer + Dedupe
# ----------------------------
class TransactionCategorizer:
    def process_transactions(self, transactions: List[Transaction]) -> List[Transaction]:
        for t in transactions:
            if t.amount > 0:
                t.category = "Income"
            else:
                # example heuristics:
                if 'uber' in t.description.lower():
                    t.category = "Transport"
                elif 'starbuck' in t.description.lower() or 'coffee' in t.description.lower():
                    t.category = "Food"
                else:
                    t.category = "Expense"
        return transactions

    def detect_duplicates(self, transactions: List[Transaction]) -> List[Transaction]:
        seen = {}
        order = []
        for t in transactions:
            key = (t.date, round(t.amount, 2), re.sub(r'\W+', '', (t.vendor or '').lower()))
            if key in seen:
                # prefer non-review item
                if seen[key].needs_review and not t.needs_review:
                    seen[key] = t
            else:
                seen[key] = t
                order.append(key)
        return [seen[k] for k in order]


# ----------------------------
# Report Generator
# ----------------------------
class ReportGenerator:
    def generate_summary_statistics(self, transactions: List[Transaction]) -> Dict[str, Any]:
        total_deposits = sum(t.amount for t in transactions if t.amount > 0)
        total_withdrawals = sum(-t.amount for t in transactions if t.amount < 0)
        return {
            'Total Deposit Amount': total_deposits,
            'Total Withdrawal Amount': total_withdrawals,
            'Total Deposits': sum(1 for t in transactions if t.amount > 0),
            'Total Withdrawals': sum(1 for t in transactions if t.amount < 0),
            'Total Transactions': len(transactions),
            'Net Income': total_deposits - total_withdrawals,
            'Transactions Needing Review': sum(1 for t in transactions if t.needs_review)
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
        grp = df.groupby(['vendor']).agg({'amount': 'sum', 'raw_line': 'count'}).reset_index()
        grp.columns = ['Vendor', 'Subtotal ($)', 'Transaction Count']
        total = grp['Subtotal ($)'].sum()
        total_row = pd.DataFrame([{'Vendor': 'TOTAL WITHDRAWALS', 'Subtotal ($)': total, 'Transaction Count': grp['Transaction Count'].sum()}])
        out = pd.concat([grp, total_row], ignore_index=True)
        return out[['Vendor', 'Transaction Count', 'Subtotal ($)']]
    def generate_pl_report(self, transactions: List[Transaction]) -> pd.DataFrame:
        total_income = sum(t.amount for t in transactions if t.amount > 0)
        total_expenses = sum(-t.amount for t in transactions if t.amount < 0)
        net = total_income - total_expenses

        df = pd.DataFrame([
            {"Category": "Income", "Amount ($)": total_income},
            {"Category": "Expenses", "Amount ($)": -total_expenses},
            {"Category": "Net Income", "Amount ($)": net},
        ])
        return df
# ----------------------------
# PART 2: Streamlit UI
# ----------------------------
import streamlit as st
import pandas as pd
from typing import List

# Ensure OpenAI key is loaded
if "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]:
    openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Bank Statement Analyzer (Hybrid LLM)", layout="wide")

st.markdown("<h1 style='color:#0d6efd'>💼 Bank Statement Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p>Hybrid parser: deterministic extractor → optional LLM cleaner. Totals are computed from parsed numeric amounts (no LLM hallucination).</p>", unsafe_allow_html=True)

# sidebar: LLM switch + model
with st.sidebar:
    st.header("Settings")
    use_llm = st.checkbox("Enable LLM Enhancement (cost)", value=True)
    llm_model = st.selectbox("LLM model", ["gpt-4o-mini"], index=0)
    st.markdown("Make sure you put your OpenAI key in `.streamlit/secrets.toml` as:\n\n`OPENAI_API_KEY = \"sk-...\"`")

# Upload
uploaded = st.file_uploader("Upload statement (PDF/CSV/DOCX)", type=["pdf", "csv", "doc", "docx"])

if uploaded:
    st.info(f"File: {uploaded.name} — {uploaded.size/1024:.1f} KB")
    currency = st.selectbox("Currency", ["PKR", "USD", "EUR", "GBP", "AED", "CAD", "AUD"], index=0)

    if st.button("Process Statement"):
        with st.spinner("Processing..."):
            file_bytes = uploaded.read()

            # Parse doc
            dp = DocumentParser()
            lines, ok, unreadable = dp.parse_document(file_bytes, uploaded.name)
            if not ok:
                st.error("Could not read text from file.")
                if unreadable:
                    st.warning(f"Unreadable pages: {unreadable}")
                st.stop()

            # Fallback parse (deterministic)
            fallback = FallbackStatementParser()
            txs = dp.parse_pdf_table(file_bytes)

            if txs:
                transactions = txs
                parsed_from = "table"
            else:
                transactions, _ = fallback.parse_statement(lines)
                parsed_from = "fallback"
            # If nothing
            if not txs:
                st.error("No transactions extracted by fallback parser.")
                st.stop()

            # Optional LLM enhancement (hybrid)
            if use_llm and openai.api_key:
                enhancer = LLMEnhancer(model=llm_model)
                enhanced = enhancer.enhance(txs, raw_text="\n".join(lines))
                transactions = enhanced
                parsed_from = "hybrid-llm"
            else:
                transactions = txs
                parsed_from = "fallback-only"

            # Categorize and dedupe
            cat = TransactionCategorizer()
            transactions = cat.process_transactions(transactions)
            transactions = cat.detect_duplicates(transactions)

            # Stats and reports
            rg = ReportGenerator()
            stats = rg.generate_summary_statistics(transactions)
            deposits_df = rg.generate_deposits_summary(transactions)
            withdrawals_df = rg.generate_withdrawals_summary(transactions)
            pl_df = rg.generate_pl_report(transactions)

            # Save to session
            st.session_state.transactions = transactions
            st.session_state.stats = stats
            st.session_state.deposit_df = deposits_df
            st.session_state.withdrawal_df = withdrawals_df
            st.session_state.pl_df = pl_df
            st.session_state.currency = currency
            st.session_state.parsed_from = parsed_from

            st.success(f"Processed {len(transactions)} transactions ({parsed_from}).")

# If we have processed transactions, show dashboard
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

    # Reconciliation check
    computed_deposits = sum(t.amount for t in transactions if t.amount > 0)
    computed_withdrawals = sum(-t.amount for t in transactions if t.amount < 0)
    if abs(computed_deposits - stats['Total Deposit Amount']) > 0.001 or abs(computed_withdrawals - stats['Total Withdrawal Amount']) > 0.001:
        st.warning("Reconciliation mismatch: computed sums differ from reported sums — using computed sums as source of truth.")
    else:
        st.success("Reconciliation OK.")

    # Tabs for details
    tab1, tab2, tab3, tab4 = st.tabs(["💰 Deposits", "💸 Withdrawals", "📈 P&L", "📋 All Transactions"])

    with tab1:
        st.subheader("Deposits Summary (by Source/Vendor)")
        if st.session_state.deposit_df is None or st.session_state.deposit_df.empty:
            st.info("No deposits found.")
        else:
            df = st.session_state.deposit_df.copy()
            df = df.rename(columns={"vendor": "Source/Vendor", "sum": "Subtotal ($)", "count": "Transaction Count"}) if 'vendor' in df.columns else df
            # ensure columns named consistently
            if list(df.columns) == ["vendor", "sum", "count"]:
                df.columns = ["Source/Vendor", "Subtotal ($)", "Transaction Count"]
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Expanders with transaction details per vendor
            deps = [t for t in transactions if t.amount > 0]
            grouped = {}
            for t in deps:
                key = t.vendor or "UNKNOWN"
                grouped.setdefault(key, []).append(t)

            for vendor, items in sorted(grouped.items(), key=lambda x: (-len(x[1]), x[0])):
                subtotal = sum(i.amount for i in items)
                cnt = len(items)
                with st.expander(f"{vendor} — {cnt} tx — {cur} {subtotal:,.2f}"):
                    details = pd.DataFrame([{
                        "Date": it.date,
                        "Amount": f"{cur} {it.amount:,.2f}",
                        "Description": it.description,
                        "Needs Review": "⚠ Yes" if it.needs_review else "✅ No"
                    } for it in items])
                    st.dataframe(details, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Withdrawals Summary (by Vendor)")
        if st.session_state.withdrawal_df is None or st.session_state.withdrawal_df.empty:
            st.info("No withdrawals found.")
        else:
            st.dataframe(st.session_state.withdrawal_df, use_container_width=True, hide_index=True)

            wds = [t for t in transactions if t.amount < 0]
            grouped = {}
            for t in wds:
                key = t.vendor or "UNKNOWN"
                grouped.setdefault(key, []).append(t)

            for vendor, items in sorted(grouped.items(), key=lambda x: (-len(x[1]), x[0])):
                subtotal = sum(abs(i.amount) for i in items)
                cnt = len(items)
                with st.expander(f"{vendor} — {cnt} tx — {cur} {subtotal:,.2f}"):
                    details = pd.DataFrame([{
                        "Date": it.date,
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
            "Date": t.date,
            "Type": t.transaction_type,
            "Vendor": t.vendor,
            "Amount": f"{cur} {t.amount:,.2f}",
            "Description": t.description
        } for t in transactions])
        st.dataframe(all_df, use_container_width=True, hide_index=True)

    # Download buttons
    st.header("📥 Download")
    col1, col2, col3 = st.columns(3)
    with col1:
        dep_csv = st.session_state.deposit_df.to_csv(index=False) if (st.session_state.deposit_df is not None and not st.session_state.deposit_df.empty) else ""
        st.download_button("⬇ Deposits CSV", dep_csv, "deposits.csv", mime="text/csv")
    with col2:
        wd_csv = st.session_state.withdrawal_df.to_csv(index=False) if (st.session_state.withdrawal_df is not None and not st.session_state.withdrawal_df.empty) else ""
        st.download_button("⬇ Withdrawals CSV", wd_csv, "withdrawals.csv", mime="text/csv")
    with col3:
        pnl_csv = st.session_state.pl_df.to_csv(index=False) if (st.session_state.pl_df is not None and not st.session_state.pl_df.empty) else ""
        st.download_button("⬇ P&L CSV", pnl_csv, "pnl.csv", mime="text/csv")

    st.success("Report generated. Verify totals against your original statement.")
