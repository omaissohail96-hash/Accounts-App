# bank_data_analysis.py
# Single-file hybrid Bank Statement Analyzer (deterministic + optional LLM)
# Save as bank_data_analysis.py and run: streamlit run bank_data_analysis.py

import io
import os
import re
import json
import tempfile
import logging
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import pdfplumber
import pandas as pd
import streamlit as st

# Optional OpenAI usage (LLM enhancement). If you don't want LLM, leave secrets empty.
try:
    import openai
except Exception:
    openai = None

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
# Helper: robust amount cleaner (single place)
# ----------------------------
def clean_amount_token(token: str) -> Optional[float]:
    """
    Convert tokens like:
      "29 083.00", "2,500", "(2,500.00)", "+2500", "-2500", "2500.00"
    Returns signed float (negative if parentheses or leading '-'), or None if can't parse.
    """
    if not token:
        return None

    s = str(token).strip()

    # If token contains many digits (like trace IDs) treat with caution:
    digits_only = re.sub(r'\D', '', s)
    if len(digits_only) >= 13:
        # too many digits — likely not a monetary value
        return None

    negative = False
    # parentheses = negative (common)
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()

    # leading sign
    if s.startswith("-"):
        negative = True
        s = s[1:].strip()
    elif s.startswith("+"):
        s = s[1:].strip()

    # remove currency symbols and letters
    s = re.sub(r'[A-Za-z₹₨$€£,]', '', s)
    # normalize spaces between thousands "29 083.00"
    s = s.replace(" ", "")
    # handle multiple dots like "1.234.567,89" — we don't try locale detection here; keep as simple float
    # if there are more than 1 dot, join all but last as integer part
    parts = s.split(".")
    if len(parts) > 2:
        s = "".join(parts[:-1]) + "." + parts[-1]

    # If empty after cleaning:
    if not re.search(r'\d', s):
        return None

    try:
        val = float(s)
    except Exception:
        # try integer
        try:
            val = int(s)
        except Exception:
            return None

    return -abs(val) if negative else abs(val)

# ----------------------------
# Document parser: PDF tables + text fallback + CSV/DOCX
# ----------------------------
class DocumentParser:
    COL_DATE = re.compile(r"date", re.I)
    COL_DEBIT = re.compile(r"debit|withdraw|paid|sent|out|dr", re.I)
    COL_CREDIT = re.compile(r"credit|deposit|received|in|cr", re.I)
    COL_DESC = re.compile(r"desc|details|narration|merchant|vendor|description", re.I)

    def _clean_vendor(self, desc: str) -> str:
        if not desc:
            return "UNKNOWN"
        v = re.sub(r'\s{2,}', ' ', desc).strip()
        v = " ".join([w.upper() if w.isupper() and len(w) <= 4 else w.title() for w in v.split()])
        return v or "UNKNOWN"

    def parse_pdf_table(self, file_bytes: bytes) -> List[Transaction]:
        txs: List[Transaction] = []
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    if not tables:
                        continue
                    for table in tables:
                        if len(table) < 2:
                            continue
                        header = [ (c or "").strip() for c in table[0] ]
                        date_idx = next((i for i,h in enumerate(header) if self.COL_DATE.search(h)), None)
                        debit_idx = next((i for i,h in enumerate(header) if self.COL_DEBIT.search(h)), None)
                        credit_idx = next((i for i,h in enumerate(header) if self.COL_CREDIT.search(h)), None)
                        desc_idx = next((i for i,h in enumerate(header) if self.COL_DESC.search(h)), None)

                        if date_idx is None or (debit_idx is None and credit_idx is None):
                            continue

                        for row in table[1:]:
                            if not row or len(row) <= max(date_idx, debit_idx or 0, credit_idx or 0):
                                continue
                            date = (row[date_idx] or "").strip()
                            desc = (row[desc_idx] or "").strip() if desc_idx is not None else ""
                            debit = (row[debit_idx] or "").strip() if debit_idx is not None else ""
                            credit = (row[credit_idx] or "").strip() if credit_idx is not None else ""
                            amount = None
                            direction = None

                            # prefer credit column as deposit
                            if credit:
                                cleaned = re.sub(r'[^\d\-\.\(\),\s\+]', '', credit)
                                amt = clean_amount_token(cleaned)
                                if amt is not None:
                                    amount = abs(amt)
                                    direction = "deposit"

                            if amount is None and debit:
                                cleaned = re.sub(r'[^\d\-\.\(\),\s\+]', '', debit)
                                amt = clean_amount_token(cleaned)
                                if amt is not None:
                                    amount = abs(amt)
                                    direction = "withdrawal"

                            # fallback: detect CR/DR signs in desc if direction unknown
                            if amount is not None and direction is None:
                                low = desc.lower()
                                if "cr" in low or "credit" in low or "received" in low:
                                    direction = "deposit"
                                elif "dr" in low or "debit" in low or "sent" in low or "purchase" in low:
                                    direction = "withdrawal"

                            if amount is None or direction is None:
                                continue

                            signed = amount if direction == "deposit" else -amount

                            txs.append(Transaction(
                                date=date or "",
                                transaction_type=direction,
                                vendor=self._clean_vendor(desc),
                                amount=signed,
                                description=desc,
                                raw_line=" | ".join(str(x) for x in row)
                            ))
        except Exception as e:
            logger.exception("parse_pdf_table failed: %s", e)
        return txs

    def parse_document(self, file_bytes: bytes, filename: str) -> Tuple[List[str], bool, List[int]]:
        ext = filename.lower().split('.')[-1]
        unreadable_pages: List[int] = []
        lines: List[str] = []

        if ext == "csv":
            try:
                txt = file_bytes.decode("utf-8", errors="ignore")
            except Exception:
                txt = str(file_bytes)
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            return lines, True, []

        if ext == "pdf":
            try:
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    for page in pdf.pages:
                        txt = page.extract_text() or ""
                        txt = txt.replace('\xa0',' ')
                        if txt.strip():
                            lines.extend([ln.strip() for ln in txt.splitlines() if ln.strip()])
                        else:
                            unreadable_pages.append(page.page_number)
                return lines, len(lines) > 0, unreadable_pages
            except Exception as e:
                logger.exception("PDF text parse failed: %s", e)
                return [], False, []

        if ext in ("doc","docx"):
            try:
                import docx
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.'+ext)
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

    def extract_transactions(self, file_bytes: bytes, filename: str) -> Tuple[List[Transaction], Dict[str,Any]]:
        ext = filename.lower().split('.')[-1]
        if ext == "pdf":
            table_tx = self.parse_pdf_table(file_bytes)
            if table_tx:
                return table_tx, {"parsed_from":"pdf-table", "transactions_extracted": len(table_tx)}

        lines, ok, unreadable = self.parse_document(file_bytes, filename)
        if not ok or not lines:
            return [], {"parsed_from":"failed", "raw_lines": len(lines)}

        fallback = FallbackStatementParser()
        txs, meta = fallback.parse_statement(lines)
        meta["raw_lines"] = len(lines)
        return txs, meta

# ----------------------------
# Deterministic fallback parser (robust)
# ----------------------------
class FallbackStatementParser:
    # broad date patterns
    DATE_RE = re.compile(
    r'(\b\d{1,2}[/\-. ]\d{1,2}[/\-. ]\d{2,4}\b|\b\d{1,2}[/\-. ]\d{1,2}\b|'
    r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[ .-]?\d{1,2}[, ]*\d{2,4}?\b)',
    re.I
)

    # amount-like tokens: allow spaces/comma thousands, parentheses, leading +/-.
    AMOUNT_RE = re.compile(r'([+\-]?\(?\s*\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?\)?)')

    deposit_keys = ["deposit", "received", "credit", "salary", "refund", "payroll", "payment received"]
    withdraw_keys = ["withdraw", "purchase", "sent", "payment", "debit", "card", "pos", "paid to"]
    def _normalize_date(self, d: str) -> str:
        d = d.strip()
        # Try DD/MM or MM/DD without year
        m = re.match(r'(\d{1,2})[/\-. ](\d{1,2})$', d)
        if m:
            day, month = m.group(1), m.group(2)
            # Assume current year
            year = datetime.now().year
            try:
                obj = datetime(int(year), int(day), int(month))
                return obj.strftime("%Y-%m-%d")
            except:
                try:
                    obj = datetime(int(year), int(month), int(day))
                    return obj.strftime("%Y-%m-%d")
                except:
                    return d

        # Full date formats
        try:
            for fmt in ("%d/%m/%Y","%m/%d/%Y","%d-%m-%Y","%m-%d-%Y",
                        "%d/%m/%y","%m/%d/%y","%d-%m-%y","%m-%d-%y"):
                try:
                    return datetime.strptime(d, fmt).strftime("%Y-%m-%d")
                except:
                    pass
        except:
            return d

        return d

    def _infer_vendor(self, line: str, date_str: str, amount_token: str) -> str:
        tmp = line
        if date_str:
            tmp = tmp.replace(date_str, ' ')
        if amount_token:
            tmp = tmp.replace(amount_token, ' ')
        tmp = re.sub(r'\b(PKR|USD|EUR|GBP|AED|CAD|AUD|Rs|Balance|Available)\b', ' ', tmp, flags=re.I)
        tmp = re.sub(r'[\d\|\-:\/\.]', ' ', tmp)
        tmp = re.sub(r'[\(\)\[\],]', ' ', tmp)
        tmp = re.sub(r'\s{2,}', ' ', tmp).strip()
        return tmp if tmp else "UNKNOWN"

    def parse_statement(self, lines: List[str]) -> Tuple[List[Transaction], Dict[str,Any]]:
        txs: List[Transaction] = []

        for ln in lines:
            low = ln.lower().strip()
            # skip trivial lines
            if not ln.strip() or any(k in low for k in ("available balance", "closing balance", "total", "statement")):
                continue

            d_match = self.DATE_RE.search(ln)
            date_str = self._normalize_date(d_match.group(1)) if d_match else ""


            # find all amount-like tokens
            matches = list(self.AMOUNT_RE.finditer(ln))
            if not matches:
                continue

            # build tokens with spans
            tokens = [(m.group(1), m.start(1), m.end(1)) for m in matches]

            # heuristic choose:
            chosen_token = None
            chosen_span = (0, len(ln))

            # 1) prefer token with explicit sign or parentheses
            for tok, s, e in tokens:
                if tok.strip().startswith("+") or tok.strip().startswith("-") or ("(" in tok and ")" in tok):
                    chosen_token = tok
                    chosen_span = (s, e)
                    break

            # 2) prefer token with nearby currency
            if chosen_token is None:
                for tok, s, e in tokens:
                    ctx_start = max(0, s-12)
                    context = ln[ctx_start:s].upper()
                    if any(cur in context for cur in ('PKR','USD','EUR','GBP','AED','CAD','AUD','RS')):
                        chosen_token = tok
                        chosen_span = (s, e)
                        break

            # 3) prefer token nearest to line end
            if chosen_token is None:
                best = None
                best_dist = None
                L = len(ln)
                for tok,s,e in tokens:
                    dist = L - e
                    if best is None or dist < best_dist:
                        best = (tok,s,e)
                        best_dist = dist
                if best:
                    chosen_token, chosen_span = best[0], (best[1], best[2])

            # 4) fallback: last token
            if chosen_token is None and tokens:
                chosen_token, chosen_span = tokens[-1][0], (tokens[-1][1], tokens[-1][2])

            if not chosen_token:
                continue

            # Try to parse chosen token
            amt_val = clean_amount_token(chosen_token)
            # If chosen token invalid, try other tokens (reverse)
            if amt_val is None:
                for tok, s, e in tokens[::-1]:
                    parsed = clean_amount_token(tok)
                    if parsed is not None:
                        amt_val = parsed
                        chosen_token = tok
                        chosen_span = (s, e)
                        break
            if amt_val is None:
                continue

            # Determine sign primarily from token itself
            tokstr = chosen_token.strip()
            if tokstr.startswith("-") or (tokstr.startswith("(") and tokstr.endswith(")")):
                amt_val = -abs(amt_val)
            elif tokstr.startswith("+"):
                amt_val = abs(amt_val)

            # If token had no explicit sign, use keywords as fallback
            if not (tokstr.startswith(("+","-")) or (tokstr.startswith("(") and tokstr.endswith(")"))):
                if any(k in low for k in self.withdraw_keys):
                    amt_val = -abs(amt_val)
                if any(k in low for k in self.deposit_keys):
                    amt_val = abs(amt_val)

            # Sanity: skip ridiculously large numbers (IDs)
            if abs(amt_val) > 1e12:
                continue

            vendor = self._infer_vendor(ln, date_str, chosen_token).title()
            if not vendor:
                vendor = "UNKNOWN"

            txs.append(Transaction(
                date=date_str,
                transaction_type="deposit" if amt_val > 0 else "withdrawal",
                vendor=vendor,
                amount=amt_val,
                description=ln,
                raw_line=ln
            ))

        meta = {"parsed_from":"universal_fallback", "transactions_extracted": len(txs)}
        return txs, meta

# ----------------------------
# (Optional) LLM enhancer — unchanged logic but uses clean amounts
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
                messages=[{"role":"user","content":prompt}],
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
            return enhanced + transactions[200:]
        except Exception as e:
            logger.exception("LLM enhancement error: %s", e)
            return transactions

# ----------------------------
# Categorizer & dedupe
# ----------------------------
class TransactionCategorizer:
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
                else:
                    t.category = "Expense"
        return txs

    def detect_duplicates(self, txs: List[Transaction]) -> List[Transaction]:
        seen = {}
        order = []
        for t in txs:
            key = (t.date, round(t.amount,2), re.sub(r'\W+','', (t.vendor or '').lower()))
            if key in seen:
                existing = seen[key]
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
    def generate_summary_statistics(self, transactions: List[Transaction]) -> Dict[str,Any]:
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
        grp = df.groupby('vendor').agg({'amount':'sum', 'raw_line':'count'}).reset_index()
        grp.columns = ['Source/Vendor', 'Subtotal ($)', 'Transaction Count']
        grp['Subtotal ($)'] = grp['Subtotal ($)'].astype(float)
        total = grp['Subtotal ($)'].sum()
        total_row = pd.DataFrame([{'Source/Vendor':'TOTAL DEPOSITS','Subtotal ($)': total, 'Transaction Count': grp['Transaction Count'].sum()}])
        out = pd.concat([grp, total_row], ignore_index=True)
        return out[['Source/Vendor','Transaction Count','Subtotal ($)']]

    def generate_withdrawals_summary(self, transactions: List[Transaction]) -> pd.DataFrame:
        wds = [t for t in transactions if t.amount < 0]
        if not wds:
            return pd.DataFrame()
        df = pd.DataFrame([asdict(t) for t in wds])
        df['amount'] = df['amount'].abs()
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
        grp = df.groupby('vendor').agg({'amount':'sum', 'raw_line':'count'}).reset_index()
        grp.columns = ['Vendor','Subtotal ($)','Transaction Count']
        total = grp['Subtotal ($)'].sum()
        total_row = pd.DataFrame([{'Vendor':'TOTAL WITHDRAWALS','Subtotal ($)': total, 'Transaction Count': grp['Transaction Count'].sum()}])
        out = pd.concat([grp, total_row], ignore_index=True)
        return out[['Vendor','Transaction Count','Subtotal ($)']]

    def generate_pl_report(self, transactions: List[Transaction]) -> pd.DataFrame:
        s = self.generate_summary_statistics(transactions)
        total_income = s['Total Deposit Amount']
        total_expenses = s['Total Withdrawal Amount']
        net = s['Net Income']
        return pd.DataFrame([
            {'Category':'Total Income','Amount ($)': total_income},
            {'Category':'Total Expenses','Amount ($)': -total_expenses},
            {'Category':'NET INCOME','Amount ($)': net}
        ])

# ----------------------------
# Streamlit UI (single-file)
# ----------------------------
st.set_page_config(page_title="Bank Statement Analyzer (Hybrid)", layout="wide")
st.title("💼 Bank Statement Analyzer — Deterministic + LLM (Hybrid)")

st.markdown(
    "Upload a bank statement (PDF / CSV / DOCX). The app extracts deterministically, "
    "then optionally cleans rows with an LLM (gpt-4o-mini). Totals are computed from parsed numeric amounts."
)

with st.sidebar:
    st.header("Settings")
    use_llm = st.checkbox("Enable LLM enhancement (cost)", value=True)
    llm_model = st.selectbox("LLM model", ["gpt-4o-mini"], index=0)
    st.write("Put your OpenAI key in `.streamlit/secrets.toml` as: `OPENAI_API_KEY = \"sk-...\"`")

uploaded = st.file_uploader("Upload statement (PDF, CSV, DOCX)", type=["pdf","csv","doc","docx"])

if uploaded:
    st.info(f"File: {uploaded.name} — {uploaded.size/1024:.1f} KB")
    currency = st.selectbox("Currency", ["PKR","USD","EUR","GBP","AED","CAD","AUD"], index=0)

    if st.button("Process Statement"):
        with st.spinner("Parsing & processing..."):
            file_bytes = uploaded.read()
            dp = DocumentParser()

            # Try table extraction first
            txs_table = dp.parse_pdf_table(file_bytes) if uploaded.name.lower().endswith(".pdf") else []
            if txs_table:
                transactions = txs_table
                parsed_from = "pdf-table"
            else:
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

            st.session_state.transactions = transactions
            st.session_state.stats = stats
            st.session_state.deposit_df = deposits_df
            st.session_state.withdrawal_df = withdrawals_df
            st.session_state.pl_df = pl_df
            st.session_state.currency = currency
            st.session_state.parsed_from = parsed_from

            st.success(f"Processed {len(transactions)} transactions ({parsed_from}).")

# Dashboard display
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

    # reconcile computed sums
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
                        "Date": it.date,
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