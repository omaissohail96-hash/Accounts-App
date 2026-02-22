"""
Verification script for AMEX bank-mode parsing.
"""
import re
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
from pathlib import Path
import sys
from typing import Optional, Dict, Any, List, Tuple

# Mocking enough to run the functions
import types
bda = types.ModuleType('bda')

# Copy necessary code
src = open("bank_data_analysis.py", encoding="utf-8").read()

# Define environment for exec
env = {}
exec("import re\nfrom dataclasses import dataclass\nfrom typing import Optional, List, Dict, Any, Tuple\nfrom datetime import datetime", env)

# Extract specific parts to avoid streamlit errors
for line in src.splitlines():
    if line.startswith(("DATE_AT_START", "DATE_TOKEN_RE", "AMOUNT_RE", "CHECK_ROW_RE", "FEE_KEYWORDS_RE", "OPENING_BALANCE_RE")):
        exec(line, env)

# Helper functions
def get_func(name):
    start_marker = f"def {name}"
    part = src.split(start_marker)[1]
    # find next def or end of class
    end_indices = [part.find("\ndef "), part.find("\nclass ")]
    end_indices = [i for i in end_indices if i != -1]
    limit = min(end_indices) if end_indices else len(part)
    return f"def {name}" + part[:limit]

def get_class(name):
    start_marker = f"class {name}"
    part = src.split(start_marker)[1]
    # find next class
    end_idx = part.find("\nclass ")
    limit = end_idx if end_idx != -1 else len(part)
    return f"class {name}" + part[:limit]

exec(get_func("_normalize_date_token"), env)
exec(get_func("_md_key"), env) # needed for split check? no
exec(get_func("_clean_amount_token"), env)
exec(get_func("extract_true_amount"), env)
exec(get_func("_clean_description"), env)
exec(get_func("normalize_vendor"), env)
exec(get_func("clean_text_lines"), env)
exec(get_class("Transaction"), env)
exec(get_class("FallbackStatementParser"), env)

def get_text(p):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    POPPLER_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"
    try:
        with pdfplumber.open(p) as pdf:
            t = "\n".join(page.extract_text() or "" for page in pdf.pages)
            if t.strip(): return t
    except: pass
    img = convert_from_bytes(p.read_bytes(), poppler_path=POPPLER_PATH)
    return "\n".join(pytesseract.image_to_string(i) for i in img)

# RUN
print("--- AMEX BANK MODE VERIFICATION ---")
t = get_text(Path("bank statments/AMEX.pdf"))
parser = env["FallbackStatementParser"]()
lines = env["clean_text_lines"](t)
txs = parser.parse_statement(lines)

print(f"Total Transactions Found: {len(txs)}")
cats = {}
for tx in txs:
    cats[tx.category] = cats.get(tx.category, 0) + 1
    if "UNKNOWN" in tx.vendor and len(tx.date) != 10:
        print(f"MAYBE TRASH: {tx.date} | {tx.vendor} | {tx.amount} | {tx.raw_line[:50]}")
    elif "MATARI" in tx.vendor.upper():
        print(f"VALID TX: {tx.date} | {tx.vendor} | {tx.amount} | {tx.category}")

for c, count in cats.items():
    print(f"Category '{c}': {count}")

# 2. BMO CHECK
print("\n--- BMO CHECK ---")
t_bmo = get_text(Path("bank statments/BMO.pdf"))
lines_bmo = env["clean_text_lines"](t_bmo)
txs_bmo = parser.parse_statement(lines_bmo)
cats_bmo = {}
for tx in txs_bmo: cats_bmo[tx.category] = cats_bmo.get(tx.category, 0) + 1
for c, count in cats_bmo.items(): print(f"BMO Category '{c}': {count}")
