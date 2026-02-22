import re
from pathlib import Path

BASE_DIR = r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App"
content = (Path(BASE_DIR) / "bank_data_analysis.py").read_text(encoding='utf-8')

# Get regexes from UniversalParser class
date_re_str = re.search(r'class UniversalParser:.*?DATE_RE = re.compile\(\s*r\'(.*?)\'', content, re.S).group(1)
amount_re_str = re.search(r'class UniversalParser:.*?AMOUNT_RE = re.compile\(\s*r\'(.*?)\'', content, re.S).group(1)

date_re = re.compile(date_re_str)
amount_re = re.compile(amount_re_str)

print(f"DATE_RE: {date_re_str}")
print(f"AMOUNT_RE: {amount_re_str}")

# Load Page 1 specifically
lines = (Path(BASE_DIR) / "amex_raw_full.txt").read_text(encoding='utf-16le').splitlines()

print("\n--- Page 1 Regex Analysis ---")
for i, ln in enumerate(lines[:100]):
    d = date_re.search(ln)
    m = amount_re.findall(ln)
    if d or m:
        print(f"L{i+1:3} | DateMatch: {bool(d)} | Amts: {m} | Text: {ln[:80]}")

print("\n--- Potential leakage points ---")
# Check if 11/19/25 matches AMOUNT_RE
test_str = "11/19/25"
print(f"Testing '{test_str}' against AMOUNT_RE: {amount_re.findall(test_str)}")
