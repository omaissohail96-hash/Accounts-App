import re
from pathlib import Path

# Load classes from file
BASE_DIR = r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App"
FILE_PATH = Path(BASE_DIR) / "bank_data_analysis.py"

# Simple regex extraction from file to ensure we use the REAL ones
content = FILE_PATH.read_text(encoding='utf-8')

# Extract UniversalParser.DATE_RE and AMOUNT_RE
date_re_str = re.search(r'DATE_RE = re.compile\(\s*r\'(.*?)\'', content, re.S).group(1)
amount_re_str = re.search(r'AMOUNT_RE = re.compile\(\s*r\'(.*?)\'', content, re.S).group(1)
date_re = re.compile(date_re_str)
amount_re = re.compile(amount_re_str)

print(f"Using DATE_RE: {date_re_str}")
print(f"Using AMOUNT_RE: {amount_re_str}")

# Load AMEX lines
lines = [ln.strip() for ln in (Path(BASE_DIR) / "amex_raw_full.txt").read_text(encoding='utf-16le').splitlines() if ln.strip()]

def diag():
    found = 0
    for i, ln in enumerate(lines):
        d = date_re.search(ln)
        if not d: continue
        
        m = amount_re.findall(ln)
        if not m: continue
        
        found += 1
        print(f"Line {i+1:4} | Matches: {m} | Text: {ln}")

    print(f"\nTotal match-lines found: {found}")

diag()
