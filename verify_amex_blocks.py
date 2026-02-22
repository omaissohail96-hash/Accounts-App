import re
import pdfplumber
from pathlib import Path

# THE TIGHTENED REGEX
DATE_AT_START = re.compile(
    r'^\s*('
    r'(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12][0-9]|3[01])(?:/\d{2,4})?|'
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(?:0?[1-9]|[12][0-9]|3[01])(?:,?\s*\d{2,4})?'
    r')\b', re.I
)

p = Path("bank statments/AMEX.pdf")
with pdfplumber.open(p) as pdf:
    text = "\n".join(page.extract_text() or "" for page in pdf.pages)

lines = [l.strip() for l in text.splitlines() if l.strip()]
blocks = []
for ln in lines:
    if DATE_AT_START.match(ln):
        blocks.append(ln)

print(f"Total blocks (lines starting with valid date): {len(blocks)}")
for b in blocks:
    print(f"  MATCH: {b}")

# Also check for trash markers
trash_markers = ["minimum payment due", "pay in full portion", "pay over time portion", "account total", "days in billing period"]
print("\nChecking for summary lines that should be filtered:")
for ln in lines:
    low = ln.lower()
    if any(k in low for k in trash_markers):
        print(f"  FILTER: {ln}")
