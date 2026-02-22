import re
import os
import pytesseract
from pdf2image import convert_from_bytes
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Any
from datetime import datetime

# CONFIG
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"

# THE TIGHTENED REGEX (THE FIX)
DATE_AT_START = re.compile(
    r'^\s*('
    r'(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12][0-9]|3[01])(?:/\d{2,4})?|'
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(?:0?[1-9]|[12][0-9]|3[01])(?:,?\s*\d{2,4})?'
    r')\b', re.I
)

trash_markers = ["minimum payment due", "pay in full portion", "pay over time portion", "account total", "days in billing period"]

p = Path("bank statments/AMEX.pdf")
print(f"Running OCR on {p}...")
images = convert_from_bytes(p.read_bytes(), poppler_path=POPPLER_PATH)
text = "\n".join(pytesseract.image_to_string(img) for img in images)

lines = [l.strip() for l in text.splitlines() if l.strip()]
blocks = []
filtered_count = 0

for ln in lines:
    low = ln.lower()
    if any(k in low for k in trash_markers):
        filtered_count += 1
        continue
    
    if DATE_AT_START.match(ln):
        blocks.append(ln)

print(f"Total lines extracted: {len(lines)}")
print(f"Summary lines filtered: {filtered_count}")
print(f"Total blocks (lines starting with valid date): {len(blocks)}")
print("\nSample matches:")
for b in blocks[:20]:
    print(f"  {b}")

# Specifically check for any remaining trash like "47-779"
trash_dates = [b for b in blocks if "-" in b and not re.search(r'\b(?:20|19)\d{2}\b', b)]
print(f"\nTrash dates found: {len(trash_dates)}")
for d in trash_dates:
    print(f"  TRASH: {d}")
