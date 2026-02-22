import pdfplumber
from pathlib import Path

p = Path("bank statments/AMEX.pdf")
print(f"File exists: {p.exists()}")
try:
    with pdfplumber.open(p) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        print(f"Extracted Length: {len(text)}")
        print("First 1000 characters:")
        print(text[:1000])
except Exception as e:
    print(f"Error: {e}")
