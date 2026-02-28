import sys
from document_parser import DocumentParser

PDF = "bank statments/20 Dec Chase.pdf"
dp = DocumentParser()
with open(PDF, "rb") as f:
    text_lines, _, _ = dp.parse_document(f.read(), "20 Dec Chase.pdf")

print("=== Checks and ATM lines ===")
for i, line in enumerate(text_lines):
    if "Check" in line or "ATM" in line or "268" in line or "500.00" in line or "800" in line:
        print(f"{i}: {line}")
