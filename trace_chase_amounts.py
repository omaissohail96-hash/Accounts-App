import re
from document_parser import DocumentParser
from bank_statement_parser import BANK_LAYOUTS, BankName

PDF = "bank statments/20 Dec Chase.pdf"

print("=== Reading PDF with DocumentParser ===")
dp = DocumentParser()
with open(PDF, "rb") as f:
    text_lines, ok, unreadable = dp.parse_document(f.read(), "20 Dec Chase.pdf")

layout = BANK_LAYOUTS[BankName.CHASE]
regex = layout.transaction_regex

print("\n=== Unmatched lines with amounts ===")

for i, line in enumerate(text_lines):
    # If not matched by main regex
    if not re.search(regex, line):
        # But has a monetary amount pattern e.g. 1,000.00
        if re.search(r'\d+,\d{3}\.\d{2}|\d+\.\d{2}', line):
            print(f"{i:3d}: {line}")
