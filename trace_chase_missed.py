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

print("\n=== Lines with MM/DD but no regex match ===")
matched_count = 0
missed_lines = []

for line in text_lines:
    # Check if it looks like a transaction line (starts with a date)
    if re.search(r'\d{1,2}/\d{1,2}', line):
        match = re.search(regex, line)
        if match:
            matched_count += 1
        else:
            missed_lines.append(line)

print(f"Matched: {matched_count}")
print(f"Missed: {len(missed_lines)}")
for i, l in enumerate(missed_lines[:30]):
    print(f"  {i}: {l}")

