from document_parser import DocumentParser
from bank_statement_parser import parse_bank_statement
import logging
logging.basicConfig(level=logging.DEBUG)

PDF = "bank statments/20 Dec Chase.pdf"
dp = DocumentParser()
with open(PDF, "rb") as f:
    text_lines, _, _ = dp.parse_document(f.read(), "20 Dec Chase.pdf")

res = parse_bank_statement('\n'.join(text_lines))

print(f"Total transactions parsed: {len(res.transactions)}")
for t in res.transactions:
    if "ATM" in t.raw_line.upper() or "CHECK" in t.raw_line.upper() or "268" in t.raw_line:
        print(f"FOUND: {t.raw_line}")
