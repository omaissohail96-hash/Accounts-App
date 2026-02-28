"""
End-to-end test: use DocumentParser (production path) to parse AMEX.pdf
and then run parse_with_multi_bank_parser on it.
"""
from document_parser import DocumentParser
from bank_statement_parser import detect_bank, parse_bank_statement

PDF = "bank statments/AMEX.pdf"

print("=== DocumentParser (production path) ===")
dp = DocumentParser()
with open(PDF, "rb") as f:
    file_bytes = f.read()

lines, ok, unreadable = dp.parse_document(file_bytes, "AMEX.pdf")
print(f"  ok={ok}, lines={len(lines)}, unreadable pages={unreadable}")
if lines:
    print(f"  First line: {lines[0][:80]}")

print("\n=== detect_bank on extracted text ===")
text = "\n".join(lines)
bank = detect_bank(text)
print(f"  -> {bank} ({bank.value})")

print("\n=== parse_bank_statement ===")
result = parse_bank_statement(text)
print(f"  bank_name  : {result.bank_name}")
print(f"  transactions: {len(result.transactions)}")
print(f"  errors      : {result.errors}")
for i, t in enumerate(result.transactions[:5], 1):
    print(f"  {i}: {t.date} | {t.amount:10.2f} | {t.description[:50]}")
