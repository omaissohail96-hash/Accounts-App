"""
Trace what bank detect_bank() resolves AMEX.pdf to, then show what parse_bank_statement returns.
"""
from document_parser import DocumentParser
from bank_statement_parser import detect_bank, parse_bank_statement, BankName

PDF = "bank statments/AMEX.pdf"

print("=== Reading PDF with DocumentParser (OCR Support) ===")
dp = DocumentParser()
with open(PDF, "rb") as f:
    text_lines, ok, unreadable = dp.parse_document(f.read(), "AMEX.pdf")
text = "\n".join(text_lines)

print(f"Total chars: {len(text)}")
print(f"First 500 chars:\n{text[:500]}")

print("\n=== detect_bank result ===")
bank = detect_bank(text)
print(f"  -> {bank}  ({bank.value})")

print("\n=== parse_bank_statement ===")
result = parse_bank_statement(text)
print(f"  bank_name  : {result.bank_name}")
print(f"  transactions: {len(result.transactions)}")
print(f"  errors      : {result.errors}")
for i, t in enumerate(result.transactions[:5], 1):
    print(f"  {i}: {t.date} | {t.amount:10.2f} | {t.description[:50]}")
