from document_parser import DocumentParser
from bank_statement_parser import detect_bank, parse_bank_statement, BankName

PDF = "bank statments/BoA.pdf"

print("=== Reading PDF with DocumentParser ===")
dp = DocumentParser()
with open(PDF, "rb") as f:
    text_lines, ok, unreadable = dp.parse_document(f.read(), "BoA.pdf")
text = "\n".join(text_lines)

print(f"Total chars: {len(text)}")
print("\n=== First 2000 chars of OCR text ===")
print(text[:2000])

print("\n=== detect_bank result ===")
bank = detect_bank(text)
print(f"  -> {bank}  ({bank.value})")

print("\n=== parse_bank_statement ===")
result = parse_bank_statement(text)
print(f"  bank_name  : {result.bank_name}")
print(f"  TOTAL transactions extracted: {len(result.transactions)}")
print(f"  errors      : {result.errors}")

print("\nFirst 20 transactions:")
for i, t in enumerate(result.transactions[:20], 1):
    print(f"  {i:2d}: {t.date} | {t.amount:10.2f} | {t.type.value:7} | {t.category.value:10} | {t.description[:80]}")
