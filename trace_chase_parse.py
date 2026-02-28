from document_parser import DocumentParser
from bank_statement_parser import detect_bank, parse_bank_statement, BankName

PDF = "bank statments/20 Dec Chase.pdf"

print("=== Reading PDF with DocumentParser ===")
dp = DocumentParser()
with open(PDF, "rb") as f:
    text_lines, ok, unreadable = dp.parse_document(f.read(), "20 Dec Chase.pdf")
text = "\n".join(text_lines)

print(f"Total chars: {len(text)}")
print("\n=== First 1000 chars of OCR text ===")
print(text[:1000])

print("\n=== detect_bank result ===")
bank = detect_bank(text)
print(f"  -> {bank}  ({bank.value})")

print("\n=== parse_bank_statement ===")
result = parse_bank_statement(text)
print(f"  bank_name  : {result.bank_name}")
print(f"  errors      : {result.errors}")
print(f"\n  TOTAL transactions extracted: {len(result.transactions)}")

print("\nFirst 20 transactions detail:")
for i, t in enumerate(result.transactions[:20], 1):
    print(f"  {i:2d}: Parsed={t.date} | Amt={t.amount:10.2f} | Desc={t.description[:60]}")
