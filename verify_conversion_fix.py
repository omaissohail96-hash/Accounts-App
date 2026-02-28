"""
End-to-end production test: DocumentParser -> parse_with_multi_bank_parser
"""
from document_parser import DocumentParser
from bank_statement_parser import parse_bank_statement, detect_bank

PDF = "bank statments/AMEX.pdf"

# Step 1: DocumentParser (OCR)
dp = DocumentParser()
with open(PDF, "rb") as f:
    file_bytes = f.read()

lines, ok, unreadable = dp.parse_document(file_bytes, "AMEX.pdf")
print(f"DocumentParser: ok={ok}, lines={len(lines)}, unreadable={unreadable}")

# Step 2: Bank detection
text = "\n".join(lines)
bank = detect_bank(text)
print(f"detect_bank: {bank}")

# Step 3: parse_bank_statement (this is what multi-bank parser calls)
result = parse_bank_statement(text)
print(f"parse_bank_statement: {len(result.transactions)} transactions, errors={result.errors}")

# Step 4: Simulate the broken conversion (old code)
print("\n--- Checking TransactionType values ---")
from bank_statement_parser import TransactionType
for attr in ['DEPOSIT', 'WITHDRAWAL', 'CREDIT', 'DEBIT', 'FEE', 'CHECK']:
    try:
        val = getattr(TransactionType, attr)
        print(f"  TransactionType.{attr} = {val}")
    except AttributeError:
        print(f"  TransactionType.{attr} = DOES NOT EXIST!")

# Step 5: Check types of the actual transactions
for tx in result.transactions[:3]:
    print(f"\n  tx.type = {tx.type!r} (class: {type(tx.type).__name__})")
    print(f"  tx.amount = {tx.amount}")
