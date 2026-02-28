import os
import sys
from document_parser import DocumentParser
from bank_statement_parser import parse_bank_statement, BankName

def diag_bmo():
    pdf_path = r"C:\Users\Farooq\Desktop\New Folder (4)\Accounts-App\bank statments\BMO.pdf"
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    print(f"--- Diagnostic for {pdf_path} ---")
    dp = DocumentParser()
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()
        lines, ok, unreadable = dp.parse_document(file_bytes, "BMO.pdf")

    if not ok:
        print("DocumentParser failed")
        return

    text = "\n".join(lines)
    parsed_result = parse_bank_statement(text, BankName.BMO)
    
    print(f"Bank: {parsed_result.bank_name}")
    print(f"Total Transactions: {len(parsed_result.transactions)}")
    print(f"Opening Balance: {parsed_result.beginning_balance}")
    print(f"Closing Balance: {parsed_result.ending_balance}")
    
    print("\n--- First 10 Transactions (Detailed) ---")
    for i, tx in enumerate(parsed_result.transactions[:10]):
        print(f"TX {i+1}:")
        print(f"  Date: {tx.date} (Type: {type(tx.date)})")
        print(f"  Amount: {tx.amount}")
        print(f"  Desc: {tx.description}")
        print(f"  Category: {tx.category}")
        print(f"  Raw: {tx.raw_line}")

    # Search for suspicious amounts
    print("\n--- Suspiciously Large Transactions (> $10,000) ---")
    for tx in parsed_result.transactions:
        if abs(tx.amount) > 10000:
            print(f"  {tx.date} | {tx.amount} | {tx.description[:40]} | Category: {tx.category}")

if __name__ == "__main__":
    diag_bmo()
