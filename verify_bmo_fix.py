import os
import sys
from bank_statement_parser import parse_bank_statement, BankName
from document_parser import DocumentParser

def verify_bmo_fix():
    pdf_path = r"C:\Users\Farooq\Desktop\New Folder (4)\Accounts-App\bank statments\BMO.pdf"
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    print("--- Verifying BMO Fixes ---")
    dp = DocumentParser()
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()
        lines, ok, unreadable = dp.parse_document(file_bytes, "BMO.pdf")

    text = "\n".join(lines)
    parsed_result = parse_bank_statement(text, BankName.BMO, manual_year=2025)

    print(f"Bank: {parsed_result.bank_name}")
    print(f"Beginning Balance: ${parsed_result.beginning_balance:,.2f} (Expected: $8,248.15)")
    print(f"Ending Balance: ${parsed_result.ending_balance:,.2f} (Expected: $34,745.80)")
    
    print("\n--- Verifying Dates ---")
    first_tx = parsed_result.transactions[0]
    print(f"First TX Date: {first_tx.date} | Desc: {first_tx.description[:40]}")
    if first_tx.date.day == 1 and first_tx.date.month == 12 and first_tx.date.year == 2025:
        print("✅ Date correctly parsed as Dec 01, 2025")
    else:
        print(f"❌ Date misparsed: {first_tx.date}")

    print("\n--- Summary Check ---")
    total_withdrawals = sum(tx.amount for tx in parsed_result.transactions if tx.amount < 0)
    total_deposits = sum(tx.amount for tx in parsed_result.transactions if tx.amount > 0)
    print(f"Total Withdrawals: ${total_withdrawals:,.2f}")
    print(f"Total Deposits: ${total_deposits:,.2f}")
    
    # Check if we stopped before daily balance
    daily_bal_txs = [tx for tx in parsed_result.transactions if "DAILY BALANCE" in tx.description.upper()]
    if daily_bal_txs:
        print(f"❌ Found {len(daily_bal_txs)} Daily Balance rows misidentified as transactions")
    else:
        print("✅ No Daily Balance rows misidentified as transactions")

if __name__ == "__main__":
    verify_bmo_fix()
