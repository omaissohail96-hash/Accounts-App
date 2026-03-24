import os
import sys
from document_parser import DocumentParser
from bank_statement_parser import parse_bank_statement, BankName

def reproduce_bmo():
    # Use relative path or local absolute path
    base_dir = r"d:\Program Files\New Folder (4)\Accounts-App"
    pdf_path = os.path.join(base_dir, "bank statments", "BMO.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    print(f"--- Reproduction for {pdf_path} ---")
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
    
    deposits = [tx for tx in parsed_result.transactions if tx.amount > 0]
    withdrawals = [tx for tx in parsed_result.transactions if tx.amount < 0]
    
    print(f"Number of Deposits: {len(deposits)}")
    print(f"Number of Withdrawals: {len(withdrawals)}")
    print(f"Opening Balance: {parsed_result.beginning_balance}")
    print(f"Ending Balance: {parsed_result.ending_balance}")
    
    # Calculate expected ending balance
    calc_ending = (parsed_result.beginning_balance or 0) + sum(tx.amount for tx in parsed_result.transactions)
    print(f"Calculated Ending Balance: {calc_ending:.2f}")
    if parsed_result.ending_balance:
        print(f"Difference: {parsed_result.ending_balance - calc_ending:.2f}")
    
    print("\n--- All Transactions ---")
    for i, tx in enumerate(parsed_result.transactions):
        print(f"{i+1:2}: {tx.date.strftime('%Y-%m-%d') if tx.date else 'N/A'} | {tx.amount:10.2f} | {tx.description[:40]}")

if __name__ == "__main__":
    reproduce_bmo()
