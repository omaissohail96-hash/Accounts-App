import sys
import os
from bank_statement_parser import parse_bank_statement, BankName
from document_parser import DocumentParser

def reproduce_boa():
    pdf_path = "bank statments/BoA.pdf"
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    print(f"--- Processing {pdf_path} ---")
    doc_parser = DocumentParser()
    with open(pdf_path, 'rb') as f:
        file_bytes = f.read()
    
    lines, _, _ = doc_parser.parse_document(file_bytes, pdf_path)
    text = '\n'.join(lines)
    
    statement = parse_bank_statement(text, BankName.BOA)
    
    print(f"Beginning Balance: {statement.beginning_balance}")

    for i, t in enumerate(statement.transactions, 1):
        desc_up = t.description.upper()
        if "BALANCE" in desc_up or "TOTAL" in desc_up or "SUMMARY" in desc_up:
            print(f"SUSPICIOUS TRANSACTION: {i}: {t.date.date()} | {t.amount} | {t.description} | RAW: {t.raw_line}")
    
    deposits = [t for t in statement.transactions if t.amount > 0]
    withdrawals = [t for t in statement.transactions if t.amount < 0]
    print(f"Number of Deposits: {len(deposits)}")
    print(f"Number of Withdrawals: {len(withdrawals)}")
    return

    # Re-calculate
    deposits = [t for t in statement.transactions if t.amount > 0]
    withdrawals = [t for t in statement.transactions if t.amount < 0]
    dep_sum = sum(t.amount for t in deposits)
    wd_sum = sum(abs(t.amount) for t in withdrawals)

    print(f"\n--- FINAL SUMMARY ---")
    print(f"Bank: {statement.bank_name}")
    print(f"Total Transactions: {len(statement.transactions)}")
    print(f"Number of Deposits: {len(deposits)} | Total Amount: {dep_sum:,.2f}")
    print(f"Number of Withdrawals: {len(withdrawals)} | Total Amount: {wd_sum:,.2f}")
    print(f"Opening Balance: {statement.beginning_balance}")
    print(f"Ending Balance: {statement.ending_balance}")
    
    calc_ending = (statement.beginning_balance or 0) + dep_sum - wd_sum
    print(f"Calculated Ending Balance: {calc_ending:,.2f}")
    if statement.ending_balance:
        print(f"Difference: {abs(calc_ending - statement.ending_balance):.2f}")

if __name__ == "__main__":
    reproduce_boa()
