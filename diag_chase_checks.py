import sys
import os
from document_parser import DocumentParser
from bank_statement_parser import parse_bank_statement, BankName, TransactionType

def diag_chase():
    pdf_path = r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App\bank statments\20 Dec Chase.pdf"
    if not os.path.exists(pdf_path):
        print(f"File {pdf_path} not found")
        return
        
    dp = DocumentParser()
    with open(pdf_path, "rb") as f:
        content = f.read()
        lines, ok, unreadable = dp.parse_document(content, "20 Dec Chase.pdf")
        
    if not ok:
        print("Failed to parse document")
        return
        
    text = "\n".join(lines)
    stmt = parse_bank_statement(text, BankName.CHASE)
    
    print(f"Total Transactions: {len(stmt.transactions)}")
    
    checks = [t for t in stmt.transactions if "CHECK" in t.description.upper() or getattr(t, 'category', '') == "CHECK"]
    print(f"Detected Checks: {len(checks)}")
    for tx in checks:
        print(f"  {tx.date.strftime('%m/%d')} | {tx.amount:10.2f} | {tx.description}")
        
    print("\n--- Raw Lines around 'CHECKS PAID' ---")
    found_section = False
    for i, line in enumerate(lines):
        if "CHECKS PAID" in line.upper():
            found_section = True
            for j in range(max(0, i-5), min(len(lines), i+20)):
                print(f"{j:4}: {lines[j]}")
            break
    if not found_section:
        print("'CHECKS PAID' section not found in raw lines")

if __name__ == "__main__":
    diag_chase()
