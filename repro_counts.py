import os
from bank_statement_parser import parse_bank_statement, BankName
from document_parser import DocumentParser

def audit_counts(pdf_path, bank_name):
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    print(f"\n--- AUDITING {pdf_path} ({bank_name}) ---")
    doc_parser = DocumentParser()
    with open(pdf_path, 'rb') as f:
        file_bytes = f.read()
    
    lines, _, _ = doc_parser.parse_document(file_bytes, pdf_path)
    text = '\n'.join(lines)
    
    statement = parse_bank_statement(text, bank_name)
    
    ob = statement.beginning_balance or 0
    eb = statement.ending_balance or 0
    
    deposits = [t for t in statement.transactions if t.amount > 0]
    withdrawals = [t for t in statement.transactions if t.amount < 0]
    
    print(f"Opening Balance: {ob:,.2f}")
    print(f"Ending Balance: {eb:,.2f}")
    print(f"Detected Deposits count: {len(deposits)}")
    print(f"Detected Withdrawals count: {len(withdrawals)}")
    
    if bank_name == BankName.BOA:
        print("\nALL BOA DEPOSITS:")
        for i, t in enumerate(deposits, 1):
            print(f"  {i:3d}: {t.date.date()} | {t.amount:10.2f} | {t.description[:50]} | RAW: {t.raw_line}")
    
    dep_sum = sum(t.amount for t in deposits)
    wd_sum = sum(abs(t.amount) for t in withdrawals)
    
    calc_eb = ob + dep_sum - wd_sum
    print(f"Calculated EB: {calc_eb:,.2f}")
    print(f"Difference: {abs(calc_eb - eb):.4f}")

    print("\nSearch for OB in transactions:")
    for i, t in enumerate(statement.transactions, 1):
        if abs(abs(t.amount) - abs(ob)) < 0.01:
             print(f"  MATCH: Line {i} matches OB amount: {t.amount} ({t.description})")

    print("\nSearch for INTEREST in transactions:")
    for i, t in enumerate(statement.transactions, 1):
        if "INTEREST" in t.description.upper():
             print(f"  INTEREST: Line {i}: {t.amount} ({t.description})")

if __name__ == "__main__":
    audit_counts("bank statments/BoA.pdf", BankName.BOA)
    audit_counts("bank statments/BMO.pdf", BankName.BMO)
