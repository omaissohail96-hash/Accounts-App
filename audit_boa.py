import os
from datetime import datetime
from bank_statement_parser import parse_bank_statement, BankName
from document_parser import DocumentParser

def audit_boa():
    pdf_path = "bank statments/BoA.pdf"
    doc_parser = DocumentParser()
    with open(pdf_path, 'rb') as f:
        file_bytes = f.read()
    lines, _, _ = doc_parser.parse_document(file_bytes, pdf_path)
    text = '\n'.join(lines)
    statement = parse_bank_statement(text, BankName.BOA)
    
    deposits = [t for t in statement.transactions if t.amount > 0]
    print(f"Total Deposits: {len(deposits)}")
    
    repaired_days = ["2025-12-04", "2025-12-08", "2025-12-16", "2025-12-29"]
    for d_str in repaired_days:
        print(f"\nTRANSACTIONS FOR {d_str}:")
        d = datetime.strptime(d_str, "%Y-%m-%d").date()
        for i, t in enumerate(statement.transactions, 1):
            if t.date.date() == d:
                print(f"  {i:3d}: {t.amount:10.2f} | {t.description[:40]} | RAW: {t.raw_line}")

if __name__ == "__main__":
    audit_boa()
