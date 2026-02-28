import os
import sys
from document_parser import DocumentParser
from bank_statement_parser import parse_bank_statement, BankName

def repro_bmo_pdf():
    pdf_path = r"C:\Users\Farooq\Desktop\New Folder (4)\Accounts-App\bank statments\BMO.pdf"
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    print(f"Processing {pdf_path}...")
    dp = DocumentParser()
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()
        lines, ok, unreadable = dp.parse_document(file_bytes, "BMO.pdf")

    if not ok:
        print("DocumentParser failed to extract text")
        return

    print(f"Extracted {len(lines)} lines")
    text = "\n".join(lines)
    
    # Save extracted text to a temp file for inspection
    with open("bmo_extracted_debug.txt", "w", encoding="utf-8") as out:
        out.write(text)
    print("Saved extracted text to bmo_extracted_debug.txt")

    parsed_result = parse_bank_statement(text, BankName.BMO)
    print(f"Detected Bank: {parsed_result.bank_name}")
    print(f"Total Transactions: {len(parsed_result.transactions)}")
    
    # Trace why it might stop
    withdrawals = [tx for tx in parsed_result.transactions if tx.amount < 0]
    print(f"Withdrawals count: {len(withdrawals)}")
    
    if len(parsed_result.transactions) < 50:
        print("\n--- Snippet of first 20 transactions ---")
        for tx in parsed_result.transactions[:20]:
            print(f"{tx.date.strftime('%m/%d')} | {tx.amount:10.2f} | {tx.description[:50]}")
            
    # Check for summary markers in the extracted text
    print("\n--- Checking for summary markers ---")
    markers = ["OPENING BALANCE", "CLOSING BALANCE", "MONTHLY TRANSACTION SUMMARY"]
    for i, line in enumerate(lines):
        for m in markers:
            if m in line.upper():
                print(f"Marker '{m}' found at line {i}: '{line}'")

if __name__ == "__main__":
    repro_bmo_pdf()
