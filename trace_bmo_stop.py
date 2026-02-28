import sys
import os
from bank_statement_parser import BankStatementParser, BankName, BANK_LAYOUTS, parse_bank_statement
import logging

logging.basicConfig(level=logging.INFO)

def trace_bmo():
    txt_path = r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App\ocr_dumps\BMO.txt"
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    lines = text.split("\n")
    layout = BANK_LAYOUTS[BankName.BMO]
    
    print("=== Tracing BMO Summary Markers ===")
    for i, line in enumerate(lines):
        line_upper = line.upper()
        for marker in layout.summary_markers:
            if marker in line_upper:
                print(f"Match found at line {i}: '{line}' (Marker: {marker})")
                
    stmt = parse_bank_statement(text, BankName.BMO)
    print(f"\nTotal Transactions: {len(stmt.transactions)}")

if __name__ == "__main__":
    trace_bmo()
