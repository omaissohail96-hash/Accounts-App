import sys
import os
from pathlib import Path

# Add project root to path
BASE_DIR = r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App"
sys.path.append(BASE_DIR)

from bank_data_analysis import FallbackStatementParser, UniversalParser, DocumentParser

def diag():
    p_amex = Path(BASE_DIR) / "bank statments" / "AMEX.pdf"
    file_bytes = p_amex.read_bytes()
    
    dp = DocumentParser()
    lines, ok, unreadable = dp.parse_document(file_bytes, "AMEX.pdf")
    
    if not ok:
        print("Failed to read AMEX.pdf")
        return

    print(f"--- PRIMARY PARSER (FallbackStatementParser) ---")
    fb = FallbackStatementParser()
    txs, meta = fb.parse_statement(lines)
    print(f"Total Transactions: {len(txs)}")
    print(f"Meta: {meta}")
    for i, t in enumerate(txs):
        print(f"{i+1:3} | {t.date} | {t.amount:10.2f} | {t.vendor} | {t.section}")

    if len(txs) < 3:
        print(f"\n--- FALLBACK PARSER (UniversalParser) ---")
        up = UniversalParser()
        txs_up = up.parse(lines)
        print(f"Total Transactions: {len(txs_up)}")
        for i, t in enumerate(txs_up):
            print(f"{i+1:3} | {t.date} | {t.amount:10.2f} | {t.vendor}")

diag()
