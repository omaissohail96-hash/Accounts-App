import sys
import os
from dataclasses import dataclass
from typing import Optional, List

# 1. Define the Transaction dataclass exactly as in bank_data_analysis.py
@dataclass
class Transaction:
    date: str
    transaction_type: str    # 'deposit' or 'withdrawal'
    vendor: str
    amount: float            # signed: deposits > 0, withdrawals < 0
    description: str
    raw_line: str
    section: Optional[str] = None
    category: Optional[str] = None
    needs_review: bool = False
    source: Optional[str] = "BANK"

# 2. Import components from document_parser and bank_statement_parser
from document_parser import DocumentParser
from bank_statement_parser import parse_bank_statement, TransactionType

def verify_amex_pipeline():
    PDF_PATH = "bank statments/AMEX.pdf"
    if not os.path.exists(PDF_PATH):
        print(f"Error: {PDF_PATH} not found.")
        return

    print(f"Verifying {PDF_PATH}...")
    
    # 1. Document Extraction (with OCR)
    dp = DocumentParser()
    with open(PDF_PATH, "rb") as f:
        file_bytes = f.read()
    
    lines, ok, unreadable = dp.parse_document(file_bytes, "AMEX.pdf")
    print(f"DocumentParser: ok={ok}, lines={len(lines)}, unreadable={unreadable}")
    
    if not ok or not lines:
        print("Error: Failed to extract text.")
        return

    # 2. Parse using AmexParser (multi-bank parser core)
    text = "\n".join(lines)
    parsed_result = parse_bank_statement(text)
    
    if not parsed_result or not parsed_result.transactions:
        print("Error: AmexParser returned no transactions.")
        return

    print(f"AmexParser extracted {len(parsed_result.transactions)} transactions.")
    
    # 3. Simulate the conversion logic from bank_data_analysis.py
    old_transactions = []
    for new_tx in parsed_result.transactions:
        # This is the logic I fixed in bank_data_analysis.py:
        amount = new_tx.amount
        if amount >= 0:
            tx_type = "deposit"
            amount = abs(amount)
        else:
            tx_type = "withdrawal"
            # amount stays negative
        
        # Format date
        if hasattr(new_tx.date, 'strftime'):
            date_str = new_tx.date.strftime("%m/%d/%Y")
        else:
            date_str = str(new_tx.date)
        
        # Vendor extraction
        desc = new_tx.description or ""
        vendor = desc.split('-')[0].strip() if desc else ""
        
        # Create Transaction
        old_tx = Transaction(
            date=date_str,
            transaction_type=tx_type,
            vendor=vendor,
            amount=amount,
            description=desc,
            raw_line=f"{date_str} {desc} {new_tx.amount}",
            source="BANK"
        )
        old_transactions.append(old_tx)

    print(f"Conversion: Successfully converted to {len(old_transactions)} internal Transaction objects.")
    
    # Verify the "Br Q35" merging issue is gone
    # Specifically, check if we have multiple transactions or just 1 giant description
    br_q35_txs = [t for t in old_transactions if "Br Q35" in t.description]
    print(f"Found {len(br_q35_txs)} transactions containing 'Br Q35'.")
    
    if len(br_q35_txs) > 1:
        print("SUCCESS: 'Br Q35' transactions are correctly split (not merged into one).")
    elif len(br_q35_txs) == 1:
        if len(br_q35_txs[0].description) > 200:
            print("WARNING: Still seeing a giant merged description for 'Br Q35'.")
            print(f"Description: {br_q35_txs[0].description[:150]}...")
        else:
            print("Note: Only 1 small 'Br Q35' transaction found.")
    
    # Final checks
    if len(old_transactions) == 38:
        print("\n=== VERIFICATION PASSED: 38 TRANSACTIONS CORRECTLY PARSED AND CONVERTED ===")
    else:
        print(f"\n=== VERIFICATION FAILED: Found {len(old_transactions)} transactions instead of 38 ===")

if __name__ == "__main__":
    verify_amex_pipeline()
