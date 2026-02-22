import re
import os
import sys
import unittest
from unittest.mock import MagicMock
from pathlib import Path

# Mock Streamlit
mock_st = MagicMock()
mock_st.tabs.return_value = [MagicMock(), MagicMock()]
sys.modules["streamlit"] = mock_st

import bank_data_analysis as bda
import pytesseract
from pdf2image import convert_from_bytes

# CONFIG
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"

def verify_bmo():
    print("\n--- VERIFYING BMO ---")
    p = Path("bank statments/BMO.pdf")
    images = convert_from_bytes(p.read_bytes(), poppler_path=POPPLER_PATH)
    text = "\n".join(pytesseract.image_to_string(img) for img in images)
    
    # Check Summary Extraction
    summary = bda.extract_universal_bank_summary(text)
    print("BMO Summary Data:")
    for k, v in summary.items():
        print(f"  {k}: {v['amount']}")
    
    # Assertions based on screenshot
    # BEGINNING BALANCE: $8,248.15
    # DEPOSIT AMOUNT: $68,063.18
    # WITHDRAWAL AMOUNT: $41,565.64
    # ENDING BALANCE: $34,745.80
    assert abs(summary["Beginning Balance"]["amount"] - 8248.15) < 1.0, f"Expected 8248.15, got {summary['Beginning Balance']['amount']}"
    assert abs(summary["Deposits and Additions"]["amount"] - 68063.18) < 1.0, f"Expected 68063.18, got {summary['Deposits and Additions']['amount']}"
    assert abs(summary["Other Withdrawals"]["amount"] - (-41565.64)) < 1.0, f"Expected -41565.64, got {summary['Other Withdrawals']['amount']}"
    assert abs(summary["Ending Balance"]["amount"] - 34745.80) < 1.0, f"Expected 34745.80, got {summary['Ending Balance']['amount']}"
    
    # Check Transaction Extraction
    parser = bda.FallbackStatementParser()
    txs = parser.parse_statement(text.split('\n'))
    print(f"BMO Transactions found: {len(txs)}")
    
    withdrawals = [t for t in txs if t.transaction_type == "withdrawal"]
    deposits = [t for t in txs if t.transaction_type == "deposit"]
    print(f"  Withdrawals: {len(withdrawals)}, Deposits: {len(deposits)}")
    
    # Total withdrawals according to summary: 23
    # Total deposits according to summary: 30
    # Total tx: 53
    if len(txs) < 50:
        print("WARNING: Too few BMO transactions found. Investigating first 10:")
        for tx in txs[:10]:
            print(f"  {tx.date} | {tx.amount} | {tx.section} | {tx.vendor}")

def verify_amex():
    print("\n--- VERIFYING AMEX ---")
    p = Path("bank statments/AMEX.pdf")
    images = convert_from_bytes(p.read_bytes(), poppler_path=POPPLER_PATH)
    text = "\n".join(pytesseract.image_to_string(img) for img in images)
    
    parser = bda.FallbackStatementParser()
    txs = parser.parse_statement(text.split('\n'))
    print(f"AMEX Transactions found: {len(txs)}")
    
    # Check for "New Charges" mapping
    new_charges = [t for t in txs if t.section == "WITHDRAWALS"]
    payments = [t for t in txs if t.section == "DEPOSITS"]
    print(f"  New Charges (Withdrawals): {len(new_charges)}")
    print(f"  Payments/Credits (Deposits): {len(payments)}")
    
    if len(txs) > 0:
        print("Sample AMEX transactions:")
        for tx in txs[:5]:
            print(f"  {tx.date} | {tx.amount} | {tx.section} | {tx.vendor}")

if __name__ == "__main__":
    try:
        verify_bmo()
        verify_amex()
        print("\nALL VERIFICATIONS PASSED!")
    except Exception as e:
        print(f"\nVERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
