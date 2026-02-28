import sys
import os
from unittest.mock import MagicMock

# Mock Streamlit before importing any modules that use it
mock_st = MagicMock()
mock_st.session_state = MagicMock()
mock_st.session_state.user_id = 'test_user'
mock_st.tabs.return_value = [MagicMock(), MagicMock()]
mock_st.columns.return_value = [MagicMock(), MagicMock()]
sys.modules['streamlit'] = mock_st

# Import after mocking
from document_parser import DocumentParser
from bank_data_analysis import parse_with_multi_bank_parser

def test_amex_parsing():
    PDF_PATH = "bank statments/AMEX.pdf"
    if not os.path.exists(PDF_PATH):
        print(f"Error: {PDF_PATH} not found.")
        return

    print(f"Testing {PDF_PATH}...")
    
    # 1. Document Extraction
    dp = DocumentParser()
    with open(PDF_PATH, "rb") as f:
        file_bytes = f.read()
    
    lines, ok, unreadable = dp.parse_document(file_bytes, "AMEX.pdf")
    print(f"DocumentParser: ok={ok}, lines={len(lines)}, unreadable={unreadable}")
    
    if not ok or not lines:
        print("Failed to extract text.")
        return

    # 2. Multi-Bank Parsing & Conversion
    transactions, meta = parse_with_multi_bank_parser(
        lines, 
        "AMEX.pdf", 
        include_opening_balance=False, 
        extract_check_memos=True
    )
    
    if transactions is None:
        print("Multi-bank parser failed (returned None). Check for exceptions in console.")
        return

    print(f"Successfully parsed {len(transactions)} transactions.")
    print(f"Bank detected: {meta.get('bank')}")
    
    # Check for the Br Q35 issue
    found_br_q35 = False
    for t in transactions:
        if "Br Q35" in t.description:
            found_br_q35 = True
            print(f"DEBUG: Found Br Q35 transaction: {t.date} | {t.amount} | {t.description[:100]}")
    
    if not found_br_q35:
        print("Note: 'Br Q35' text not found in transactions (this is good if they were correctly split).")
    
    # Print first few transactions for inspection
    print("\nFirst 5 transactions:")
    for i, t in enumerate(transactions[:5], 1):
        print(f"  {i}: {t.date} | {t.amount:10.2f} | {t.description[:50]}... | Source: {t.source}")

    if len(transactions) == 38:
        print("\nSUCCESS: Found exactly 38 transactions as expected.")
    else:
        print(f"\nWARNING: Found {len(transactions)} transactions. Expected 38.")

if __name__ == "__main__":
    test_amex_parsing()
