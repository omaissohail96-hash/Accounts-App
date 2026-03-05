import sys
import os
import re
from unittest.mock import MagicMock

# Create a more precise Streamlit mock to handle the specific UI structure
class StMock(MagicMock):
    def tabs(self, labels):
        return [MagicMock() for _ in range(len(labels))]
    
    def columns(self, spec):
        if isinstance(spec, int):
            return [MagicMock() for _ in range(spec)]
        return [MagicMock() for _ in spec]
    
    @property
    def session_state(self):
        return {
            "user": {"id": "test"},
            "active_business": "test_biz",
            "all_transactions": [],
            "custom_rules": []
        }

class SessionStateMock(dict):
    def __getattr__(self, key):
        return self.get(key)
    def __setattr__(self, key, value):
        self[key] = value

# Inject mocks
st = StMock()
st.session_state = SessionStateMock({
    "user": {"id": "test"},
    "active_business": "test_biz",
    "all_transactions": [],
    "custom_rules": []
})
sys.modules["streamlit"] = st

# Import the actual parser
from bank_data_analysis import FallbackStatementParser, DocumentParser

def test_chase_real_parser():
    pdf_path = r"bank statments\bank statement 1.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        return

    print(f"--- Verifying with real FallbackStatementParser logic ---")
    
    with open(pdf_path, 'rb') as f:
        file_bytes = f.read()
    
    doc_parser = DocumentParser()
    lines, success, _ = doc_parser.parse_document(file_bytes, "bank statement 1.pdf")
    
    if not success:
        print("Failed to read PDF")
        return

    parser = FallbackStatementParser(include_opening_balance=False)
    transactions, meta = parser.parse_statement(lines)
    
    print(f"Total Transactions: {len(transactions)}")
    
    fees = [t for t in transactions if t.vendor == "Bank Fees"]
    print(f"Fees detected: {len(fees)}")
    for f in fees:
        print(f"  {f.date} | {f.amount} | {f.vendor} | {f.description[:50]}")
        
    other = [t for t in transactions if t.section == "OTHER_WITHDRAWALS"]
    print(f"Other Withdrawals: {len(other)}")
    
    if len(transactions) == 55:
        print("\nSUCCESS: Found exactly 55 transactions as expected!")
    else:
        print(f"\nRESULT: Found {len(transactions)} transactions. (User expected 55)")

if __name__ == "__main__":
    test_chase_real_parser()
