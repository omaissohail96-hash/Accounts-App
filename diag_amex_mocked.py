import sys
from unittest.mock import MagicMock

# Comprehensive Streamlit Mock
class MockStreamlit:
    def __init__(self):
        self.session_state = MagicMock()
        self.session_state.user_id = "test_user"
        self.session_state.active_business = "test_business"
        self.session_state.user = {"id": "test_user"}
        self.session_state.get.return_value = None
    
    def tabs(self, titles):
        return [MagicMock() for _ in titles]
    
    def __getattr__(self, name):
        return MagicMock()

mock_st = MockStreamlit()
sys.modules["streamlit"] = mock_st

# Mock other potential blockers
sys.modules["plotly"] = MagicMock()
sys.modules["plotly.graph_objects"] = MagicMock()
sys.modules["plotly.express"] = MagicMock()
sys.modules["pytesseract"] = MagicMock() # Mock pytesseract too to avoid cmd errors

from pathlib import Path
BASE_DIR = r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App"
sys.path.append(BASE_DIR)

# Directly import the components we need, bypassing top-level execution if possible?
# Unfortunately, Python executes the module on import.
# But with the mocks above, it should pass.

from bank_data_analysis import FallbackStatementParser, UniversalParser, DocumentParser

def diag():
    p_amex = Path(BASE_DIR) / "bank statments" / "AMEX.pdf"
    file_bytes = p_amex.read_bytes()
    
    # We need real OCR for diagnostic, so let's use the real pytesseract here
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    
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

if __name__ == "__main__":
    diag()
