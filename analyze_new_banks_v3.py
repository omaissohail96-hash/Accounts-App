import sys
from unittest.mock import MagicMock
from pathlib import Path

# Mock streamlit before importing bank_data_analysis
st = MagicMock()
# Mock session_state as a dict-like object
st.session_state = MagicMock()
st.session_state.user_id = "test_user"
st.session_state.get.return_value = None # Default for .get()

sys.modules["streamlit"] = st

from bank_data_analysis import DocumentParser

def analyze_statements():
    sys.stdout.reconfigure(encoding='utf-8')
    bank_statements_dir = Path("bank statments")
    files = list(bank_statements_dir.glob("*.pdf"))
    
    dp = DocumentParser()
    
    for pdf_file in files:
        print(f"--- Analyzing {pdf_file.name} ---")
        try:
            with open(pdf_file, "rb") as f:
                file_bytes = f.read()
            
            # Use DocumentParser which handles OCR
            lines, ok, unreadable = dp.parse_document(file_bytes, pdf_file.name)
            
            if not ok:
                print(f"FAILED to parse {pdf_file.name}")
                continue
            
            # Print first 100 lines
            for i, line in enumerate(lines[:100]):
                print(f"{i:3}: {line}")
                
            print(f"--- End of {pdf_file.name} (Top 100 lines) ---\n")
            
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")

if __name__ == "__main__":
    analyze_statements()
