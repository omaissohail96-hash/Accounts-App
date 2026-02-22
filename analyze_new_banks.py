import os
import sys
from pathlib import Path

# Add the current directory to sys.path so we can import modules
sys.path.append(str(Path().absolute()))

from bank_data_analysis import DocumentParser

def analyze_statements():
    bank_statements_dir = Path("bank statments")
    files = list(bank_statements_dir.glob("*.pdf"))
    
    dp = DocumentParser()
    
    for pdf_file in files:
        print(f"--- Analyzing {pdf_file.name} ---")
        try:
            with open(pdf_file, "rb") as f:
                file_bytes = f.read()
            
            lines, ok, unreadable = dp.parse_document(file_bytes, pdf_file.name)
            
            if not ok:
                print(f"FAILED to parse {pdf_file.name}")
                continue
            
            # Print first 50 lines to see structure
            for i, line in enumerate(lines[:100]):
                print(f"{i:3}: {line}")
                
            print(f"--- End of {pdf_file.name} (Top 100 lines) ---\n")
            
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")

if __name__ == "__main__":
    analyze_statements()
