import pdfplumber
from pathlib import Path

def analyze_statements():
    bank_statements_dir = Path("bank statments")
    files = list(bank_statements_dir.glob("*.pdf"))
    
    for pdf_file in files:
        print(f"--- Analyzing {pdf_file.name} ---")
        try:
            with pdfplumber.open(pdf_file) as pdf:
                all_text = ""
                for page in pdf.pages[:3]: # Look at first 3 pages
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"
                
                lines = all_text.split("\n")
                for i, line in enumerate(lines[:100]):
                    print(f"{i:3}: {line}")
                    
            print(f"--- End of {pdf_file.name} (Top 100 lines) ---\n")
            
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")

if __name__ == "__main__":
    analyze_statements()
