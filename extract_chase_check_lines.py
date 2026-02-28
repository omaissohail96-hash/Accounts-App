import sys
import os
from document_parser import DocumentParser

def extract_check_lines():
    pdf_path = r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App\bank statments\20 Dec Chase.pdf"
    dp = DocumentParser()
    with open(pdf_path, "rb") as f:
        content = f.read()
        lines, ok, unreadable = dp.parse_document(content, "20 Dec Chase.pdf")
        
    print("=== Lines 290 to 310 ===")
    for i in range(290, min(310, len(lines))):
        print(f"{i:4}: {lines[i]}")

if __name__ == "__main__":
    extract_check_lines()
