import sys
import os
from document_parser import DocumentParser

def search_checks():
    pdf_path = r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App\bank statments\20 Dec Chase.pdf"
    dp = DocumentParser()
    with open(pdf_path, "rb") as f:
        content = f.read()
        lines, ok, unreadable = dp.parse_document(content, "20 Dec Chase.pdf")
        
    print("=== Searching for CHECKS detail ===")
    for i, line in enumerate(lines):
        if "CHECK" in line.upper():
            print(f"[{i}]: {line}")

if __name__ == "__main__":
    search_checks()
