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
    found = False
    for i, line in enumerate(lines):
        if "CHECKS PAID" in line.upper() and i > 100: # Skip the summary section
            print(f"Found Detail Section at line {i}: {line}")
            for j in range(max(0, i-2), min(len(lines), i+30)):
                print(f"{j:4}: {lines[j]}")
            found = True
            # Don't break, there might be multiple pages
            
    if not found:
        print("Detail 'CHECKS PAID' section not found after line 100")
        # Let's just print a chunk of lines from page 2 onwards
        print("\n--- Sample lines from Page 2 onwards ---")
        for i in range(100, 200):
            if i < len(lines):
                print(f"{i:4}: {lines[i]}")

if __name__ == "__main__":
    search_checks()
