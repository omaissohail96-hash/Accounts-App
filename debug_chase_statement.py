import pdfplumber
import sys

pdf_path = r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App\bank statments\2CED020D-078D-423B-A7BE-CFE61B680C1D-list.pdf"

print("=== CHASE BANK STATEMENT DEBUG ===\n")

try:
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Total pages: {len(pdf.pages)}\n")
        
        # Extract text from all pages
        for i, page in enumerate(pdf.pages):
            print(f"--- PAGE {i+1} ---")
            text = page.extract_text()
            print(text[:2000] if text else "No text extracted")
            print("\n")
            
            if i >= 2:  # Limit to first 3 pages for debugging
                print("... (showing first 3 pages only)")
                break
                
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
