import pdfplumber
import os

pdf_path = 'bank statments/FifthThird 53.pdf'

try:
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() or '' for page in pdf.pages])
        
    print(f"Total extracted chars: {len(text)}")
    
    if "Q35" in text or "847" in text:
        print("FOUND Q35 or 847 in Fifth Third!")
        
    # Also print any lines with Q35
    for ln in text.split('\n'):
        if 'Q35' in ln:
            print("LINE:", repr(ln))
            
except Exception as e:
    print(f"Error: {e}")
