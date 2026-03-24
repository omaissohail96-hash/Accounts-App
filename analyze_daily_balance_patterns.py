#!/usr/bin/env python3
"""
Examine bank statement PDFs to find daily ending balance patterns
"""
import pdfplumber
from pathlib import Path

# List of PDF files to check
pdf_files = [
    "bank statments/20 Dec Chase.pdf",
    "bank statments/AMEX.pdf",
    "bank statments/BMO.pdf",
    "bank statments/BoA.pdf",
]

for pdf_file in pdf_files:
    if not Path(pdf_file).exists():
        continue
    
    print(f"\n{'='*70}")
    print(f"FILE: {pdf_file}")
    print(f"{'='*70}")
    
    try:
        with pdfplumber.open(pdf_file) as pdf:
            # Check first 3 pages
            for page_num, page in enumerate(pdf.pages[:3], 1):
                text = page.extract_text()
                if text:
                    lines = text.split('\n')
                    # Find lines with "daily" or "ending"
                    matching_lines = [
                        line for line in lines 
                        if 'daily' in line.lower() or 'ending' in line.lower()
                    ]
                    if matching_lines:
                        print(f"\nPage {page_num} - Daily/Ending Balance lines:")
                        for i, line in enumerate(matching_lines[:10]):
                            print(f"  [{i+1}] {line}")
    except Exception as e:
        print(f"Error reading {pdf_file}: {e}")

print("\n" + "="*70)
print("Analysis complete")
