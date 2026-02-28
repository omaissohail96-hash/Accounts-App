import pdfplumber

pdf_path = 'bank statments/2CED020D-078D-423B-A7BE-CFE61B680C1D-list.pdf'

try:
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Total pages: {len(pdf.pages)}")
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ''
            print(f'=== PAGE {i} ({len(text)} chars) ===')
            lines = text.split('\n')
            for ln in lines[:30]:  # Print first 30 lines of each page
                print(repr(ln))
            print("...")
            for ln in lines[-10:]: # Print last 10 lines of each page
                print(repr(ln))
            print()
except Exception as e:
    print(f"Error: {e}")
