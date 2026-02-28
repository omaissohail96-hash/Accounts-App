import os
from pathlib import Path
import PyPDF2

def search_pdfs():
    folder = Path('bank statments')
    for pdf_path in sorted(folder.glob('*.pdf')):
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                found = False
                for page in reader.pages:
                    text = page.extract_text()
                    if text and ('Q35' in text or '847-635-3342' in text):
                        print(f"FOUND MATCH IN: {pdf_path.name}")
                        found = True
                        break
                if found:
                    continue
        except Exception as e:
            print(f"Error reading {pdf_path.name}: {e}")

search_pdfs()
