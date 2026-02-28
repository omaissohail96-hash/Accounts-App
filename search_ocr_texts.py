from document_parser import DocumentParser
from pathlib import Path

pdf_folder = Path('bank statments')
dp = DocumentParser()

for pdf_path in sorted(pdf_folder.glob('*.pdf')):
    print(f"\nEvaluating {pdf_path.name}...")
    try:
        with open(pdf_path, 'rb') as f:
            file_bytes = f.read()
            
        lines, ok, unreadable = dp.parse_document(file_bytes, pdf_path.name)
        if ok and lines:
            text = '\n'.join(lines)
            if 'Q35' in text or '847-635-3342' in text:
                print(f"!!! MATCH FOUND IN {pdf_path.name} !!!")
                for ln in lines:
                    if 'Q35' in ln or '847-635' in ln:
                        print("LINE:", repr(ln))
        else:
            print("Failed to read text")
    except Exception as e:
        print(f"Error: {e}")
