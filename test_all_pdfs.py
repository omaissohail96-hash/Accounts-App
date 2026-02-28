import os
from pathlib import Path
from document_parser import DocumentParser
from bank_data_analysis import parse_with_multi_bank_parser

pdf_folder = Path('bank statments')
dp = DocumentParser()

for pdf_path in sorted(pdf_folder.glob('*.pdf')):
    print(f"\nProcessing {pdf_path.name}...")
    try:
        with open(pdf_path, 'rb') as f:
            file_bytes = f.read()
            
        lines, ok, unreadable = dp.parse_document(file_bytes, pdf_path.name)
        if not ok or not lines:
            print(f"Failed to read text from {pdf_path.name}")
            continue
            
        bank_txs, b_meta = parse_with_multi_bank_parser(
            lines,
            pdf_path.name,
            include_opening_balance=False,
            extract_check_memos=True
        )
        
        if bank_txs is not None:
            print(f"Multi-bank parser extracted {len(bank_txs)} transactions.")
            for i, t in enumerate(bank_txs[:3]):
                print(f"  {i}: {t.get('date')} | {t.get('amount')} | {t.get('description')[:50]}")
            if len(bank_txs) < 5:
                for t in bank_txs:
                    if 'Q35' in str(t.get('description')):
                        print(f"!!! FOUND IT IN {pdf_path.name} !!!")
        else:
            print(f"Multi-bank parser returned None.")
            
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {e}")
