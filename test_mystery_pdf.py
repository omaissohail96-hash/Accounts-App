import pdfplumber
from bank_statement_parser import detect_bank, parse_bank_statement

pdf_path = 'bank statments/2CED020D-078D-423B-A7BE-CFE61B680C1D-list.pdf'

try:
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() or '' for page in pdf.pages])
        
    print(f"Total extracted chars: {len(text)}")
    bank = detect_bank(text)
    print(f"Detected bank: {bank}")
    
    result = parse_bank_statement(text, bank_name=bank)
    print(f"Parsed transactions: {len(result.transactions)}")
    
    for i, t in enumerate(result.transactions[:5], 1):
        print(f"  {i}: {t.date} | {t.amount:10.2f} | {t.description[:50]}")
        
    if len(result.transactions) < 5:
        for t in result.transactions:
            print(f"  ALL: {t.date} | {t.amount} | {t.description}")
            
    print(f"Errors: {result.errors}")

except Exception as e:
    print(f"Script Error: {e}")
