"""
Diagnostic script for Fifth Third Bank parsing.
Shows OCR lines and which ones are NOT being matched as transactions.
Usage: python diag_53.py "path/to/FifthThird.pdf"
"""
import sys
import re
from document_parser import DocumentParser
from bank_statement_parser import BankStatementParser, BankName, BANK_LAYOUTS

def main():
    if len(sys.argv) < 2:
        print("Usage: python diag_53.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    doc_parser = DocumentParser()
    
    print(f"--- Reading {pdf_path} ---")
    with open(pdf_path, 'rb') as f:
        file_bytes = f.read()
    
    lines, is_readable, bad_pages = doc_parser.parse_document(file_bytes, pdf_path)
    print(f"Extracted {len(lines)} lines. Unreadable pages: {bad_pages}")
    print()

    parser = BankStatementParser(BankName.FIFTH_THIRD)
    text = '\n'.join(lines)
    statement = parser.parse(text)

    print(f"\n=== PARSED {len(statement.transactions)} TRANSACTIONS ===")
    for t in statement.transactions:
        print(f"  {t.date.date()} | {t.type.value:8s} | {t.amount:>12,.2f} | {t.description[:60]}")

    print(f"\n=== ALL OCR LINES ({len(lines)}) ===")
    layout = BANK_LAYOUTS[BankName.FIFTH_THIRD]
    section_kw = list(layout.section_headers.keys())
    current_section = None
    date_re = re.compile(r'^\s*\d{1,2}/\d{1,2}')
    
    for i, line in enumerate(lines):
        lu = line.upper().strip()
        
        # Track section
        for kw in section_kw:
            if kw in lu:
                current_section = kw
                break

        # Highlight lines that look transactional (start with a date) but may not be parsed
        if date_re.match(line):
            matched = any(
                t.raw_line and t.raw_line.strip() == line.strip()
                for t in statement.transactions
            )
            flag = "✅" if matched else "❌ MISSED"
            print(f"  [{i:03d}] {flag} | section={current_section} | {line.strip()[:100]}")
        else:
            print(f"  [{i:03d}]          | {lu[:60]}")

if __name__ == "__main__":
    main()
