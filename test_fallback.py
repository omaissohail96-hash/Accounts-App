import pdfplumber
from bank_data_analysis import FallbackStatementParser
from document_parser import DocumentParser

dp = DocumentParser()
with open('bank statments/AMEX.pdf', 'rb') as f:
    text = f.read()

lines, ok, _ = dp.parse_document(text, 'AMEX.pdf')

parser = FallbackStatementParser()
txs, meta = parser.parse_statement(lines)
print(f"Fallback extracted {len(txs)} transactions.")
for i, t in enumerate(txs):
    print(f"  {i}: {t['date']} | {t['amount']} | {t['description']}")
