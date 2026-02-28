import json
from bank_statement_parser import BankStatementParser

parser = BankStatementParser()
try:
    res = parser.parse_statement(r'bank statments/AMEX.pdf')
    for t in res:
        if t.get('description') == 'UNKNOWN' or t.get('vendor') == 'UNKNOWN':
            print(t)
    print(f"Total extracted: {len(res)}")
except Exception as e:
    import traceback
    traceback.print_exc()
