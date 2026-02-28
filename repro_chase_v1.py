import re
from bank_statement_parser import BankStatementParser, BankName

def test_chase_parsing():
    parser = BankStatementParser(BankName.CHASE)
    
    # Mock Chase line
    line = "12/16 CCD UnitedHealthcare HCCLAIMPMT 1,113.55"
    
    raw_text = """
DEPOSITS AND ADDITIONS
12/16 CCD UnitedHealthcare HCCLAIMPMT 1,113.55
"""
    try:
        parsed = parser.parse(raw_text, manual_year=2026)
        print(f"Parsed {len(parsed.transactions)} transactions.")
        for tx in parsed.transactions:
            print(f"Date: {tx.date}, Amount: {tx.amount}, Desc: {tx.description}")
    except Exception as e:
        print(f"❌ PARSING FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chase_parsing()
