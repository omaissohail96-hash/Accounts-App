import re
from bank_statement_parser import BankStatementParser, BankName

def test_bmo_stripping():
    parser = BankStatementParser(BankName.BMO)
    
    # These are lines from the user's screenshot
    lines = [
        "MONTHLY ACTIVITY DETAILS",
        "Dec 01 BEGINNING BALANCE $8,248.15",
        "Dec 16 EDI/EFT CCD+ CREDIT $464.43",
        "Dec 16 EDI/EFT CCD+ CREDIT $666.37",
        "Dec 16 EDI/EFT CCD+ CREDIT $803.99",
        "Dec 16 EDI/EFT CCD+ CREDIT $1,113.55",
        "Dec 16 EDI/EFT CCD+ CREDIT $1,394.64",
        "Dec 16 EDI/EFT CCD+ CREDIT $2,639.76",
        "Dec 31 INTEREST PAID $0.11 $34,745.80",
        "CHECKS",
        "Check 123456 Dec 20 $1,500.00", # Potential BMO check format
        "Dec 21 TRANSFER TO 123456789 (100.00) $33,145.80" # Transfer format
    ]
    
    raw_text = "\n".join(lines)
    parsed = parser.parse(raw_text, manual_year=2026)
    
    print(f"Parsed {len(parsed.transactions)} transactions.")
    for tx in parsed.transactions:
        print(f"Date: {tx.date}, Amount: {tx.amount}, Desc: {tx.description}")
        
    # Check for stripping
    for tx in parsed.transactions:
        if "464.43" in tx.description and abs(tx.amount) != 464.43:
            print(f"❌ STRIPPING DETECTED: Expected 464.43, got {tx.amount}")
        if "803.99" in tx.description and abs(tx.amount) != 803.99:
            print(f"❌ STRIPPING DETECTED: Expected 803.99, got {tx.amount}")

    # Check for missing
    if len(parsed.transactions) != len(lines):
        print(f"❌ MISSING TRANSACTIONS: Expected {len(lines)}, got {len(parsed.transactions)}")

if __name__ == "__main__":
    test_bmo_stripping()
