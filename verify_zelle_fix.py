import sys, os
from datetime import datetime

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from bank_statement_parser import parse_bank_statement, BankName

def verify_zelle_fix():
    line = '12/17/25 Zelle payment from BENEDICTO PANA JR for "US Immigration Physical Exams"; Conf# 270944002 930.00'
    text = f"DEPOSITS AND OTHER CREDITS\n{line}\n"
    
    print("Testing Zelle Amount Fix...")
    result = parse_bank_statement(text, bank_name=BankName.BOA, manual_year=2025)
    
    if not result.transactions:
        print("FAIL: No transactions parsed")
        return False
        
    tx = result.transactions[0]
    print(f"  Parsed Description: {tx.description}")
    print(f"  Parsed Amount:      {tx.amount}")
    
    # Validation
    expected_amount = 930.00
    if abs(tx.amount - expected_amount) > 0.01:
        print(f"FAIL: Expected {expected_amount}, got {tx.amount}")
        return False
        
    if "BENEDICTO" not in tx.description:
        print("FAIL: Description lost vendor name")
        return False
        
    if "Conf# 270944002" not in tx.description:
        print("FAIL: Confirmation number missing from description")
        return False
        
    print("PASS: Zelle fix verified")
    return True

if __name__ == "__main__":
    if verify_zelle_fix():
        print("\nFix Verified Successfully!")
    else:
        sys.exit(1)
