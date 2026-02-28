import sys, os
import re
from datetime import datetime

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from bank_statement_parser import parse_bank_statement, BankName, BankLayout

def test_boa_robust_parsing():
    # Target Line 1: VW CREDIT with ~
    line1 = '12/22/25 VW CREDIT, INC. DES:AUTO DEBIT ID:000008146959399 INDN:AZIZ FAMILY MEDICAL CE CO ~499.99'
    
    # Target Line 2 & 3: Checks with messy artifacts
    line2 = '12/22/25 207┬░ -35,000.00 12/22/25 394┬░ -350.00'
    line3 = '12/22/25 354┬░ -500.00 12/22/25 402 -200.00'
    
    text = f"WITHDRAWALS AND OTHER DEBITS\n{line1}\nCHECKS\n{line2}\n{line3}\n"
    
    print("Testing BoA Robust Parsing...")
    # Debug the regex directly first
    layout = BankLayout(bank_name=BankName.BOA, check_regex=r'(\d{1,2}/\d{1,2}(?:/\d{2,4}|/\d{2})?)\s+(\S+)\s+(?:~|_|:)*([-~+]?[\d,\.]+\.\d{2})', transaction_regex='', column_mapping={}, section_headers={})
    print("\nDirect Regex Debug on Line 2:")
    for i, m in enumerate(re.finditer(layout.check_regex, line2)):
        print(f"  Match {i+1}: G1='{m.group(1)}', G2='{m.group(2)}', G3='{m.group(3)}'")
    
    result = parse_bank_statement(text, bank_name=BankName.BOA, manual_year=2025)
    
    print(f"Transactions found: {len(result.transactions)}")
    
    for i, tx in enumerate(result.transactions):
        date_str = tx.date.strftime('%Y-%m-%d') if tx.date else "NONE"
        print(f"  Tx {i+1}: Date: {date_str:<10} | Desc: {tx.description[:30]:<30} | Amt: {tx.amount:>10.2f}")

    # Validation
    expected_count = 5 
    if len(result.transactions) != expected_count:
        print(f"FAIL: Expected {expected_count} transactions, got {len(result.transactions)}")
        return False
        
    for i, tx in enumerate(result.transactions):
        if tx.date is None:
            print(f"FAIL: Transaction {i+1} has None date. Description: {tx.description}")
            return False

    print("PASS: BoA robustness verified")
    return True

if __name__ == "__main__":
    test_boa_robust_parsing()
