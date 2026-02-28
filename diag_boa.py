"""
Diagnostic script for Bank of America (BoA) parsing.
"""
import sys, os, logging

# Suppress logging for clean output
logging.disable(logging.CRITICAL)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bank_statement_parser import parse_bank_statement, BankName, TransactionCategory

def run_diag():
    with open("ocr_dumps/BoA.txt", "r", encoding="utf-8") as f:
        text = f.read()

    result = parse_bank_statement(text)
    
    print(f"Bank: {result.bank_name}")
    print(f"Beginning Balance: {result.beginning_balance}")
    print(f"Ending Balance: {result.ending_balance}")
    print(f"Total Transactions: {len(result.transactions)}")
    
    deposits = [t for t in result.transactions if t.amount > 0]
    withdrawals = [t for t in result.transactions if t.amount < 0 and t.category != TransactionCategory.CHECK]
    checks = [t for t in result.transactions if t.category == TransactionCategory.CHECK]
    
    print(f"\nSummary:")
    print(f"  Deposits:    {len(deposits)} (Expected 116)")
    print(f"  Withdrawals: {len(withdrawals)} (Expected 8 if checks separate, 22 total)")
    print(f"  Checks:      {len(checks)} (Expected 14)")
    
    print("\n--- FIRST 20 TRANSACTIONS ---")
    for t in result.transactions[:20]:
        print(f"  {t.date.strftime('%Y-%m-%d')} | {t.amount:10.2f} | {t.category.value:12} | {t.description[:50]}")

    print("\n--- CHECKS ---")
    for t in checks:
        print(f"  {t.date.strftime('%Y-%m-%d')} | {t.amount:10.2f} | {t.check_number:8} | {t.description}")

    # Check for specific lines missing amounts
    print("\n--- POSSIBLY MISSING (e.g. 12/04 Square) ---")
    for line in text.splitlines():
        if "12/04/25" in line and "Square" in line:
            print(f"  RAW: {line}")
            found = [t for t in result.transactions if t.date.month == 12 and t.date.day == 4 and "Square" in t.description]
            print(f"  EXTRACTED: {found}")

if __name__ == "__main__":
    run_diag()
