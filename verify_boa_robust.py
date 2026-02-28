"""
Final verification for Bank of America (BoA) robust parsing.
"""
import sys, os, logging

# Configure logging to see the repair messages
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from bank_statement_parser import parse_bank_statement, BankName, TransactionCategory

def verify_boa():
    with open("ocr_dumps/BoA.txt", "r", encoding="utf-8") as f:
        text = f.read()

    print("\n--- Parsing BoA Statement ---")
    result = parse_bank_statement(text)
    
    print(f"Detected Bank: {result.bank_name}")
    print(f"Beginning Balance: {result.beginning_balance}")
    print(f"Ending Balance: {result.ending_balance}")
    print(f"Total Transactions: {len(result.transactions)}")
    
    deposits = [t for t in result.transactions if t.amount > 0]
    withdrawals = [t for t in result.transactions if t.amount < 0 and t.category != TransactionCategory.CHECK]
    checks = [t for t in result.transactions if t.category == TransactionCategory.CHECK]
    
    print(f"\nSummary Counts:")
    print(f"  Deposits:    {len(deposits)} (Expected 116)")
    print(f"  Withdrawals: {len(withdrawals)} (Expected 8 if checks separate, 22 total with checks)")
    print(f"  Checks:      {len(checks)} (Expected 14)")
    
    # Validation checks
    if result.bank_name != "boa":
        print("FAIL: Bank detection failed")
    
    if len(deposits) >= 115: # Allow slight OCR deviation if necessary, but aim for exact
        print("PASS: Deposit count is correct/near-perfect")
    else:
        print(f"FAIL: Only {len(deposits)} deposits found")

    if len(checks) == 14:
        print("PASS: All 14 checks extracted")
    else:
        print(f"FAIL: {len(checks)} checks extracted")

    # Specific Repair Verification
    # Line 154: MERCHANT SERVICE ... 329.41 (Missing minus in OCR?)
    ms_fee = next((t for t in result.transactions if "MERCH FEE" in t.description), None)
    if ms_fee:
        print(f"MERCH FEE Amount: {ms_fee.amount}")
        if ms_fee.amount < 0:
            print("PASS: MERCH FEE sign repaired/correct")
        else:
            print("FAIL: MERCH FEE sign is positive")

    # Line 160: FLAGG CREEK ... 151.67 (Missing minus in OCR?)
    flagg_creek = next((t for t in result.transactions if "FLAGG CREEK" in t.description), None)
    if flagg_creek:
        print(f"FLAGG CREEK Amount: {flagg_creek.amount}")
        if flagg_creek.amount < 0:
             print("PASS: FLAGG CREEK sign repaired/correct")

if __name__ == "__main__":
    verify_boa()
