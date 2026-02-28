import sys, os
import re
from datetime import datetime

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import logging
logging.basicConfig(level=logging.INFO)
from bank_statement_parser import parse_bank_statement, BankName, TransactionCategory, TransactionType

def find_missing_deposit():
    ocr_path = r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App\ocr_dumps\BoA.txt"
    with open(ocr_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print("Parsing BoA Statement...")
    result = parse_bank_statement(text, bank_name=BankName.BOA, manual_year=2025)
    
    # Daily Reconciliation
    print("\n--- Daily Reconciliation ---")
    current_bal = result.beginning_balance or 0
    tx_by_date = {}
    for t in result.transactions:
        d_key = t.date.strftime('%m/%d') if t.date else "N/A"
        tx_by_date.setdefault(d_key, []).append(t)
    
    # BoA daily ledger
    ledger_content_match = re.search(r'(?i)Daily ledger balances(.*?)(?:Service fees|Checks|Account summary|Total checks|\Z)', text, re.DOTALL)
    ledger = {}
    if ledger_content_match:
        for entry in re.finditer(r'(\d{1,2}/\d{1,2})\s+([\d, ]+\.\d{2})', ledger_content_match.group(1)):
            ledger[entry.group(1)] = float(re.sub(r'[^\d.]', '', entry.group(2)))
    
    dates = sorted(ledger.keys(), key=lambda x: datetime.strptime(x, "%m/%d"))
    prev_date_bal = result.beginning_balance or 226369.75
    for d in dates:
        expected = ledger[d]
        day_txs = tx_by_date.get(d, [])
        net_change = sum(t.amount for t in day_txs)
        actual = prev_date_bal + net_change
        diff = expected - actual
        
        status = "OK" if abs(diff) < 0.01 else f"GAP: ${diff:,.2f}"
        print(f"  {d}: Expected: ${expected:10,.2f} | Actual: ${actual:10,.2f} | {status}")
        prev_date_bal = expected # Anchor to ledger to isolate gaps

    # Final Balance Check
    final_actual = prev_date_bal 
    final_expected = ledger.get(dates[-1], 0)
    print(f"\nFinal Expected Balance: ${final_expected:,.2f}")
    print(f"Final Actual Balance (anchored): ${final_actual:,.2f}")
    
    # Real current balance in parser
    real_final = result.beginning_balance + sum(t.amount for t in result.transactions)
    print(f"Parser Final Balance: ${real_final:,.2f}")
    print(f"Diff: ${final_expected - real_final:,.2f}")

    print("\n--- Transactions for 12/16 ---")
    day_1216 = [t for t in result.transactions if t.date and t.date.strftime('%m/%d') == '12/16']
    for t in day_1216:
        print(f"  {t.type.value:<10} {t.description[:40]:<40} ${t.amount:10,.2f}")
    
    # Show deposit lines without dates
    print("\n--- Deposit Lines WITHOUT Dates ---")
    sections = re.split(r'(?i)(Deposits and other credits|Withdrawals and other debits|Daily ledger balances)', text)
    deposit_text = ""
    for i in range(len(sections)):
        if "Deposits and other credits" in sections[i]:
            if i+1 < len(sections):
                deposit_text += sections[i+1]
                
    for line in deposit_text.split('\n'):
        line = line.strip()
        if line and not re.match(r'^\d{1,2}/\d{1,2}', line):
            if not any(k in line.upper() for k in ["CONTINUED", "DATE", "DESCRIPTION", "AMOUNT"]):
                print(f"  ORPHAN: {line}")

if __name__ == "__main__":
    find_missing_deposit()
