
from bank_statement_parser import USBankParser, BankName
import logging

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO)

ocr_text = """
Beginning Balance on Nov 3 $ 26,427.22 Number of Days in Statement
Other Deposits 2 8,500.00
Other Withdrawals 4 3,592.48-
Ending Balance on Nov 30, 2025 $ 31,334.74
energy Sune i nee ee ever v wien
Other Deposits
Date see of Transaction Ref Number Amount
Nov 4 Mobile 8352054773 6,500.00
Nov 25 Mobile Check Deposit 8352120036 2,000.00
Total Other Deposits $ 8,500.00
Other Withdrawals
Date Description of Transaction Ref Number Amount
Nov 5 Mobile Banking Transfer To Account 158205632041 $ 1,500.00-
Nov 13 Electronic Withdrawal To IL DEPT OF REVEN 100.00-
Nov 17 Card Withdrawal To TMG*T-MOBILE 1,114.70-
Nov 20 Electronic Withdrawal To COMCAST 877.78-
Total Other Withdrawals $ 3,592.48-
"""

parser = USBankParser()
statement = parser.parse(ocr_text, manual_year=2025)

print(f"Bank: {statement.bank_name}")
print(f"Beginning Balance: {statement.beginning_balance}")
print(f"Ending Balance:    {statement.ending_balance}")
print(f"Transactions detected: {len(statement.transactions)}")
for tx in statement.transactions:
    print(f"  {tx.date.strftime('%b %d')} | {tx.amount:10.2f} | {tx.description}")

expected_count = 6
expected_beg = 26427.22
expected_end = 31334.74

if (len(statement.transactions) == expected_count and 
    statement.beginning_balance == expected_beg and 
    statement.ending_balance == expected_end):
    print("\nSUCCESS: All 6 transactions and balances detected!")
else:
    print(f"\nFAILURE:")
    print(f"  Transactions: {len(statement.transactions)}/6")
    print(f"  Beginning:    {statement.beginning_balance}/{expected_beg}")
    print(f"  Ending:       {statement.ending_balance}/{expected_end}")
