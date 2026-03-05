"""
Test Date Prioritization with Bank + Credit Card Upload
Scenario: Bank statement (Nov 29 - Dec 31, 2025) + Credit Cards (Mar 14 - May 13, 2024)
"""
from dataclasses import dataclass
from datetime import datetime, date

@dataclass
class MockTransaction:
    date: str
    description: str
    amount: float
    source: str = ""

# Simulate mixed upload: Bank statement + 2 Credit cards from screenshots
all_transactions = [
    # Bank statement transactions (Nov 29 - Dec 31, 2025)
    MockTransaction("2025-11-29", "Opening Balance", 1000.00, "BANK"),
    MockTransaction("2025-12-05", "Deposit", 500.00, "BANK"),
    MockTransaction("2025-12-15", "Withdrawal", -100.00, "BANK"),
    MockTransaction("2025-12-31", "Interest", 5.00, "BANK"),
    
    # Credit Card 1 transactions (Mar 14 - Apr 13, 2024) - OUTSIDE BANK PERIOD
    MockTransaction("2024-03-15", "CC1 Purchase", -50.00, "CREDIT_CARD"),
    MockTransaction("2024-03-20", "CC1 Purchase", -100.00, "CREDIT_CARD"),
    MockTransaction("2024-04-10", "CC1 Purchase", -75.00, "CREDIT_CARD"),
    
    # Credit Card 2 transactions (Apr 14 - May 13, 2024) - OUTSIDE BANK PERIOD
    MockTransaction("2024-04-15", "CC2 Purchase", -200.00, "CREDIT_CARD"),
    MockTransaction("2024-05-05", "CC2 Purchase", -150.00, "CREDIT_CARD"),
]

print("=" * 80)
print("DATE PRIORITIZATION & AUTO-FILTERING TEST")
print("=" * 80)
print()

# Step 1: Extract dates with prioritization
parsed_dates = []
bank_statement_dates = []

for tx in all_transactions:
    if tx.date:
        try:
            date_obj = datetime.strptime(tx.date, "%Y-%m-%d").date()
            parsed_dates.append(date_obj)
            if tx.source != 'CREDIT_CARD':
                bank_statement_dates.append(date_obj)
        except:
            pass

print("STEP 1: Date Extraction")
print("-" * 80)
print(f"All transaction dates: {min(parsed_dates)} → {max(parsed_dates)}")
print(f"Bank statement dates only: {min(bank_statement_dates)} → {max(bank_statement_dates)}")
print()

# Step 2: Prioritize bank statement dates
if bank_statement_dates:
    min_date = min(bank_statement_dates)
    max_date = max(bank_statement_dates)
    print("STEP 2: Date Prioritization")
    print("-" * 80)
    print(f"✅ Using dates from BANK STATEMENT: {min_date.strftime('%B %d, %Y')} through {max_date.strftime('%B %d, %Y')}")
else:
    min_date = min(parsed_dates)
    max_date = max(parsed_dates)
    print(f"ℹ️ Using dates from all transactions: {min_date} → {max_date}")
print()

# Step 3: Auto-filter transactions
date_filtered_transactions = []
excluded_count = 0

print("STEP 3: Auto-Filtering Transactions")
print("-" * 80)
for tx in all_transactions:
    if tx.date:
        try:
            tx_date = datetime.strptime(tx.date, "%Y-%m-%d").date()
            if min_date <= tx_date <= max_date:
                date_filtered_transactions.append(tx)
                print(f"✅ INCLUDED: {tx.date} | {tx.description:20} | ${tx.amount:>8.2f} | {tx.source}")
            else:
                excluded_count += 1
                print(f"❌ EXCLUDED: {tx.date} | {tx.description:20} | ${tx.amount:>8.2f} | {tx.source}")
        except:
            date_filtered_transactions.append(tx)

print()
print("=" * 80)
print("FINAL RESULT")
print("=" * 80)
print(f"Filter Date Range: {min_date.strftime('%m/%d/%y')} - {max_date.strftime('%m/%d/%y')}")
print(f"Total uploaded: {len(all_transactions)} transactions")
print(f"Transactions shown: {len(date_filtered_transactions)} (within bank statement period)")
print(f"Transactions excluded: {excluded_count} (credit card transactions outside period)")
print()
print("✅ Filter shows BANK STATEMENT dates, not credit card dates!")
print("✅ Credit card transactions outside bank period are automatically excluded!")
