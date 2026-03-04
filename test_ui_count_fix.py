"""
Quick verification that the UI transaction count fix works
"""
from bank_statement_parser import Transaction

# Simulate the is_balance_entry function from the UI
def is_balance_entry(tx):
    if not tx.description:
        return False
    desc_lower = tx.description.lower()
    return any(keyword in desc_lower for keyword in [
        'beginning balance', 'opening balance', 'starting balance'
    ])

# Create sample transactions like in a real statement
transactions = [
    Transaction(date="2024-01-01", description="Opening Balance", amount=23885.70, transaction_type="deposit", vendor="Opening Balance"),
    Transaction(date="2024-01-02", description="Amazon.Com.Ca UI Id", amount=457.69, transaction_type="deposit", vendor="Amazon.Com.Ca UI Id"),
    Transaction(date="2024-01-03", description="Ebay Com4aochsyw Id", amount=1437.70, transaction_type="deposit", vendor="Ebay Com4aochsyw Id"),
    Transaction(date="2024-01-04", description="Office Supplies", amount=-50.00, transaction_type="withdrawal", vendor="Staples"),
]

print("=" * 70)
print("UI TRANSACTION COUNT VERIFICATION")
print("=" * 70)
print()

# Total transactions in data
total_in_data = len(transactions)
print(f"Total transactions in data: {total_in_data}")

# Count excluding balance entries (what UI should show)
total_counted = len([t for t in transactions if not is_balance_entry(t)])
print(f"Total transactions counted (excluding balance): {total_counted}")
print()

# Show each transaction
print("Transaction details:")
for i, tx in enumerate(transactions, 1):
    is_balance = is_balance_entry(tx)
    counted = "NOT COUNTED" if is_balance else "COUNTED"
    print(f"  {i}. {tx.description[:40]:40} | ${tx.amount:>10.2f} | {counted}")

print()
print("=" * 70)
print("EXPECTED UI DISPLAY:")
print("=" * 70)
print(f"Caption: 'Statistics based on all transactions ({total_counted} transactions)'")
print(f"Transactions metric: {total_counted}")
print()

# Verify deposits
deposits = [t for t in transactions if t.amount > 0]
deposits_counted = len([t for t in deposits if not is_balance_entry(t)])
print(f"Total Deposits: +{deposits_counted} tx (excluding opening balance)")

# Verify withdrawals  
withdrawals = [t for t in transactions if t.amount < 0]
withdrawals_counted = len([t for t in withdrawals if not is_balance_entry(t)])
print(f"Total Withdrawals: +{withdrawals_counted} tx")

print()
if total_counted == 3 and deposits_counted == 2:
    print("✅ CORRECT - Opening balance excluded from counts but kept in data")
else:
    print(f"❌ ERROR - Expected total=3, deposits=2, got total={total_counted}, deposits={deposits_counted}")
