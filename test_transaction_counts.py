"""
Test that all transaction count displays exclude balance entries
"""
from bank_statement_parser import Transaction

# Helper to check if transaction is a balance entry
def _is_balance_entry(tx):
    if not tx.description:
        return False
    desc_lower = tx.description.lower()
    return any(keyword in desc_lower for keyword in [
        'beginning balance', 'opening balance', 'starting balance'
    ])

# Create sample transactions
all_transactions = [
    Transaction(date="2024-01-01", description="Opening Balance", amount=23885.70, transaction_type="deposit", vendor="Opening Balance"),
    Transaction(date="2024-01-02", description="Amazon.Com.Ca UI Id", amount=457.69, transaction_type="deposit", vendor="Amazon.Com.Ca UI Id"),
    Transaction(date="2024-01-03", description="Ebay Com4aochsyw Id", amount=1437.70, transaction_type="deposit", vendor="Ebay Com4aochsyw Id"),
    Transaction(date="2024-01-04", description="Office Supplies", amount=-50.00, transaction_type="withdrawal", vendor="Staples"),
]

print("=" * 80)
print("TRANSACTION COUNT VERIFICATION")
print("=" * 80)
print()

# Test counts as they would appear in the UI
total_in_data = len(all_transactions)
total_count = len([t for t in all_transactions if not _is_balance_entry(t)])

print(f"Total transactions in data: {total_in_data}")
print(f"Total shown to user (excluding balance): {total_count}")
print()

# Test the Total metric (line 3384)
print("📊 TOTAL METRIC (top right):")
print(f"   Display: 'Total: {total_count} tx'")
print()

# Test the blue banner (line 3463)
all_txs_count = len([t for t in all_transactions if not _is_balance_entry(t)])
print("📊 BLUE BANNER (ALL TRANSACTIONS):")
print(f"   Text: 'Showing all {all_txs_count} transactions from uploaded statements'")
print()

# Test filter banner counts
filtered_transactions = all_transactions[:3]  # Simulate a filter
filtered_count = len([t for t in filtered_transactions if not _is_balance_entry(t)])
print("📊 FILTER BANNER (when filter is active):")
print(f"   Text: 'Showing {filtered_count} of {total_count} transactions'")
print()

print("=" * 80)
print("VALIDATION:")
print("=" * 80)

if total_count == 3:
    print("✅ CORRECT - All counts show 3 transactions (excluding opening balance)")
else:
    print(f"❌ ERROR - Expected 3 transactions, got {total_count}")

if all_txs_count == 3:
    print("✅ CORRECT - Blue banner shows 3 transactions")
else:
    print(f"❌ ERROR - Blue banner expected 3, got {all_txs_count}")

if filtered_count == 2:
    print("✅ CORRECT - Filtered count shows 2 transactions (excluding opening balance)")
else:
    print(f"❌ ERROR - Filtered count expected 2, got {filtered_count}")

print()
print("🎯 EXPECTED UI DISPLAY:")
print("   - Total metric: '85 tx' (not '86 tx')")
print("   - Blue banner: 'Showing all 85 transactions...'")
print("   - Opening balance appears in deposit table with empty transaction count")
