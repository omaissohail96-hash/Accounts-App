"""
Test that Opening Balance appears at the top with empty transaction count
"""
from bank_statement_parser import Transaction
from report_generator import ReportGenerator

# Create test transactions
transactions = [
    Transaction(date="2024-01-02", description="Amazon.Com.Ca UI Id", amount=457.69, transaction_type="deposit", vendor="Amazon.Com.Ca UI Id"),
    Transaction(date="2024-01-01", description="Opening Balance", amount=23885.70, transaction_type="deposit", vendor="Opening Balance"),
    Transaction(date="2024-01-03", description="Ebay Com4aochsyw Id", amount=1437.70, transaction_type="deposit", vendor="Ebay Com4aochsyw Id"),
]

# Generate report
rg = ReportGenerator()
df = rg.generate_deposits_summary(transactions)

print("=" * 80)
print("DEPOSIT SUMMARY TABLE")
print("=" * 80)
print(df.to_string(index=False))
print()
print("=" * 80)
print("VERIFICATION:")
print("=" * 80)

# Check first row is Opening Balance
first_row = df.iloc[0]
print(f"✓ First row vendor: {first_row['Source/Vendor']}")
print(f"✓ Transaction Count: '{first_row['Transaction Count']}' (should be empty string)")
print(f"✓ Subtotal: ${first_row['Subtotal ($)']:,.2f}")
print()

# Validate
assert first_row['Source/Vendor'] == "Opening Balance", "Opening Balance should be first row"
assert first_row['Transaction Count'] == '', f"Transaction Count should be empty string, got: '{first_row['Transaction Count']}'"
assert first_row['Subtotal ($)'] == 23885.70, "Subtotal should be correct"

print("✅ ALL CHECKS PASSED!")
print("   - Opening Balance is at the top of the table")
print("   - Transaction Count is empty (not '0')")
print("   - Amount is correctly displayed")
