"""
Visual test to confirm Opening Balance display exactly as shown in UI
"""
from bank_statement_parser import Transaction
from report_generator import ReportGenerator
import pandas as pd

# Create realistic transaction set matching user's data
transactions = [
    Transaction(date="2024-01-01", description="Opening Balance", amount=23885.70, transaction_type="deposit", vendor="Opening Balance"),
    Transaction(date="2024-01-02", description="Amazon.Com.Ca UI Id", amount=457.69, transaction_type="deposit", vendor="Amazon.Com.Ca UI Id"),
    Transaction(date="2024-01-03", description="Amazon.Com.Ca UI Id", amount=200.00, transaction_type="deposit", vendor="Amazon.Com.Ca UI Id"),  # Another Amazon to get count=2
    Transaction(date="2024-01-04", description="Ebay Com4Aochsyw Id", amount=1437.70, transaction_type="deposit", vendor="Ebay Com4Aochsyw Id"),
    Transaction(date="2024-01-05", description="Ebay ComfxSEff9E Id", amount=1897.04, transaction_type="deposit", vendor="Ebay ComfxSEff9E Id"),
]

# Generate report
rg = ReportGenerator()
df = rg.generate_deposits_summary(transactions)

print("=" * 100)
print("ALL DEPOSITS SUMMARY (BY SOURCE/VENDOR)")
print("=" * 100)
print()

# Display with proper formatting
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# Show just the key columns
display_df = df[['Source/Vendor', 'Transaction Count', 'Subtotal ($)']].copy()

print(display_df.to_string(index=False))
print()
print("=" * 100)
print("VERIFICATION CHECKLIST:")
print("=" * 100)

# Check row by row
first_row = df.iloc[0]
print(f"✓ Row 1: {first_row['Source/Vendor']}")
print(f"  - Transaction Count: '{first_row['Transaction Count']}' → Should be EMPTY (no '0')")
print(f"  - Subtotal: ${first_row['Subtotal ($)']:,.2f}")
print()

for idx in range(1, min(4, len(df)-1)):  # Show next few rows
    row = df.iloc[idx]
    count_val = row['Transaction Count']
    print(f"✓ Row {idx+1}: {row['Source/Vendor']}")
    print(f"  - Transaction Count: {count_val if count_val != '' else '(empty)'}")
    print(f"  - Subtotal: ${row['Subtotal ($)']:,.2f}")
    print()

# Final validation
print("=" * 100)
assert df.iloc[0]['Source/Vendor'] == 'Opening Balance', "Opening Balance must be first!"
assert df.iloc[0]['Transaction Count'] == '', f"Transaction Count must be empty string, got: '{df.iloc[0]['Transaction Count']}'"
assert df.iloc[0]['Subtotal ($)'] == 23885.70, "Subtotal must be correct!"

print("✅✅✅ ALL VALIDATIONS PASSED! ✅✅✅")
print()
print("🎯 EXPECTED IN STREAMLIT:")
print("   Opening Balance is at the TOP")
print("   Opening Balance Transaction Count cell is EMPTY (not showing '0')")
print("   Opening Balance amount shows: $23,885.70")
print()
print("💡 PLEASE REFRESH YOUR STREAMLIT APP NOW (F5 or Ctrl+R)")
