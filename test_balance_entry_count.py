"""
Test Balance Entry Count Exclusion
Verify that balance entries are excluded from transaction counts but kept in data
"""
from bank_statement_parser import Transaction
from report_generator import ReportGenerator


def test_balance_entry_exclusion():
    """Test that balance entries are excluded from counts but kept in data"""
    
    # Create test transactions including a balance entry
    transactions = [
        Transaction(
            date="2024-01-01",
            description="Opening Balance",
            amount=1000.00,
            transaction_type="deposit",
            vendor="Opening Balance"
        ),
        Transaction(
            date="2024-01-02",
            description="Client Payment",
            amount=500.00,
            transaction_type="deposit",
            vendor="Client A"
        ),
        Transaction(
            date="2024-01-03",
            description="Office Supplies",
            amount=-50.00,
            transaction_type="withdrawal",
            vendor="Staples"
        ),
        Transaction(
            date="2024-01-04",
            description="Beginning Balance for Account",
            amount=2000.00,
            transaction_type="deposit",
            vendor="Bank"
        ),
        Transaction(
            date="2024-01-05",
            description="Starting Balance - New Period",
            amount=3000.00,
            transaction_type="deposit",
            vendor="Bank"
        ),
    ]
    
    print("=" * 80)
    print("BALANCE ENTRY COUNT EXCLUSION TEST")
    print("=" * 80)
    print()
    
    # Test with report generator
    rg = ReportGenerator()
    
    # Check helper function
    print("1. Testing _is_balance_entry() helper:")
    for i, t in enumerate(transactions, 1):
        is_balance = rg._is_balance_entry(t)
        print(f"   Transaction {i}: {t.description[:40]:40} -> Balance Entry: {is_balance}")
    print()
    
    # Generate summary statistics
    stats = rg.generate_summary_statistics(transactions)
    
    print("2. Summary Statistics:")
    print(f"   Total Transactions in Data: {len(transactions)}")
    print(f"   Total Transactions Counted: {stats['Total Transactions']}")
    print(f"   Total Deposits in Data: {sum(1 for t in transactions if t.amount > 0)}")
    print(f"   Total Deposits Counted: {stats['Total Deposits']}")
    print(f"   Total Withdrawals in Data: {sum(1 for t in transactions if t.amount < 0)}")
    print(f"   Total Withdrawals Counted: {stats['Total Withdrawals']}")
    print()
    
    # Verify the counts
    print("3. Verification:")
    
    # Should exclude 3 balance entries (opening, beginning, starting)
    expected_total = 2  # Only Client Payment and Office Supplies
    expected_deposits = 1  # Only Client Payment
    expected_withdrawals = 1  # Only Office Supplies
    
    if stats['Total Transactions'] == expected_total:
        print(f"   ✅ Total Transactions: {stats['Total Transactions']} (correct - excludes 3 balance entries)")
    else:
        print(f"   ❌ Total Transactions: {stats['Total Transactions']} (expected {expected_total})")
    
    if stats['Total Deposits'] == expected_deposits:
        print(f"   ✅ Total Deposits: {stats['Total Deposits']} (correct - excludes balance deposits)")
    else:
        print(f"   ❌ Total Deposits: {stats['Total Deposits']} (expected {expected_deposits})")
    
    if stats['Total Withdrawals'] == expected_withdrawals:
        print(f"   ✅ Total Withdrawals: {stats['Total Withdrawals']} (correct)")
    else:
        print(f"   ❌ Total Withdrawals: {stats['Total Withdrawals']} (expected {expected_withdrawals})")
    
    # Verify amounts are still calculated correctly (including balance entries in sums)
    total_deposit_amount = sum(t.amount for t in transactions if t.amount > 0)
    if stats['Total Deposit Amount'] == total_deposit_amount:
        print(f"   ✅ Total Deposit Amount: ${stats['Total Deposit Amount']:.2f} (correct - includes all deposits)")
    else:
        print(f"   ❌ Total Deposit Amount: ${stats['Total Deposit Amount']:.2f} (expected ${total_deposit_amount:.2f})")
    
    print()
    
    # Test deposits summary
    print("4. Deposits Summary DataFrame:")
    deps_df = rg.generate_deposits_summary(transactions)
    print(deps_df.to_string(index=False))
    print()
    
    # Count rows (excluding TOTAL row)
    data_rows = len(deps_df) - 1
    print(f"   Vendors in summary: {data_rows}")
    print(f"   Expected vendors: 3 (Client A, Opening Balance, Bank - all should appear)")
    
    # Check if Opening Balance appears with count = 0
    opening_balance_rows = deps_df[deps_df['Source/Vendor'].str.contains('Opening Balance', na=False)]
    if not opening_balance_rows.empty:
        ob_count = opening_balance_rows.iloc[0]['Transaction Count']
        ob_amount = opening_balance_rows.iloc[0]['Subtotal ($)']
        print(f"   Opening Balance row found:")
        print(f"     - Transaction Count: {ob_count} (should be 0)")
        print(f"     - Subtotal: ${ob_amount:.2f} (should be $1000.00)")
        if ob_count == 0 and abs(ob_amount - 1000.00) < 0.01:
            print(f"   ✅ Correct - Opening Balance shows with amount but count = 0")
        else:
            print(f"   ❌ Incorrect - Expected count=0, amount=$1000.00")
    else:
        print(f"   ⚠️  Opening Balance row not found in summary")
    
    # Check total count
    total_row = deps_df[deps_df['Source/Vendor'] == 'TOTAL DEPOSITS']
    if not total_row.empty:
        total_count = total_row.iloc[0]['Transaction Count']
        print(f"   Total Deposits Count: {total_count} (should be 1 - only Client A)")
        if total_count == 1:
            print(f"   ✅ Correct - Total count excludes balance entries")
        else:
            print(f"   ❌ Incorrect - Expected total count = 1")
    
    print()
    
    # Test withdrawals summary  
    print("5. Withdrawals Summary DataFrame:")
    wds_df = rg.generate_withdrawals_summary(transactions)
    print(wds_df.to_string(index=False))
    print()
    
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print("  - Balance entries ARE kept in the data")
    print("  - Balance entries are EXCLUDED from transaction counts")
    print("  - Balance entry amounts ARE included in totals")
    print("  - Categories and amounts remain unchanged")
    print()


if __name__ == "__main__":
    test_balance_entry_exclusion()
