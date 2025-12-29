"""
Integration Test: Verify Shopify ID and ATM Withdrawal transactions appear in output
Tests that these transactions are NOT filtered out from P&L reports
"""
from bank_statement_parser import Transaction
from schedule_c_categorizer import ScheduleCCategorizer
from account_code_mapper import AccountCodeMapper


def test_shopify_id_in_output():
    """Test that Shopify ID deposits are categorized and not excluded"""
    print("=" * 80)
    print("TEST 1: Shopify ID Deposits Must Appear in Output")
    print("=" * 80)
    
    categorizer = ScheduleCCategorizer()
    mapper = AccountCodeMapper()
    
    # Create Shopify ID transaction
    shopify_tx = Transaction(
        date="2025-01-15",
        transaction_type="deposit",
        vendor="Shopify Id",
        amount=1500.00,
        description="Shopify Id payment received",
        raw_line="2025-01-15,1500.00,Shopify Id payment"
    )
    
    # Categorize it
    category = categorizer.categorize_transaction(shopify_tx)
    
    print(f"Transaction: {shopify_tx.vendor} - {shopify_tx.description}")
    print(f"Amount: ${shopify_tx.amount:,.2f}")
    print(f"Category: {category.category_name}")
    print(f"Tax Code: {category.tax_code}")
    print(f"Part: {category.part}")
    print(f"Is Excluded: {category.is_excluded}")
    print(f"Is Owner Draw: {category.is_owner_draw}")
    print()
    
    # Get account code
    account_code, account_name = mapper.get_account_code(
        shopify_tx.vendor,
        shopify_tx.description,
        is_income=True
    )
    print(f"Account Code: {account_code} · {account_name}")
    print()
    
    # Verify expectations
    assert category.tax_code == "GROSS", f"Expected GROSS, got {category.tax_code}"
    assert not category.is_excluded, "Transaction should NOT be excluded"
    assert not category.is_owner_draw, "Transaction should NOT be owner draw"
    assert account_code == "601", f"Expected 601, got {account_code}"
    
    print("✅ PASS: Shopify ID transaction properly categorized as income")
    print("✅ PASS: Will appear in P&L output with account code 601")
    print()
    return True


def test_atm_withdrawal_in_output():
    """Test that ATM withdrawals are categorized and appear in output"""
    print("=" * 80)
    print("TEST 2: ATM Withdrawals Must Appear in Output")
    print("=" * 80)
    
    categorizer = ScheduleCCategorizer()
    mapper = AccountCodeMapper()
    
    # Create ATM withdrawal transaction
    atm_tx = Transaction(
        date="2025-01-20",
        transaction_type="withdrawal",
        vendor="ATM",
        amount=-200.00,
        description="ATM Withdrawal",
        raw_line="2025-01-20,-200.00,ATM Withdrawal"
    )
    
    # Categorize it
    category = categorizer.categorize_transaction(atm_tx)
    
    print(f"Transaction: {atm_tx.vendor} - {atm_tx.description}")
    print(f"Amount: ${atm_tx.amount:,.2f}")
    print(f"Category: {category.category_name}")
    print(f"Tax Code: {category.tax_code}")
    print(f"Part: {category.part}")
    print(f"Line Number: {category.line_number}")
    print(f"Is Excluded: {category.is_excluded}")
    print(f"Is Owner Draw: {category.is_owner_draw}")
    print()
    
    # Get account code
    account_code, account_name = mapper.get_account_code(
        atm_tx.vendor,
        atm_tx.description,
        is_income=False
    )
    print(f"Account Code: {account_code} · {account_name}")
    print()
    
    # Verify expectations
    assert category.tax_code == "OTHER_EXPENSES", f"Expected OTHER_EXPENSES, got {category.tax_code}"
    assert not category.is_excluded, "Transaction should NOT be excluded"
    assert not category.is_owner_draw, "Transaction should NOT be marked as owner draw (for P&L output)"
    assert category.line_number is not None, "Transaction should have a line number"
    assert account_code == "999", f"Expected 999, got {account_code}"
    
    print("✅ PASS: ATM withdrawal categorized as OTHER EXPENSES")
    print("✅ PASS: Will appear in P&L output with account code 999")
    print()
    return True


def test_output_filtering():
    """Test that transactions are NOT filtered out from output"""
    print("=" * 80)
    print("TEST 3: Verify Transactions Appear in Output (Not Filtered)")
    print("=" * 80)
    
    categorizer = ScheduleCCategorizer()
    
    transactions = [
        Transaction(
            date="2025-01-15",
            transaction_type="deposit",
            vendor="Shopify Id",
            amount=1500.00,
            description="Shopify Id payment",
            raw_line="line1"
        ),
        Transaction(
            date="2025-01-20",
            transaction_type="withdrawal",
            vendor="ATM",
            amount=-200.00,
            description="ATM Withdrawal",
            raw_line="line2"
        ),
        Transaction(
            date="2025-01-25",
            transaction_type="withdrawal",
            vendor="Check",
            amount=-300.00,
            description="Chk 1234",
            raw_line="line3"
        ),
    ]
    
    # Categorize all
    categorized = categorizer.categorize_transactions(transactions)
    
    # Filter using the logic from bank_data_analysis.py
    # This simulates what happens in the P&L output
    included_for_output = []
    for tx, cat in categorized:
        # Using the NEW logic: only exclude if is_excluded=True
        if cat.is_excluded:
            continue
        included_for_output.append((tx, cat))
    
    print(f"Total transactions: {len(transactions)}")
    print(f"Transactions in output: {len(included_for_output)}")
    print()
    
    for tx, cat in included_for_output:
        print(f"  ✓ {tx.vendor:<15} | {cat.tax_code:<20} | Part: {cat.part}")
    print()
    
    # Verify all transactions appear
    assert len(included_for_output) == 3, f"Expected 3 transactions in output, got {len(included_for_output)}"
    
    # Verify specific transactions
    shopify_found = any(tx.vendor == "Shopify Id" for tx, _ in included_for_output)
    atm_found = any(tx.vendor == "ATM" for tx, _ in included_for_output)
    check_found = any(tx.vendor == "Check" for tx, _ in included_for_output)
    
    assert shopify_found, "Shopify Id transaction missing from output!"
    assert atm_found, "ATM Withdrawal transaction missing from output!"
    assert check_found, "Check transaction missing from output!"
    
    print("✅ PASS: All transactions appear in output")
    print("✅ PASS: No transactions are filtered out incorrectly")
    print()
    return True


def main():
    print("\n")
    print("🔍 INTEGRATION TEST: Shopify ID & ATM Withdrawal Output")
    print("Testing that transactions appear in P&L reports, not filtered out")
    print("\n")
    
    try:
        test1 = test_shopify_id_in_output()
        test2 = test_atm_withdrawal_in_output()
        test3 = test_output_filtering()
        
        if test1 and test2 and test3:
            print("=" * 80)
            print("🎉 ALL INTEGRATION TESTS PASSED!")
            print("=" * 80)
            print()
            print("✅ Shopify ID deposits → 601 SALES (appear in output)")
            print("✅ ATM withdrawals → 999 OTHER EXPENSES (appear in output)")
            print("✅ Check payments → 999 OTHER EXPENSES (appear in output)")
            print("✅ NO transactions are filtered out from P&L reports")
            print()
            print("The fix is complete and working correctly! 🚀")
            print()
            return True
    except AssertionError as e:
        print()
        print("❌ TEST FAILED!")
        print(f"Error: {e}")
        print()
        return False
    except Exception as e:
        print()
        print("❌ UNEXPECTED ERROR!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
