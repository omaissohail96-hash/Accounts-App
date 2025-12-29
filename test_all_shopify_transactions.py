"""
Test: Verify ALL Shopify transactions appear in P&L output
Tests that both Shopify deposits AND withdrawals are properly categorized
"""
from bank_statement_parser import Transaction
from schedule_c_categorizer import ScheduleCCategorizer
from account_code_mapper import AccountCodeMapper


def test_all_shopify_transactions():
    """Test that ALL Shopify transactions (deposits and withdrawals) are categorized"""
    print("=" * 80)
    print("TEST: ALL Shopify Transactions Must Appear in P&L")
    print("=" * 80)
    print()
    
    categorizer = ScheduleCCategorizer()
    mapper = AccountCodeMapper()
    
    test_cases = [
        # Shopify DEPOSITS (positive amounts)
        {
            "name": "Shopify deposit (regular)",
            "vendor": "Shopify",
            "description": "Shopify payout for sales",
            "amount": 1500.00,
            "type": "deposit",
            "expected_category": "GROSS",
            "expected_code": "601",
            "expected_type": "Income",
            "should_appear": True
        },
        {
            "name": "Shopify ID deposit",
            "vendor": "Shopify Id",
            "description": "Payment from Shopify Id",
            "amount": 2000.00,
            "type": "deposit",
            "expected_category": "GROSS",
            "expected_code": "601",
            "expected_type": "Income",
            "should_appear": True
        },
        
        # Shopify WITHDRAWALS (negative amounts - sales related)
        {
            "name": "Shopify withdrawal (sale reversal)",
            "vendor": "Shopify",
            "description": "Shopify sale payment reversal",
            "amount": -100.00,
            "type": "withdrawal",
            "expected_category": "GROSS",  # Still sales-related
            "expected_code": "601",
            "expected_type": "Income",  # 601 is always Income
            "should_appear": True
        },
        {
            "name": "Shopify refund",
            "vendor": "Shopify",
            "description": "Shopify refund processed",
            "amount": -50.00,
            "type": "withdrawal",
            "expected_category": "RETURNS",  # Refund keyword
            "expected_code": "602",
            "expected_type": "Income",  # 602 is always Income
            "should_appear": True
        },
        
        # Shopify FEES (negative amounts - expense related)
        {
            "name": "Shopify fee",
            "vendor": "Shopify",
            "description": "Shopify transaction fee",
            "amount": -25.00,
            "type": "withdrawal",
            "expected_category": None,  # Will be expense
            "expected_code": "860",  # Fee keyword triggers 860
            "expected_type": "Expense",
            "should_appear": True
        },
        {
            "name": "Shopify monthly charge",
            "vendor": "Shopify",
            "description": "Shopify monthly subscription charge",
            "amount": -29.00,
            "type": "withdrawal",
            "expected_category": None,  # Will be expense
            "expected_code": "860",  # Charge keyword triggers 860
            "expected_type": "Expense",
            "should_appear": True
        },
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        name = test["name"]
        vendor = test["vendor"]
        description = test["description"]
        amount = test["amount"]
        tx_type = test["type"]
        expected_category = test.get("expected_category")
        expected_code = test["expected_code"]
        expected_type = test["expected_type"]
        should_appear = test["should_appear"]
        
        print(f"Test #{i}: {name}")
        print(f"  Vendor: {vendor}")
        print(f"  Amount: ${amount:,.2f} ({tx_type})")
        print(f"  Description: {description}")
        
        # Create transaction
        tx = Transaction(
            date="2025-01-15",
            transaction_type=tx_type,
            vendor=vendor,
            amount=amount,
            description=description,
            raw_line=f"line{i}"
        )
        
        # Categorize with Schedule C
        category = categorizer.categorize_transaction(tx)
        
        print(f"  Schedule C Category: {category.tax_code}")
        print(f"  Is Excluded: {category.is_excluded}")
        
        # Get account code
        is_income_hint = amount > 0 or tx_type == "deposit"
        account_code, account_name = mapper.get_account_code(
            vendor, description, is_income_hint
        )
        
        # Determine final type based on account code
        actual_type = "Income" if account_code.startswith('6') else "Expense"
        
        print(f"  Account Code: {account_code} · {account_name}")
        print(f"  Type: {actual_type}")
        
        # Check if it would appear in output (not excluded)
        appears_in_output = not category.is_excluded
        
        # Verify
        code_match = account_code == expected_code
        type_match = actual_type == expected_type
        appears_match = appears_in_output == should_appear
        
        if expected_category:
            category_match = category.tax_code == expected_category
        else:
            category_match = True  # Don't check category for fees
        
        if code_match and type_match and appears_match and category_match:
            print(f"  ✅ PASS")
            passed += 1
        else:
            print(f"  ❌ FAIL")
            if not code_match:
                print(f"     Expected code {expected_code}, got {account_code}")
            if not type_match:
                print(f"     Expected type {expected_type}, got {actual_type}")
            if not appears_match:
                print(f"     Expected appears={should_appear}, got {appears_in_output}")
            if expected_category and not category_match:
                print(f"     Expected category {expected_category}, got {category.tax_code}")
            failed += 1
        print()
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total: {len(test_cases)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print()
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
        print()
        print("✓ ALL Shopify deposits appear in P&L as 601 SALES (Income)")
        print("✓ ALL Shopify withdrawals (sales-related) appear as 601 SALES or 602 RETURNS (Income)")
        print("✓ ALL Shopify fees appear as 860 BANK & EBAY CHARGES (Expense)")
        print("✓ NO Shopify transactions are excluded from P&L")
        print("✓ Type is correctly determined by account code (600s=Income, 800s=Expense)")
        return True
    else:
        print("⚠️ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = test_all_shopify_transactions()
    exit(0 if success else 1)
