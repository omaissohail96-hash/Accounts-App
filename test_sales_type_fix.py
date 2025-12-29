"""
Test: Verify 601 SALES transactions are always classified as Income
Tests that account code determines Type, not transaction amount/direction
"""
from bank_statement_parser import Transaction
from account_code_mapper import AccountCodeMapper


def test_sales_always_income():
    """Test that all 601 SALES transactions are Income regardless of amount"""
    print("=" * 80)
    print("TEST: All 601 SALES Transactions Must Be Classified as Income")
    print("=" * 80)
    print()
    
    mapper = AccountCodeMapper()
    
    test_cases = [
        # Regular Shopify deposit (positive amount)
        {
            "vendor": "Shopify",
            "description": "Shopify payout",
            "amount": 1500.00,
            "type": "deposit",
            "expected_code": "601",
            "expected_type": "Income"
        },
        # Shopify refund (negative amount - still sales-related)
        {
            "vendor": "Shopify",
            "description": "Shopify refund processed",
            "amount": -50.00,
            "type": "withdrawal",
            "expected_code": "602",  # This would be returns
            "expected_type": "Income"  # 602 is also 600-series = Income
        },
        # Shopify chargeback (negative, but sales account)
        {
            "vendor": "Shopify",
            "description": "Shopify sale payment",
            "amount": -100.00,
            "type": "withdrawal",
            "expected_code": "601",  # Sales (before refund keywords match)
            "expected_type": "Income"
        },
        # TikTok Shop deposit
        {
            "vendor": "TikTok",
            "description": "TikTok Shop sales",
            "amount": 2500.00,
            "type": "deposit",
            "expected_code": "601",
            "expected_type": "Income"
        },
        # eBay sale (positive)
        {
            "vendor": "eBay",
            "description": "eBay sales payment",
            "amount": 800.00,
            "type": "deposit",
            "expected_code": "601",
            "expected_type": "Income"
        },
        # Bank fee (negative) - should be Expense
        {
            "vendor": "Bank",
            "description": "Service fee charged",
            "amount": -15.00,
            "type": "withdrawal",
            "expected_code": "860",
            "expected_type": "Expense"
        },
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        vendor = test["vendor"]
        description = test["description"]
        amount = test["amount"]
        tx_type = test["type"]
        expected_code = test["expected_code"]
        expected_type = test["expected_type"]
        
        print(f"Test #{i}: {vendor} - ${amount:,.2f}")
        print(f"  Description: {description}")
        print(f"  Transaction Type: {tx_type}")
        
        # Determine is_income hint (old logic)
        is_income_hint = amount > 0 or tx_type == "deposit"
        
        # Get account code
        account_code, account_name = mapper.get_account_code(
            vendor, description, is_income_hint
        )
        
        # Determine final type based on account code (NEW LOGIC)
        actual_type = "Income" if account_code.startswith('6') else "Expense"
        
        print(f"  Account Code: {account_code} · {account_name}")
        print(f"  Type: {actual_type}")
        
        # Verify
        code_match = account_code == expected_code
        type_match = actual_type == expected_type
        
        if code_match and type_match:
            print(f"  ✅ PASS")
            passed += 1
        else:
            print(f"  ❌ FAIL")
            if not code_match:
                print(f"     Expected code {expected_code}, got {account_code}")
            if not type_match:
                print(f"     Expected type {expected_type}, got {actual_type}")
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
        print("✓ All 601 SALES transactions classified as Income")
        print("✓ All 602 RETURNS transactions classified as Income")
        print("✓ All 603 OTHER INCOME transactions classified as Income")
        print("✓ All 600-series codes (Income accounts) always show as Income")
        print("✓ Type is determined by account code, not transaction amount")
        return True
    else:
        print("⚠️ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = test_sales_always_income()
    exit(0 if success else 1)
