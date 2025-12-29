"""
Test Script for Transaction Categorization Fixes
Tests that Shopify ID, ATM Withdrawal, and Check transactions are properly categorized
"""
from account_code_mapper import AccountCodeMapper
from bank_statement_parser import Transaction


def test_categorization_fixes():
    """Test all the categorization fixes"""
    mapper = AccountCodeMapper()
    
    print("=" * 80)
    print("TESTING TRANSACTION CATEGORIZATION FIXES")
    print("=" * 80)
    print()
    
    # Test cases covering the reported issues
    test_cases = [
        # Shopify transactions
        {
            "vendor": "Shopify",
            "description": "Shopify payout for sales",
            "is_income": True,
            "expected_code": "601",
            "expected_name": "SALES"
        },
        {
            "vendor": "Shopify Id",
            "description": "Payment from Shopify Id",
            "is_income": True,
            "expected_code": "601",
            "expected_name": "SALES"
        },
        {
            "vendor": "SHOPIFY ID",
            "description": "SHOPIFY ID PAYMENT RECEIVED",
            "is_income": True,
            "expected_code": "601",
            "expected_name": "SALES"
        },
        
        # ATM Withdrawal transactions
        {
            "vendor": "ATM",
            "description": "ATM Withdrawal",
            "is_income": False,
            "expected_code": "999",
            "expected_name": "OTHER EXPENSES"
        },
        {
            "vendor": "Bank ATM",
            "description": "Cash withdrawal at ATM",
            "is_income": False,
            "expected_code": "999",
            "expected_name": "OTHER EXPENSES"
        },
        
        # Check transactions
        {
            "vendor": "Check",
            "description": "Chk 1234",
            "is_income": False,
            "expected_code": "999",
            "expected_name": "OTHER EXPENSES"
        },
        {
            "vendor": "Payment",
            "description": "Check payment #5678",
            "is_income": False,
            "expected_code": "999",
            "expected_name": "OTHER EXPENSES"
        },
        {
            "vendor": "Bank",
            "description": "Chk 9876 Transaction",
            "is_income": False,
            "expected_code": "999",
            "expected_name": "OTHER EXPENSES"
        },
        
        # Existing mappings (should NOT break)
        {
            "vendor": "TikTok",
            "description": "TikTok Shop sales",
            "is_income": True,
            "expected_code": "601",
            "expected_name": "SALES"
        },
        {
            "vendor": "eBay",
            "description": "eBay marketplace fees",
            "is_income": False,
            "expected_code": "860",
            "expected_name": "BANK & EBAY CHARGES"
        },
        {
            "vendor": "Amazon",
            "description": "Amazon seller payment",
            "is_income": True,
            "expected_code": "601",
            "expected_name": "SALES"
        },
        {
            "vendor": "Wise",
            "description": "Wise international transfer",
            "is_income": False,
            "expected_code": "808",
            "expected_name": "OFFSHORE EXP"
        },
        {
            "vendor": "IRS",
            "description": "IRS payroll tax payment",
            "is_income": False,
            "expected_code": "821",
            "expected_name": "PAYROLL TAXES"
        },
        {
            "vendor": "Bank",
            "description": "Service fee charged",
            "is_income": False,
            "expected_code": "860",
            "expected_name": "BANK & EBAY CHARGES"
        },
        
        # Fallback test - unknown transaction
        {
            "vendor": "Unknown Vendor XYZ",
            "description": "Unknown transaction type",
            "is_income": False,
            "expected_code": "999",
            "expected_name": "OTHER EXPENSES"
        },
        {
            "vendor": "Random Income Source",
            "description": "Some random deposit",
            "is_income": True,
            "expected_code": "601",
            "expected_name": "SALES"
        },
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        vendor = test["vendor"]
        description = test["description"]
        is_income = test["is_income"]
        expected_code = test["expected_code"]
        expected_name = test["expected_name"]
        
        print(f"Test #{i}: {vendor} - {description}")
        print(f"  Type: {'INCOME' if is_income else 'EXPENSE'}")
        
        # Get account code
        account_code, account_name = mapper.get_account_code(vendor, description, is_income)
        
        # Check results
        if account_code == expected_code and account_name == expected_name:
            print(f"  ✅ PASS: Got {account_code} · {account_name}")
            passed += 1
        else:
            print(f"  ❌ FAIL: Expected {expected_code} · {expected_name}, Got {account_code} · {account_name}")
            failed += 1
        print()
    
    # Summary
    print("=" * 80)
    print(f"TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {len(test_cases)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Success Rate: {(passed / len(test_cases) * 100):.1f}%")
    print()
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! The categorization fixes are working correctly.")
        print()
        print("Key improvements:")
        print("  ✓ Shopify and Shopify ID transactions → 601 SALES")
        print("  ✓ ATM Withdrawal transactions → 999 OTHER EXPENSES")
        print("  ✓ Check transactions (Chk, Check payment) → 999 OTHER EXPENSES")
        print("  ✓ Case-insensitive and partial string matching enabled")
        print("  ✓ Safe fallback ensures NO transaction is left without an account code")
        print("  ✓ Existing mappings (TikTok, eBay, Amazon, Wise, IRS, Bank Fees) preserved")
        return True
    else:
        print("⚠️ SOME TESTS FAILED. Please review the failures above.")
        return False


if __name__ == "__main__":
    test_categorization_fixes()
