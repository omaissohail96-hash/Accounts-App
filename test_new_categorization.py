"""
Test script for the new keyword-based categorization logic
"""
from account_code_mapper import AccountCodeMapper

def test_categorization():
    """Test the new categorization logic with various scenarios"""
    mapper = AccountCodeMapper()
    
    print("=" * 80)
    print("TESTING NEW CATEGORIZATION LOGIC")
    print("=" * 80)
    
    # Test cases: (vendor, description, expected_code, expected_name)
    test_cases = [
        # Test 1: RETURNS & ALLOWANCES (rank 1 - highest priority)
        ("Stripe", "Customer refund processed", "602", "RETURNS & ALLOWANCES"),
        
        # Test 2: SALES (rank 2) - but should be excluded if "refund" appears
        ("PayPal", "Payment received from customer", "601", "SALES"),
        
        # Test 3: SALES with refund keyword - should match RETURNS (rank 1) instead
        ("Shopify", "Sale payment refund", "602", "RETURNS & ALLOWANCES"),
        
        # Test 4: SHIPPING
        ("FedEx", "Shipping charge", "702", "SHIPPING SUPPLIES"),
        
        # Test 5: PAYROLL TAXES
        ("IRS", "Federal tax payment", "821", "PAYROLL TAXES"),
        
        # Test 6: SALARIES-OFFICERS (has exclude for "tax")
        ("ADP", "Payroll processing", "801", "SALARIES-OFFICERS"),
        
        # Test 7: SALARIES excluded due to "tax" keyword
        ("Gusto", "Payroll tax withholding", "821", "PAYROLL TAXES"),
        
        # Test 8: ADVERTISING
        ("Google", "Google Ads campaign", "854", "ADVERTISEMENT"),
        
        # Test 9: BANK FEES
        ("Bank of America", "Service fee charged", "860", "BANK & EBAY CHARGES"),
        
        # Test 10: RENT
        ("WeWork", "Office rent payment", "928", "RENT"),
        
        # Test 11: No match - should return None
        ("Unknown Vendor", "Random transaction", None, None),
    ]
    
    print("\nRunning test cases...\n")
    
    passed = 0
    failed = 0
    
    for i, (vendor, description, expected_code, expected_name) in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")
        print(f"Input: vendor='{vendor}', description='{description}'")
        
        result = mapper._match_by_keywords(vendor, description)
        
        if result:
            actual_code, actual_name = result
            print(f"Expected: {expected_code} · {expected_name}")
            print(f"Actual:   {actual_code} · {actual_name}")
            
            if actual_code == expected_code and actual_name == expected_name:
                print("✅ PASS")
                passed += 1
            else:
                print("❌ FAIL")
                failed += 1
        else:
            print(f"Expected: {expected_code} · {expected_name}")
            print(f"Actual:   No match (None)")
            
            if expected_code is None:
                print("✅ PASS")
                passed += 1
            else:
                print("❌ FAIL")
                failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)
    
    return passed, failed

if __name__ == "__main__":
    test_categorization()
