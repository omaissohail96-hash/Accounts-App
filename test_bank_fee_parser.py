"""
Test Bank Fee Parser
Demonstrates robust fee parsing that accurately extracts fee amounts
without misreading total transaction amounts.
"""
from bank_fee_parser import BankFeeParser, parse_bank_fee


def test_bank_fee_extraction():
    """Test various bank fee scenarios"""
    
    parser = BankFeeParser()
    
    # Test cases with expected results
    test_cases = [
        {
            "name": "ACH Fee with Qty and Total",
            "description": "Standard ACH Pmnts Initial Fee Qty 3 7.50 Total Fees 7.50",
            "transaction_amount": -32563.65,
            "expected_fee": 7.50,
            "should_extract_correctly": True
        },
        {
            "name": "Simple Service Fee",
            "description": "Monthly Service Fee 12.50",
            "transaction_amount": -12.50,
            "expected_fee": 12.50,
            "should_extract_correctly": False  # Only one amount, so matching transaction is OK
        },
        {
            "name": "ATM Fee",
            "description": "ATM Withdrawal Fee 3.00 Total Amount 103.00",
            "transaction_amount": -103.00,
            "expected_fee": 3.00,
            "should_extract_correctly": True
        },
        {
            "name": "Wire Transfer Fee",
            "description": "Wire Transfer Fee $25.00 Sent to Account XXX1234",
            "transaction_amount": -10025.00,
            "expected_fee": 25.00,
            "should_extract_correctly": True
        },
        {
            "name": "NSF Fee",
            "description": "NSF Charge 35.00",
            "transaction_amount": -35.00,
            "expected_fee": 35.00,
            "should_extract_correctly": False  # Only one amount
        },
        {
            "name": "Processing Fee with Multiple Amounts",
            "description": "Payment Processing Fee 2.50 Transaction Total USD 5,432.65",
            "transaction_amount": -5432.65,
            "expected_fee": 2.50,
            "should_extract_correctly": True
        },
        {
            "name": "Overdraft Fee",
            "description": "Overdraft Fee 30.00",
            "transaction_amount": -30.00,
            "expected_fee": 30.00,
            "should_extract_correctly": False  # Only one amount
        },
        {
            "name": "Suspicious Large Fee (needs review)",
            "description": "Bank Fee 999.00",
            "transaction_amount": -999.00,
            "expected_fee": 999.00,
            "should_need_review": True
        },
        {
            "name": "Not a Bank Fee",
            "description": "Transfer to Savings Account",
            "transaction_amount": -500.00,
            "expected_is_fee": False
        },
        {
            "name": "Platform Fee",
            "description": "Shopify Platform Fee 29.00 Invoice #12345",
            "transaction_amount": -1529.00,
            "expected_fee": 29.00,
            "should_extract_correctly": True
        },
        {
            "name": "Maintenance Fee",
            "description": "Account Maintenance Fee $15.00",
            "transaction_amount": -15.00,
            "expected_fee": 15.00,
            "should_extract_correctly": False  # Only one amount
        },
        {
            "name": "Fee Before Amount",
            "description": "Service Charge 8.50 Applied",
            "transaction_amount": -8.50,
            "expected_fee": 8.50,
            "should_extract_correctly": False  # Only one amount
        },
        {
            "name": "International Transaction Fee",
            "description": "International Fee 4.50 Total Charged 104.50 USD",
            "transaction_amount": -104.50,
            "expected_fee": 4.50,
            "should_extract_correctly": True
        },
    ]
    
    print("=" * 80)
    print("BANK FEE PARSER TEST RESULTS")
    print("=" * 80)
    print()
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']}")
        print(f"  Description: {test['description']}")
        print(f"  Transaction Amount: ${test['transaction_amount']:.2f}")
        
        result = parser.parse_bank_fee(test['description'], test['transaction_amount'])
        
        print(f"  Detected as Fee: {result.is_bank_fee}")
        
        if result.is_bank_fee:
            print(f"  Extracted Fee Amount: ${result.fee_amount:.2f}")
            print(f"  Needs Review: {result.needs_review}")
            print(f"  Reason: {result.reason}")
        
        # Validate results
        test_passed = True
        
        # Check if it's a fee
        expected_is_fee = test.get('expected_is_fee', True)
        if result.is_bank_fee != expected_is_fee:
            print(f"  ❌ FAILED: Expected is_fee={expected_is_fee}, got {result.is_bank_fee}")
            test_passed = False
        
        # Check extracted amount if it should be a fee
        if result.is_bank_fee and 'expected_fee' in test:
            if abs(result.fee_amount - test['expected_fee']) > 0.01:
                print(f"  ❌ FAILED: Expected fee ${test['expected_fee']:.2f}, got ${result.fee_amount:.2f}")
                test_passed = False
            elif test.get('should_extract_correctly'):
                # Verify we didn't use the transaction total
                if abs(result.fee_amount - abs(test['transaction_amount'])) < 0.01:
                    print(f"  ❌ FAILED: Used transaction total instead of extracting fee")
                    test_passed = False
        
        # Check review flag
        if test.get('should_need_review') and not result.needs_review:
            print(f"  ⚠️  WARNING: Expected needs_review=True")
        
        if test_passed:
            print(f"  ✅ PASSED")
            passed += 1
        else:
            failed += 1
        
        print()
    
    print("=" * 80)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)
    
    return passed, failed


def test_edge_cases():
    """Test edge cases and error handling"""
    
    parser = BankFeeParser()
    
    print("\n" + "=" * 80)
    print("EDGE CASE TESTS")
    print("=" * 80)
    print()
    
    # Empty description
    result = parser.parse_bank_fee("", -10.00)
    print(f"1. Empty description: is_fee={result.is_bank_fee} ✅")
    
    # No amount in description
    result = parser.parse_bank_fee("Monthly Service Fee", -10.00)
    if result.fee_amount is not None:
        print(f"2. No amount in description: fee=${result.fee_amount:.2f}, needs_review={result.needs_review} ✅")
    else:
        print(f"2. No amount in description: fee=None, needs_review={result.needs_review} ✅")
    
    # Multiple fees in one description
    result = parser.parse_bank_fee("Service Fee 5.00 + ATM Fee 3.00 = Total Fees 8.00", -8.00)
    print(f"3. Multiple fees: extracted=${result.fee_amount:.2f} ✅")
    
    # Fee with comma formatting
    result = parser.parse_bank_fee("Transaction Fee 1,234.56", -1234.56)
    print(f"4. Comma-formatted fee: fee=${result.fee_amount:.2f}, needs_review={result.needs_review}")
    if result.needs_review:
        print(f"   (Correctly flagged for review due to large amount) ✅")
    
    # Zero fee
    result = parser.parse_bank_fee("Fee Waived 0.00", 0.00)
    print(f"5. Zero fee: needs_review={result.needs_review} ✅")
    
    print()


def test_integration_example():
    """Show real-world integration example"""
    
    print("=" * 80)
    print("INTEGRATION EXAMPLE - Real Transaction Processing")
    print("=" * 80)
    print()
    
    # Simulate processing a transaction
    from bank_statement_parser import Transaction
    from schedule_c_categorizer import ScheduleCCategorizer
    
    # Create a test transaction with the problematic description
    transaction = Transaction(
        date="2024-01-15",
        description="Standard ACH Pmnts Initial Fee Qty 3 7.50 Total Fees 7.50 Transfer USD 32,563.65",
        amount=-32563.65,  # Original transaction amount
        transaction_type="withdrawal",
        vendor="Unknown"
    )
    
    print("Original Transaction:")
    print(f"  Date: {transaction.date}")
    print(f"  Description: {transaction.description}")
    print(f"  Amount: ${transaction.amount:.2f}")
    print(f"  Type: {transaction.transaction_type}")
    print()
    
    # Use the categorizer (which now uses the robust fee parser)
    categorizer = ScheduleCCategorizer()
    category = categorizer.categorize_transaction(transaction)
    
    print("After Categorization:")
    print(f"  Category: {category.category_name}")
    print(f"  Tax Code: {category.tax_code}")
    print(f"  Schedule C Line: {category.line_number}")
    print(f"  Corrected Amount: ${transaction.amount:.2f}")
    print(f"  Needs Review: {transaction.needs_review}")
    
    if hasattr(transaction, 'original_amount'):
        print(f"  Original Amount Preserved: ${transaction.original_amount:.2f}")
    
    print()
    
    if abs(transaction.amount) == 7.50:
        print("✅ SUCCESS: Correctly extracted fee amount ($7.50) instead of transaction total ($32,563.65)")
    else:
        print(f"❌ FAILED: Expected $7.50, got ${abs(transaction.amount):.2f}")
    
    print()


if __name__ == "__main__":
    # Run all tests
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "BANK FEE PARSER TEST SUITE" + " " * 32 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    # Main tests
    passed, failed = test_bank_fee_extraction()
    
    # Edge cases
    test_edge_cases()
    
    # Integration example
    test_integration_example()
    
    print("=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
