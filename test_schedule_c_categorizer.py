"""
Test file for Schedule C Categorizer
"""
import sys
from bank_statement_parser import Transaction
from schedule_c_categorizer import ScheduleCCategorizer

def test_schedule_c_categorizer():
    """Test the Schedule C categorizer with sample transactions"""
    
    categorizer = ScheduleCCategorizer()
    
    # Test transactions
    test_transactions = [
        # Income - Gross receipts
        Transaction(
            date="2024-01-15",
            transaction_type="deposit",
            vendor="Shopify",
            amount=1500.00,
            description="Shopify payout for sales",
            raw_line=""
        ),
        # Income - Returns
        Transaction(
            date="2024-01-20",
            transaction_type="deposit",
            vendor="PayPal",
            amount=-50.00,
            description="Customer refund",
            raw_line=""
        ),
        # COGS - Inventory
        Transaction(
            date="2024-01-10",
            transaction_type="withdrawal",
            vendor="Wholesale Supplier",
            amount=-500.00,
            description="Inventory purchase - product stock",
            raw_line=""
        ),
        # Expense - Advertising
        Transaction(
            date="2024-01-12",
            transaction_type="withdrawal",
            vendor="Google Ads",
            amount=-200.00,
            description="Google Ads spend",
            raw_line=""
        ),
        # Expense - Contract labor
        Transaction(
            date="2024-01-14",
            transaction_type="withdrawal",
            vendor="Upwork",
            amount=-300.00,
            description="Freelance designer payment",
            raw_line=""
        ),
        # Expense - Utilities
        Transaction(
            date="2024-01-16",
            transaction_type="withdrawal",
            vendor="Comcast",
            amount=-80.00,
            description="Internet service",
            raw_line=""
        ),
        # Vehicle expense
        Transaction(
            date="2024-01-18",
            transaction_type="withdrawal",
            vendor="Shell",
            amount=-60.00,
            description="Gas for business vehicle",
            raw_line=""
        ),
        # Owner draw (ATM)
        Transaction(
            date="2024-01-19",
            transaction_type="withdrawal",
            vendor="ATM",
            amount=-200.00,
            description="ATM cash withdrawal",
            raw_line=""
        ),
        # Excluded - Personal
        Transaction(
            date="2024-01-20",
            transaction_type="withdrawal",
            vendor="Starbucks",
            amount=-5.00,
            description="Coffee - personal",
            raw_line=""
        ),
        # Excluded - Transfer
        Transaction(
            date="2024-01-21",
            transaction_type="withdrawal",
            vendor="Zelle",
            amount=-100.00,
            description="Transfer to personal account",
            raw_line=""
        ),
    ]
    
    # Categorize transactions
    categorized = categorizer.categorize_transactions(test_transactions)
    
    # Print results
    print("=" * 80)
    print("SCHEDULE C CATEGORIZATION TEST RESULTS")
    print("=" * 80)
    print()
    
    for transaction, category in categorized:
        print(f"Date: {transaction.date}")
        print(f"Vendor: {transaction.vendor}")
        print(f"Description: {transaction.description}")
        print(f"Amount: ${transaction.amount:,.2f}")
        print(f"Part: {category.part}")
        print(f"Line: {category.line_number}")
        print(f"Category: {category.category_name}")
        print(f"Tax Code: {category.tax_code}")
        if category.is_excluded:
            print(f"EXCLUDED: {category.exclusion_reason}")
        if category.is_owner_draw:
            print(f"OWNER DRAW: {category.exclusion_reason}")
        print("-" * 80)
    
    # Generate summary
    print()
    print("=" * 80)
    print("SCHEDULE C SUMMARY")
    print("=" * 80)
    summary = categorizer.generate_schedule_c_summary(categorized)
    print()
    
    # Generate report
    report = categorizer.generate_schedule_c_report(categorized)
    print(report)
    
    # Verify categorizations
    print()
    print("=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    expected_categories = {
        0: ("Part I", "GROSS"),  # Shopify - Income
        1: ("Part I", "RETURNS"),  # PayPal refund - Returns
        2: ("Part III", "COGS_INVENTORY"),  # Wholesale - COGS
        3: ("Part II", "ADVERTISING"),  # Google Ads - Advertising
        4: ("Part II", "CONTRACT_LABOR"),  # Upwork - Contract labor
        5: ("Part II", "UTILITIES"),  # Comcast - Utilities
        6: ("Part IV", "VEHICLE"),  # Shell - Vehicle
        7: ("Excluded", "OWNER_DRAW"),  # ATM - Owner draw
        8: ("Excluded", "PERSONAL"),  # Starbucks - Personal
        9: ("Excluded", "TRANSFER"),  # Zelle - Transfer
    }
    
    all_passed = True
    for idx, (transaction, category) in enumerate(categorized):
        expected_part, expected_code = expected_categories.get(idx, ("", ""))
        if category.part == expected_part and category.tax_code == expected_code:
            print(f"✓ Test {idx + 1}: {transaction.vendor} correctly categorized")
        else:
            print(f"✗ Test {idx + 1}: {transaction.vendor} - Expected ({expected_part}, {expected_code}), Got ({category.part}, {category.tax_code})")
            all_passed = False
    
    if all_passed:
        print()
        print("All tests passed!")
    else:
        print()
        print("Some tests failed. Please review.")
    
    return all_passed

if __name__ == "__main__":
    test_schedule_c_categorizer()



