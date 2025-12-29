#!/usr/bin/env python3
"""
Comprehensive Test for Check Transactions
Tests all variations of check-related wording to ensure they're categorized as 999 OTHER EXPENSES
"""

from schedule_c_categorizer import ScheduleCCategorizer
from account_code_mapper import AccountCodeMapper
from dataclasses import dataclass
from datetime import datetime

@dataclass
class MockTransaction:
    vendor: str
    description: str
    amount: float
    transaction_date: datetime
    transaction_type: str

categorizer = ScheduleCCategorizer()
mapper = AccountCodeMapper()

# Test all variations of check transactions
test_cases = [
    ("Chk ...7251 Transaction", "Check payment #7251 to Chk ...7251 Transaction - 12 16 Online Transfer To Chk ...7251 Transaction 7,609.34", -7609.34),
    ("Check", "Check #1234", -150.00),
    ("Check Payment", "Check payment for supplies", -250.00),
    ("Chk 5678", "Chk 5678 payment", -450.00),
    ("Bank", "Check transaction #9999", -100.00),
    ("Vendor ABC", "Payment via check number 7777", -350.00),
    ("Store XYZ", "Paid with Chk 8888", -275.00),
    ("Service Provider", "Check #5555 payment", -500.00),
    ("Contractor", "Check payment #6666 for services", -1200.00),
    ("Supplier", "Chk payment transaction", -800.00),
]

print("=" * 80)
print("COMPREHENSIVE CHECK TRANSACTION TEST")
print("=" * 80)
print()

all_passed = True
passed_count = 0
failed_count = 0

for vendor, description, amount in test_cases:
    txn = MockTransaction(
        vendor=vendor,
        description=description,
        amount=amount,
        transaction_date=datetime(2025, 12, 16),
        transaction_type='withdrawal'
    )
    
    # Test schedule_c_categorizer
    category_result = categorizer.categorize_transaction(txn)
    
    # Test account_code_mapper
    account_result = mapper.get_account_code(vendor=vendor, description=description, is_income=False)
    
    # Check if both correctly categorize as OTHER EXPENSES / 999
    schedule_c_correct = (
        category_result.tax_code == "OTHER_EXPENSES" and 
        not category_result.is_excluded
    )
    
    account_code_correct = account_result[0] == "999"
    
    test_passed = schedule_c_correct and account_code_correct
    
    if test_passed:
        print(f"✅ PASS: {vendor[:30]:30s} | {description[:45]:45s}")
        print(f"         Schedule C: {category_result.tax_code:20s} | Account: {account_result[0]} · {account_result[1]}")
        passed_count += 1
    else:
        print(f"❌ FAIL: {vendor[:30]:30s} | {description[:45]:45s}")
        print(f"         Schedule C: {category_result.tax_code:20s} (Expected: OTHER_EXPENSES)")
        print(f"         Account: {account_result[0]} · {account_result[1]} (Expected: 999 · OTHER EXPENSES)")
        print(f"         Is Excluded: {category_result.is_excluded} (Expected: False)")
        failed_count += 1
        all_passed = False
    print()

print("=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"Total Tests: {len(test_cases)}")
print(f"Passed: {passed_count} ✅")
print(f"Failed: {failed_count} ❌")
print(f"Success Rate: {(passed_count/len(test_cases)*100):.1f}%")
print()

if all_passed:
    print("🎉 ALL TESTS PASSED!")
    print()
    print("All check transaction variations are correctly categorized:")
    print("  ✓ Chk ...Transaction → 999 OTHER EXPENSES")
    print("  ✓ Check # → 999 OTHER EXPENSES")
    print("  ✓ Check payment → 999 OTHER EXPENSES")
    print("  ✓ Check transaction → 999 OTHER EXPENSES")
    print("  ✓ All appear in P&L reports")
else:
    print("❌ SOME TESTS FAILED - Please review the output above")
    exit(1)
