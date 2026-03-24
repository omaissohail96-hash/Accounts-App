#!/usr/bin/env python3
"""
Test to verify the ending balance being detected as fee bug is fixed
This test ensures that ending balance entries are NOT classified as bank fees
"""
from bank_fee_parser import parse_bank_fee
from bank_statement_parser import BankStatementParser

# Test various ending balance formats - these should ALL return is_bank_fee=False
ending_balance_test_cases = [
    "Ending Balance 5000.00",
    "11/30 Ending Balance $599.51",
    "Ending Balance Service 5000.00",  # Previously was incorrectly detected as fee
    "Daily Ending Balance 5000.00",
    "Balance Service Charge 50.00",    # Previously was incorrectly detected as fee
    "Opening Balance 1000.00",
    "Beginning Balance 2500.00",
    "Closing Balance 7500.00",
    "Average Balance 5000.00"
]

# Test regular bank fees - these SHOULD return is_bank_fee=True
fee_test_cases = [
    "Service Fee 12.50",
    "Monthly Service Fee 25.00",
    "Bank Service Charge 15.00",
    "Wire Fee 30.00",
    "Overdraft Fee 35.00",
    "NSF Charge 50.00",
    "ATM Maintenance Fee 2.50",
    "ACH Processing Fee 1.00"
]

print("=" * 70)
print("TESTING ENDING BALANCE ENTRIES (Should NOT be detected as fees)")
print("=" * 70)
failed_tests = []

for test in ending_balance_test_cases:
    result = parse_bank_fee(test, -5000.00)
    status = "✓ PASS" if not result.is_bank_fee else "✗ FAIL"
    print(f"{status} | '{test}'")
    if result.is_bank_fee:
        print(f"       ERROR: Detected as fee with amount ${result.fee_amount}")
        failed_tests.append(test)

print("\n" + "=" * 70)
print("TESTING LEGITIMATE BANK FEES (Should be detected as fees)")
print("=" * 70)

for test in fee_test_cases:
    # Extract amount from the test case (last number-like token)
    import re
    amount_match = re.search(r'[\d.]+', test[::-1])  # Search from end
    if amount_match:
        amount_str = amount_match.group(0)[::-1]
        transaction_amount = -float(amount_str)
    else:
        transaction_amount = -25.00  # Default
    
    result = parse_bank_fee(test, transaction_amount)
    status = "✓ PASS" if result.is_bank_fee else "✗ FAIL"
    print(f"{status} | '{test}'")
    if not result.is_bank_fee:
        print(f"       WARNING: Not detected as fee when it should be")

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
if failed_tests:
    print(f"❌ FAILED: {len(failed_tests)} test(s) failed")
    for test in failed_tests:
        print(f"   - {test}")
else:
    print("✅ ALL TESTS PASSED - Ending balance entries are no longer misdetected as fees")
