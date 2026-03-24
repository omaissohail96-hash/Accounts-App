#!/usr/bin/env python3
"""
Test to verify daily ending balance entries are properly excluded from transactions
"""
from bank_statement_parser import BankStatementParser

# Test cases for daily ending balance variations that should be SKIPPED
daily_balance_test_cases = [
    "Daily Ending Balance: 5,000.00",
    "12/15 Daily Ending Balance 5000.50",
    "Daily Balance 4500.00",
    "Balance Per Bank 10000.00",
    "Balance Per Books 9500.00",
    "Opening Balance $5000.00",
    "Starting Balance 4000.00",
    "Daily Ending 6500.50",
    "12/31 Daily Ending Balance $7,500.25",
    "Date Daily Ending 12/31/2024 8000.00",
]

# Test cases with regular transactions that should be INCLUDED
transaction_test_cases = [
    "12/15 Check 1234 Costco -150.50",
    "12/16 Deposit Stripe 2500.00",
    "12/17 Atm Withdrawal -200.00",
    "12/18 ACH Payment John Doe -1000.00",
]

print("=" * 80)
print("TESTING DAILY BALANCE ENTRIES (Should be Skipped by bank_statement_parser)")
print("=" * 80)

parser = BankStatementParser()
failed_daily_tests = []

for test in daily_balance_test_cases:
    # Determine transaction type - doesn't matter for this test since it should return None
    tx = parser.parse_transaction_line(test, 'deposit', 1)
    
    if tx is None:
        status = "✓ PASS"
        print(f"{status} | '{test}' -> SKIPPED (Good!)")
    else:
        status = "✗ FAIL"
        print(f"{status} | '{test}'")
        print(f"       ERROR: Should be skipped but got: {tx.description}")
        failed_daily_tests.append(test)

print("\n" + "=" * 80)
print("TESTING REGULAR TRANSACTIONS (Should be INCLUDED)")
print("=" * 80)

passed_transaction_tests = 0
for test in transaction_test_cases:
    # Determine transaction type based on amount sign
    tx = parser.parse_transaction_line(test, 'withdrawal', 1)
    
    if tx is not None and abs(tx.amount) > 0:
        status = "✓ PASS"
        print(f"{status} | '{test}' -> INCLUDED")
        print(f"       Description: {tx.description}, Amount: {tx.amount}")
        passed_transaction_tests += 1
    else:
        status = "✗ FAIL"
        print(f"{status} | '{test}'")
        print(f"       ERROR: Should be included but was skipped")

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

if failed_daily_tests:
    print(f"❌ FAILED: {len(failed_daily_tests)} daily balance test(s) failed")
    for test in failed_daily_tests:
        print(f"   - {test}")
elif passed_transaction_tests >= len(transaction_test_cases):
    print("✅ ALL TESTS PASSED")
    print(f"   ✓ {len(daily_balance_test_cases)} daily balance entries properly excluded")
    print(f"   ✓ {passed_transaction_tests} regular transactions properly included")
else:
    print(f"⚠️  WARNING: Only {passed_transaction_tests}/{len(transaction_test_cases)} transaction tests passed")
