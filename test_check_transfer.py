#!/usr/bin/env python3
"""Test check payments with Transfer in description"""

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

# Test the exact transaction from screenshot
txn = MockTransaction(
    vendor='Chk ...7251 Transaction',
    description='Check payment #7251 to Chk ...7251 Transaction - 12 16 Online Transfer To Chk ...7251 Transaction 7,609.34',
    amount=-7609.34,
    transaction_date=datetime(2025, 12, 16),
    transaction_type='withdrawal'
)

print("Testing Check Payment with 'Transfer' in description")
print("="*70)
print(f"Vendor: {txn.vendor}")
print(f"Description: {txn.description}")
print(f"Amount: ${txn.amount:,.2f}")
print()

result = categorizer.categorize_transaction(txn)
print(f"Category: {result.category_name}")
print(f"Tax Code: {result.tax_code}")
print(f"Is Excluded: {result.is_excluded}")
print(f"Exclusion Reason: {result.exclusion_reason}")
print()

if not result.is_excluded:
    account = mapper.get_account_code(vendor=txn.vendor, description=txn.description, is_income=False)
    print(f"✅ Will appear in P&L")
    print(f"Account Code: {account[0]} · {account[1]}")
else:
    print("❌ EXCLUDED - Will NOT appear in P&L")
    print(f"Reason: {result.exclusion_reason}")
