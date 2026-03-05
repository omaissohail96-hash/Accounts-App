"""
Test to demonstrate the date behavior change
"""
from datetime import datetime, date

# Simulate parsed transaction dates
min_date = date(2024, 10, 15)  # Oct 15, 2024 (first transaction)
max_date = date(2024, 12, 20)  # Dec 20, 2024 (last transaction)

print("=" * 80)
print("STATEMENT DATE EXTRACTION - BEFORE vs AFTER")
print("=" * 80)
print()
print(f"Actual Statement Dates (from transactions):")
print(f"  First transaction: {min_date.strftime('%b %d, %Y')}")
print(f"  Last transaction:  {max_date.strftime('%b %d, %Y')}")
print()

# OLD BEHAVIOR (month-extended)
from calendar import monthrange
old_start = min_date.replace(day=1)
old_end = max_date.replace(day=monthrange(max_date.year, max_date.month)[1])

print("BEFORE (Month-Extended):")
print("=" * 80)
print(f"  Filter Start Date: {old_start.strftime('%b %d, %Y')} (extended to 1st of month)")
print(f"  Filter End Date:   {old_end.strftime('%b %d, %Y')} (extended to last day of month)")
print(f"  Period for Account Codes: {old_start.strftime('%b %d, %Y')} - {old_end.strftime('%b %d, %Y')}")
print(f"  Coverage: {(old_end - old_start).days} days")
print()

# NEW BEHAVIOR (exact dates)
new_start = min_date
new_end = max_date

print("AFTER (Exact Dates):")
print("=" * 80)
print(f"  Filter Start Date: {new_start.strftime('%b %d, %Y')} (exact from statement)")
print(f"  Filter End Date:   {new_end.strftime('%b %d, %Y')} (exact from statement)")
print(f"  Period for Account Codes: {new_start.strftime('%b %d, %Y')} - {new_end.strftime('%b %d, %Y')}")
print(f"  Coverage: {(new_end - new_start).days} days")
print()

print("=" * 80)
print("BENEFITS:")
print("=" * 80)
print("✅ Filter shows EXACT dates from bank statement header")
print("✅ Account code exports use EXACT statement period dates")
print("✅ No artificial date extension to full months")
print("✅ Matches the actual statement coverage period precisely")
print()

print("🎯 EXPECTED IN STREAMLIT:")
print("   - Filter defaults will show Oct 15, 2024 → Dec 20, 2024")
print("   - Not Oct 01, 2024 → Dec 31, 2024")
print("   - Account code filenames will use exact dates")
print()
print("💡 REFRESH YOUR STREAMLIT APP NOW (F5 or Ctrl+R)")
