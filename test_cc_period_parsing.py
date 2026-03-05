"""
Test Credit Card Period Date Parsing
"""
import sys
sys.path.insert(0, '/Users/ibrahimsohail/Accounts-App')

from bank_data_analysis import parse_cc_period_dates

print("=" * 80)
print("CREDIT CARD PERIOD DATE PARSING TEST")
print("=" * 80)
print()

# Test various date formats
test_cases = [
    ("03/14/24 - 05/13/24", "2024-03-14", "2024-05-13"),
    ("3/14/24 – 5/13/24", "2024-03-14", "2024-05-13"),
    ("03/14/2024 to 05/13/2024", "2024-03-14", "2024-05-13"),
    ("12/01/23 - 01/15/24", "2023-12-01", "2024-01-15"),
    ("N/A", None, None),
    ("", None, None),
]

for date_str, expected_start, expected_end in test_cases:
    result = parse_cc_period_dates(date_str)
    
    if result:
        start_date, end_date = result
        actual_start = start_date.strftime("%Y-%m-%d")
        actual_end = end_date.strftime("%Y-%m-%d")
        
        if actual_start == expected_start and actual_end == expected_end:
            print(f"✅ PASS: '{date_str}'")
            print(f"   Result: {start_date.strftime('%m/%d/%y')} to {end_date.strftime('%m/%d/%y')}")
        else:
            print(f"❌ FAIL: '{date_str}'")
            print(f"   Expected: {expected_start} to {expected_end}")
            print(f"   Got: {actual_start} to {actual_end}")
    else:
        if expected_start is None:
            print(f"✅ PASS: '{date_str}' → None (expected)")
        else:
            print(f"❌ FAIL: '{date_str}' → None (expected dates)")
    print()

print("=" * 80)
print("EXPECTED BEHAVIOR IN APP:")
print("=" * 80)
print("1. Upload credit cards with year selection 2024")
print("2. System extracts 'Opening/Closing Date: 03/14/24 - 05/13/24' from summary")
print("3. Parses to: start=2024-03-14, end=2024-05-13")
print("4. Filter shows: 2024/03/14 to 2024/05/13")
print("5. ✅ Matches the credit card summary exactly!")
