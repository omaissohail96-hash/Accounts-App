"""
Test Credit Card Period Date Parsing (Standalone)
"""
from datetime import datetime, date
from typing import Optional, Tuple
import re

def parse_cc_period_dates(date_string: str) -> Optional[Tuple[date, date]]:
    """
    Parse credit card Opening/Closing Date string to extract start and end dates.
    
    Handles formats like:
    - "03/14/24 - 05/13/24"
    - "03/14/2024 to 05/13/2024"
    - "3/14/24 – 5/13/24"
    
    Returns tuple of (start_date, end_date) or None if parsing fails.
    """
    if not date_string or date_string == "N/A":
        return None
    
    # Pattern: MM/DD/YY or MM/DD/YYYY with various separators
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4})\s*(?:[-–to]+)\s*(\d{1,2}/\d{1,2}/\d{2,4})'
    match = re.search(pattern, date_string, re.I)
    
    if match:
        start_str = match.group(1)
        end_str = match.group(2)
        
        try:
            # Try parsing with different formats
            for fmt in ['%m/%d/%Y', '%m/%d/%y']:
                try:
                    start_date = datetime.strptime(start_str, fmt).date()
                    end_date = datetime.strptime(end_str, fmt).date()
                    return (start_date, end_date)
                except ValueError:
                    continue
        except Exception:
            pass
    
    return None

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

passed = 0
failed = 0

for date_str, expected_start, expected_end in test_cases:
    result = parse_cc_period_dates(date_str)
    
    if result:
        start_date, end_date = result
        actual_start = start_date.strftime("%Y-%m-%d")
        actual_end = end_date.strftime("%Y-%m-%d")
        
        if actual_start == expected_start and actual_end == expected_end:
            print(f"✅ PASS: '{date_str}'")
            print(f"   Result: {start_date.strftime('%m/%d/%y')} to {end_date.strftime('%m/%d/%y')}")
            passed += 1
        else:
            print(f"❌ FAIL: '{date_str}'")
            print(f"   Expected: {expected_start} to {expected_end}")
            print(f"   Got: {actual_start} to {actual_end}")
            failed += 1
    else:
        if expected_start is None:
            print(f"✅ PASS: '{date_str}' → None (expected)")
            passed += 1
        else:
            print(f"❌ FAIL: '{date_str}' → None (expected dates)")
            failed += 1
    print()

print("=" * 80)
print(f"TEST RESULTS: {passed} passed, {failed} failed")
print("=" * 80)
print()
print("EXPECTED BEHAVIOR IN APP:")
print("=" * 80)
print("1. Upload credit cards with year selection 2024")
print("2. System extracts 'Opening/Closing Date: 03/14/24 - 05/13/24' from summary")
print("3. Parses to: start=2024-03-14, end=2024-05-13")
print("4. Filter shows: 2024/03/14 to 2024/05/13")
print("5. ✅ Matches the credit card summary exactly!")
