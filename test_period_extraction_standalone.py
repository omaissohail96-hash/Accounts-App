"""
Direct test of statement period extraction logic
"""
import re
from datetime import datetime
from typing import Optional, Tuple
from datetime import date

def extract_statement_period(raw_text: str) -> Optional[Tuple[date, date]]:
    """
    Extract statement period dates from bank statement header.
    Returns tuple of (start_date, end_date) or None if not found.
    """
    # Pattern 1: MM/DD/YYYY or MM/DD/YY format with various separators
    pattern1 = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s*(?:[-–to]+|through)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
    match1 = re.search(pattern1, raw_text, re.I)
    
    if match1:
        start_str = match1.group(1)
        end_str = match1.group(2)
        
        try:
            # Try parsing with different formats
            for fmt in ['%m/%d/%Y', '%m-%d-%Y', '%m/%d/%y', '%m-%d-%y']:
                try:
                    start_date = datetime.strptime(start_str, fmt).date()
                    end_date = datetime.strptime(end_str, fmt).date()
                    return (start_date, end_date)
                except ValueError:
                    continue
        except Exception:
            pass
    
    # Pattern 2: "Month DD, YYYY through Month DD, YYYY"
    pattern2 = r'([A-Z][a-z]+\s+\d{1,2},\s+\d{4})\s+(?:through|to|-)\s+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})'
    match2 = re.search(pattern2, raw_text, re.I)
    
    if match2:
        start_str = match2.group(1)
        end_str = match2.group(2)
        
        try:
            start_date = datetime.strptime(start_str, '%B %d, %Y').date()
            end_date = datetime.strptime(end_str, '%B %d, %Y').date()
            return (start_date, end_date)
        except ValueError:
            try:
                start_date = datetime.strptime(start_str, '%b %d, %Y').date()
                end_date = datetime.strptime(end_str, '%b %d, %Y').date()
                return (start_date, end_date)
            except ValueError:
                pass
    
    return None

# Test Case 1: MM/DD/YYYY format with dash
test1 = """
Bank Statement
Statement Period: 11/29/2025 - 12/31/2025
Account Number: 123456789
"""
result1 = extract_statement_period(test1)
print("Test 1 - MM/DD/YYYY format with dash:")
print(f"  Input: 'Statement Period: 11/29/2025 - 12/31/2025'")
print(f"  Result: {result1}")
if result1:
    print(f"  ✅ Extracted: {result1[0].strftime('%B %d, %Y')} through {result1[1].strftime('%B %d, %Y')}")
else:
    print("  ❌ Failed to extract")
print()

# Test Case 2: Full month name format
test2 = """
Bank of America
November 29, 2025 through December 31, 2025
Your account summary
"""
result2 = extract_statement_period(test2)
print("Test 2 - Full month name format:")
print(f"  Input: 'November 29, 2025 through December 31, 2025'")
print(f"  Result: {result2}")
if result2:
    print(f"  ✅ Extracted: {result2[0].strftime('%B %d, %Y')} through {result2[1].strftime('%B %d, %Y')}")
else:
    print("  ❌ Failed to extract")
print()

# Test Case 3: Opening/Closing Date format (credit card style)
test3 = """
Credit Card Statement
Opening/Closing Date: 10/15/2024 - 11/14/2024
Account Number: 987654321
"""
result3 = extract_statement_period(test3)
print("Test 3 - Opening/Closing Date format:")
print(f"  Input: 'Opening/Closing Date: 10/15/2024 - 11/14/2024'")
print(f"  Result: {result3}")
if result3:
    print(f"  ✅ Extracted: {result3[0].strftime('%B %d, %Y')} through {result3[1].strftime('%B %d, %Y')}")
else:
    print("  ❌ Failed to extract")
print()

# Test Case 4: "to" separator
test4 = """
Statement Period
From: 12/01/2024 to 12/31/2024
"""
result4 = extract_statement_period(test4)
print("Test 4 - 'to' separator:")
print(f"  Input: 'From: 12/01/2024 to 12/31/2024'")
print(f"  Result: {result4}")
if result4:
    print(f"  ✅ Extracted: {result4[0].strftime('%B %d, %Y')} through {result4[1].strftime('%B %d, %Y')}")
else:
    print("  ❌ Failed to extract")
print()

# Test Case 5: Two-digit year format
test5 = """
Period: 11/29/25 - 12/31/25
"""
result5 = extract_statement_period(test5)
print("Test 5 - Two-digit year:")
print(f"  Input: 'Period: 11/29/25 - 12/31/25'")
print(f"  Result: {result5}")
if result5:
    print(f"  ✅ Extracted: {result5[0].strftime('%B %d, %Y')} through {result5[1].strftime('%B %d, %Y')}")
else:
    print("  ❌ Failed to extract")
print()

print("=" * 80)
print("SUMMARY:")
print("=" * 80)
tests_passed = sum([1 for r in [result1, result2, result3, result4, result5] if r is not None])
print(f"Tests passed: {tests_passed}/5")
print()

if tests_passed >= 4:
    print("✅ Statement period extraction working correctly!")
    print()
    print("🎯 EXPECTED IN STREAMLIT:")
    print("   - Upload your bank statement")
    print("   - App will show: '📅 Statement Period (from header): November 29, 2025 through December 31, 2025'")
    print("   - Filter will default to: Start Date: 2025/11/29, End Date: 2025/12/31")
    print("   - NOT extended to 2025/12/01 - 2025/12/31")
    print()
    print("💡 REFRESH YOUR STREAMLIT APP NOW (F5 or Ctrl+R)")
else:
    print("⚠️ Some tests failed - check the extraction patterns")
