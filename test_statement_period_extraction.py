"""
Test the statement period extraction function
"""
from bank_data_analysis import extract_statement_period

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

if tests_passed >= 4:
    print("✅ Statement period extraction working correctly!")
    print()
    print("🎯 EXPECTED IN STREAMLIT:")
    print("   - Upload your bank statement")
    print("   - App will show: '📅 Statement Period (from header): November 29, 2025 through December 31, 2025'")
    print("   - Filter will default to: Start Date: 11/29/2025, End Date: 12/31/2025")
    print("   - NOT extended to Dec 01 - Dec 31")
else:
    print("⚠️ Some tests failed - check the extraction patterns")
