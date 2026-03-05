"""
Test the updated pattern with the actual text from user's statement
"""
import re
from datetime import datetime

def extract_statement_period_test(raw_text: str):
    """Test extraction with the actual concatenated text"""
    
    # Pattern with \s* to handle zero or more spaces
    pattern2 = r'([A-Z][a-z]+\s+\d{1,2},\s+\d{4})\s*(?:through|to|-)\s*([A-Z][a-z]+\s+\d{1,2},\s+\d{4})'
    match2 = re.search(pattern2, raw_text, re.I)
    
    if match2:
        start_str = match2.group(1)
        end_str = match2.group(2)
        print(f"✅ Match found!")
        print(f"  Start string: '{start_str}'")
        print(f"  End string: '{end_str}'")
        
        try:
            start_date = datetime.strptime(start_str, '%B %d, %Y').date()
            end_date = datetime.strptime(end_str, '%B %d, %Y').date()
            return (start_date, end_date)
        except ValueError as e:
            print(f"  ❌ Parse error: {e}")
            return None
    else:
        print("❌ No match found")
        return None

# Test with actual text from user's statement (concatenated without space)
actual_text = """4200000001040071530
November 29, 2025throughDecember 31, 2025
JPMorgan Chase Bank, N.A."""

print("=" * 80)
print("TEST WITH ACTUAL STATEMENT TEXT (NO SPACE BEFORE 'through')")
print("=" * 80)
print(f"Input text:\n{actual_text}\n")

result = extract_statement_period_test(actual_text)

if result:
    print(f"\n✅ SUCCESS!")
    print(f"  Extracted: {result[0].strftime('%B %d, %Y')} through {result[1].strftime('%B %d, %Y')}")
    print(f"  Start: {result[0]}")
    print(f"  End: {result[1]}")
else:
    print(f"\n❌ FAILED to extract dates")

print("\n" + "=" * 80)
print("EXPECTED RESULT:")
print("=" * 80)
print("  Start Date: November 29, 2025")
print("  End Date: December 31, 2025")
