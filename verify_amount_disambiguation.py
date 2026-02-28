
import re
import phonenumbers
from commonregex import CommonRegex
from typing import List, Optional

def _mask_non_amount_entities(line: str) -> str:
    if not line: return ""
    masked_line = line
    try:
        for match in phonenumbers.PhoneNumberMatcher(line, "US"):
            phone_str = line[match.start:match.end]
            masked_line = masked_line.replace(phone_str, " " * len(phone_str))
    except Exception: pass
    try:
        parsed = CommonRegex(masked_line)
        for date_str in parsed.dates:
            if len(date_str) >= 5:
                masked_line = masked_line.replace(date_str, " " * len(date_str))
    except Exception: pass
    return masked_line

AMOUNT_RE = re.compile(r'([+\-]?\(?\s*\$?\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})?\)?)')

def test_disambiguation():
    test_cases = [
        {
            "line": "01/02 AMEX Payment 800-555-0123  1,234.56",
            "expected_amount": "1,234.56",
            "desc": "Phone number + Amount"
        },
        {
            "line": "Call support 1-800-555-0199 for help",
            "expected_amount": None,
            "desc": "Only phone number"
        },
        {
            "line": "Statement Date 12/25/2024 Balance 0.00",
            "expected_amount": "0.00",
            "desc": "Date + Balance"
        },
        {
            "line": "Transaction on 2024-11-20 for 99.99",
            "expected_amount": "99.99",
            "desc": "ISO Date + Amount"
        }
    ]
    
    print("Starting Amount Disambiguation Verification...")
    print("-" * 50)
    
    all_passed = True
    for case in test_cases:
        line = case["line"]
        masked = _mask_non_amount_entities(line)
        matches = AMOUNT_RE.findall(masked)
        # Filter for digits and strip whitespace
        results = [m.strip() for m in matches if re.search(r'\d', m)]
        
        # In our logic, we usually take the last one or two.
        result = results[-1] if results else None
        
        if result == case["expected_amount"]:
            print(f"PASSED: {case['desc']}")
        else:
            print(f"FAILED: {case['desc']}")
            print(f"  Line: {line}")
            print(f"  Masked: {masked}")
            print(f"  Extracted: {result}, Expected: {case['expected_amount']}")
            all_passed = False
            
    print("-" * 50)
    if all_passed:
        print("ALL TESTS PASSED! Masking logic is working correctly.")
    else:
        print("SOME TESTS FAILED. Retrying refinement.")

if __name__ == "__main__":
    test_disambiguation()
