import re

def test_line():
    lines = [
        '12/15/25 Zelle payment from MUHAMMAD SOHAIL for ΓÇ£Medical examΓÇÖ; Conf# 126494303 375.00   ',
        '12/29/25 ‘Squa re Inc DES:SQ251229 1D:T 3T732FQYDEYFAT INDN-Aziz Family Medical ce CO 20 1.47',
        '12/29/25 Squar e Inc  DES:SQ251229 ID:T 3ESQ6E8E57RQ5K INDN:Aziz Family Medical ce CO 10 5.69'
    ]
    
    # Current Mappings in BankStatementParser
    date_patterns = [
        r'(?i)Conf#\s*\w+',
    ]
    
    for line in lines:
        line = line.strip()
        masked = line
        for p in date_patterns:
            masked = re.sub(p, "[MASKED]", masked)
        
        print(f"\nOriginal: '{line}'")
        # print(f"Masked:   '{masked}'")
        
        tx_regex = r'(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\s+(.*?)\s+([-~+]?\$?[\d, \.]+\.\d{2})$'
        
        m = re.search(tx_regex, masked)
        if m:
            print("MATCH FOUND!")
            print(f"  Date: {m.group(1)}")
            print(f"  Desc: {m.group(2)}")
            print(f"  Amt:  '{m.group(3)}'")
        else:
            print("MATCH FAILED.")
        # Debug why
        if not re.match(r'^\d{1,2}/\d{1,2}', line):
            print("  Date part failed")
        if not re.search(r'([-~+]?\$?[\d,\.]+\.\d{2})$', line):
            print("  Amount part failed")

if __name__ == "__main__":
    test_line()
