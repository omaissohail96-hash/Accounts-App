import sys

# The problematic strings from the earlier output
line2 = '12/22/25 207┬░ -35,000.00 12/22/25 394┬░ -350.00'

print(f"Line: {line2}")
print("Chars and Hex:")
for char in line2:
    print(f"  '{char}' : {hex(ord(char))}")

import re
new_check_regex = r'(\d{1,2}/\d{1,2}(?:/\d{2,4}|/\d{2})?)\s+([\d_:~.*"\'°┬-]+)\s+(?:~|_|:)*([-~+]?[\d,\.]+\.\d{2})'
print(f"\nRegex: {new_check_regex}")
match = re.search(new_check_regex, line2)
if match:
    print("Match Found!")
else:
    print("Match Failed.")
    # Debug part by part
    date_part = r'(\d{1,2}/\d{1,2}(?:/\d{2,4}|/\d{2})?)'
    print(f"Date match: {re.match(date_part, line2).group(0) if re.match(date_part, line2) else 'None'}")
    
    rest = line2[8:] # after '12/22/25'
    print(f"Rest: '{rest}'")
    after_date = r'\s+([\d_:~.*"\'°┬-]+)'
    m_after = re.match(after_date, rest)
    if m_after:
        print(f"Check match: {m_after.group(1)}")
        rest2 = rest[m_after.end():]
        print(f"Rest 2: '{rest2}'")
    else:
        print("Check match failed.")
        # Try a more liberal check match
        liberal_check = r'\s+(\S+)'
        m_lib = re.match(liberal_check, rest)
        if m_lib:
            print(f"Liberal Check match: {m_lib.group(1)}")
            for c in m_lib.group(1):
                print(f"  '{c}' : {hex(ord(c))}")
