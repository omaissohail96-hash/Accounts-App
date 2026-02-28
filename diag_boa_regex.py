import re

# Current Regex for BoA
# r'(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\s+(.*?)\s+([-+]?\$?[\d, \.]+\.\d{2})$'
# Note the space in [\d, \. ]

regex_with_space = r'(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\s+(.*?)\s+([-+]?\$?[\d, \.]+\.\d{2})$'
regex_no_space = r'(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\s+(.*?)\s+([-+]?\$?[\d,\.]+\.\d{2})$'

line = '12/17/25 Zelle payment from BENEDICTO PANA JR for "US Immigration Physical Exams"; Conf# 270944002 930.00'

print(f"Line: {line}")

m1 = re.search(regex_with_space, line)
if m1:
    print("\nRegex WITH SPACE matched:")
    print(f"  Group 1 (Date): {m1.group(1)}")
    print(f"  Group 2 (Desc): {m1.group(2)}")
    print(f"  Group 3 (Amt):  {m1.group(3)}")
else:
    print("\nRegex WITH SPACE failed to match.")

m2 = re.search(regex_no_space, line)
if m2:
    print("\nRegex NO SPACE matched:")
    print(f"  Group 1 (Date): {m2.group(1)}")
    print(f"  Group 2 (Desc): {m2.group(2)}")
    print(f"  Group 3 (Amt):  {m2.group(3)}")
else:
    print("\nRegex NO SPACE failed to match.")
