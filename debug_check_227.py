import pdfplumber

pdf_path = r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App\bank statments\2CED020D-078D-423B-A7BE-CFE61B680C1D-list.pdf"

print("=== SEARCHING FOR CHECK #227 IN PDF ===\n")

with pdfplumber.open(pdf_path) as pdf:
    for page_num, page in enumerate(pdf.pages, 1):
        text = page.extract_text()
        if text and ('227' in text or 'check' in text.lower()):
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if '227' in line or ('check' in line.lower() and ('157' in line or '158' in line or '225' in line or '226' in line)):
                    print(f"Page {page_num}, Line {i}: {repr(line)}")
                    # Show context (2 lines before and after)
                    if i > 0:
                        print(f"  [BEFORE] {repr(lines[i-1])}")
                    print(f"  [MATCH]  {repr(lines[i])}")
                    if i < len(lines) - 1:
                        print(f"  [AFTER]  {repr(lines[i+1])}")
                    print()

print("\n=== EXPECTED CHECKS ===")
print("Check 157: $1,800.00")
print("Check 158: $1,000.00")
print("Check 225: $500.00")
print("Check 226: $1,235.25")
print("Check 227: $1,235.25")
print("Total: $5,770.50 (Note: Chase shows $6,270.50 which includes one more check)")
