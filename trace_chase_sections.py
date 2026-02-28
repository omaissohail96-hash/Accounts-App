import re
from document_parser import DocumentParser
from bank_statement_parser import BANK_LAYOUTS, BankName

PDF = "bank statments/20 Dec Chase.pdf"

print("=== Reading PDF with DocumentParser ===")
dp = DocumentParser()
with open(PDF, "rb") as f:
    text_lines, ok, unreadable = dp.parse_document(f.read(), "20 Dec Chase.pdf")

layout = BANK_LAYOUTS[BankName.CHASE]
regex = layout.transaction_regex

current_section = None
dropped_lines = []

for line in text_lines:
    line_upper = line.upper().strip()
    
    # Check section headers
    for header, info in layout.section_headers.items():
        if header in line_upper:
            current_section = info
            break
            
    # Check summary markers
    if any(marker in line_upper for marker in layout.summary_markers):
        current_section = None
        continue

    # Try transaction regex
    if re.search(r'\d{1,2}/\d{1,2}', line):
        if re.search(regex, line):
            if not current_section:
                dropped_lines.append(line)

print(f"\nDropped {len(dropped_lines)} lines because current_section was None:")
for i, l in enumerate(dropped_lines):
    print(f"{i}: {l}")
