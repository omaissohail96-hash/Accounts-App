import sys
import os
import re
import json
from typing import Dict, Any

def denoise_text(text: str) -> str:
    """
    Chase OCR often duplicates characters in headers: AACCCCOOUUNNTT
    This function tries to collapse them if they are in uppercase blocks.
    """
    def collapse(match):
        s = match.group(0)
        # Only collapse if most characters are duplicated
        new_s = ""
        i = 0
        while i < len(s):
            new_s += s[i]
            if i + 1 < len(s) and s[i] == s[i+1]:
                i += 2
            else:
                i += 1
        return new_s

    # Collapsing sequences of duplicated uppercase letters
    return re.sub(r'[A-Z]{4,}', collapse, text)

def clean_amt(s: str) -> float:
    if not s: return 0.0
    # Aggressive cleanup
    s = s.strip().replace('$', '').replace(',', '').replace(' ', '').replace('+', '')
    if '(' in s and ')' in s:
        s = '-' + s.replace('(', '').replace(')', '')
    if s.endswith('-'): s = '-' + s[:-1]
    
    # Remove any non-numeric/sign/dot characters
    s = re.sub(r'[^0-9.\-]', '', s)
    
    try:
        return float(s)
    except ValueError:
        return 0.0

def extract_cc_summary(raw_text: str) -> Dict[str, Any]:
    result = {
        "Account Number": "N/A",
        "Previous Balance": 0.0,
        "Payment, Credits": 0.0,
        "Purchases": 0.0,
        "Cash Advances": 0.0,
        "Balance Transfers": 0.0,
        "Fees Charged": 0.0,
        "Interest Charged": 0.0,
        "New Balance": 0.0,
        "Opening/Closing Date": "N/A",
        "Revolving Credit Amount": 0.0,
        "Available Credit": 0.0,
        "Cash Access Line": 0.0,
        "Available for Cash": 0.0,
        "Past Due Amount": 0.0,
        "Balance over the Credit Access Line": 0.0
    }
    
    # 1. Denoise headers
    denoised_text = denoise_text(raw_text)
    
    # 2. Join split numbers across lines
    lines = denoised_text.split('\n')
    joined_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
            
        # Check if this line ends with a "hanging" number and the next line finishes it
        # e.g. "10,339.5" and "5"
        if i + 1 < len(lines):
            next_line = lines[i+1].strip()
            # If line ends in something like "1,234.5" and next line is just "6"
            if re.search(r'[\d,]+\.\d$', line) and re.match(r'^\d$', next_line):
                line += next_line
                i += 1
            elif re.search(r'[\d,]+$', line) and re.match(r'^\.\d{2}$', next_line):
                line += next_line
                i += 1
                
        joined_lines.append(line)
        i += 1

    joined_text = '\n'.join(joined_lines)
    
    # 3. Extract Section
    summary_start = -1
    for hp in [r'ACCOUNT SUMMARY', r'SUMMARY OF ACCOUNT', r'ACCOUNT AT A GLANCE']:
        m = re.search(hp, joined_text, re.I)
        if m:
            summary_start = m.start()
            break
    
    if summary_start == -1:
        summary_start = 0
        
    summary_text = joined_text[summary_start:]
    m_end = re.search(r'ACCOUNT ACTIVITY|TRANSACTION DETAIL|ACTIVITY DETAIL', summary_text[50:], re.I)
    if m_end:
        summary_text = summary_text[:50+m_end.start()]
    else:
        summary_text = summary_text[:2000]

    print(f"DEBUG: Cleaned Summary Section (Start Pos: {summary_start}):")
    print("-" * 20)
    print(summary_text)
    print("-" * 20)

    # 4. Field Matching
    # Fuzzy labels (ignore extra letters/spaces)
    p_val = r'[\$]?\s*([+\-]?\s*[\d,]+\.\d{2}|[+\-]?\s*[\d,]{1,8})'
    
    fields = {
        "Account Number": r'Account Number[:\s]+([\d\s]{10,25})',
        "Previous Balance": r'Previous\s+Balance',
        "Payment, Credits": r'Pay\s?ments?,\s?Credits',
        "Purchases": r'Purchases',
        "Cash Advances": r'Cash\s+Advances(?!\s+Line)',
        "Balance Transfers": r'Balance\s+Transfers',
        "Fees Charged": r'Fees\s+Charged',
        "Interest Charged": r'Interest\s+Charged',
        "New Balance": r'New\s+Balance',
        "Revolving Credit Amount": r'Revolving\s+Credit\s+Amount|Credit\s+Limit|Total\s+Credit\s+Line',
        "Available Credit": r'Available\s+Credit',
        "Cash Access Line": r'Cash\s+Access\s+Line',
        "Available for Cash": r'Available\s+for\s+Cash',
        "Past Due Amount": r'Past\s+Due\s+Amount',
        "Balance over the Credit Access Line": r'Balance\s+over\s+the\s+Credit\s+Access\s+Line'
    }

    for key, f_pat in fields.items():
        # Match label + optional noise + amount
        full_pat = f'{f_pat}.*?{p_val}'
        match = re.search(full_pat, summary_text, re.I | re.DOTALL)
        if match:
            if key == "Account Number":
                result[key] = match.group(1).strip()
            else:
                result[key] = clean_amt(match.group(1))
            print(f"MATCH: {key} -> {result[key]}")

    return result

# PDF Extraction
import pdfplumber
pdf_path = r'c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App\credit card statments\B5A94623-DDC5-4D85-8A10-719E833A8BC4-list (1).pdf'

try:
    with pdfplumber.open(pdf_path) as pdf:
        all_text = ""
        for page in pdf.pages:
            all_text += page.extract_text() + "\n"
    
    results = extract_cc_summary(all_text)
    print("\nFINAL RESULTS:")
    print(json.dumps(results, indent=2))

except Exception as e:
    print(f"Error: {e}")
