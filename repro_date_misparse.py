
import re
from datetime import datetime
from typing import Optional

def _mask_non_amount_entities(line: str) -> str:
    # Simplified version for test
    return line

def _clean_amount_token(token: str) -> Optional[float]:
    if not token: return None
    s = str(token).strip()
    negative = False
    if s.startswith("(") and s.endswith(")"): negative = True
    if s.startswith("-"): negative = True
    s = re.sub(r'[A-Za-z\$£€₹]', '', s)  # drop currency letters
    s = s.replace(',', '').replace(' ', '')
    s = s.replace('¢', '').replace('\u00a2', '').replace('#', '').replace('*', '').replace('+', '')
    s = re.sub(r'[^0-9\.\-]', '', s)
    if not re.search(r'\d', s): return None
    parts = s.split('.')
    if len(parts) > 2:
        s = "".join(parts[:-1]) + "." + parts[-1]
    try:
        val = float(s)
        return -abs(val) if negative else abs(val)
    except:
        return None

def diag_misparse():
    # Example line from user screenshot but with possible OCR space in date
    line = "11 24 25  MICROSOFT  MSBILL.INFO  $38.12"
    
    # Current regexes in the code
    DATE_RE = re.compile(r'(?<!\d)(?<!\d-)(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)(?!\d|-)')
    AMOUNT_RE = re.compile(r'\(?\$?[\d, \.]+\d{2}\)?')
    
    print(f"Original line: {line}")
    
    # 1. Try to find date
    dm = DATE_RE.search(line)
    if dm:
        date_raw = dm.group(1)
        print(f"Date found: {date_raw}")
        ln_no_date = line.replace(date_raw, " " * len(date_raw))
    else:
        print("Date NOT found (as expected if OCR used spaces)")
        ln_no_date = line
        
    # 2. Extract amounts
    amts = AMOUNT_RE.findall(ln_no_date)
    print(f"Amounts found: {amts}")
    
    if amts:
        for a in amts:
            val = _clean_amount_token(a)
            print(f"  Token: '{a}' -> Parsed value: {val}")

if __name__ == "__main__":
    diag_misparse()
