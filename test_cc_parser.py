import re
from typing import List

# Mock elements from bank_data_analysis
def _normalize_date_token(d): return d
def _clean_amount_token(s): 
    try: return float(re.sub(r'[^\d.-]', '', s))
    except: return None
def _short_vendor(v): return v
def _clean_description(*args): return args[0]

class CreditCardParser:
    DATE_RE = re.compile(r'(?<!\d)(?<!\d-)\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?(?!\d|-)')
    AMOUNT_RE = re.compile(r'\(?\$?\d{1,3}(?:,\d{3})*\.\d{2}\)?')

    def parse(self, lines: List[str]):
        txs = []
        for ln in lines:
            if not self.DATE_RE.search(ln):
                continue
            
            date_raw = self.DATE_RE.search(ln).group()
            date_norm = _normalize_date_token(date_raw)
            
            ln_no_date = ln.replace(date_raw, "")
            amts = self.AMOUNT_RE.findall(ln_no_date)
            if not amts:
                print(f"MISSING AMOUNT: {ln_no_date}")
                continue
                
            amt = _clean_amount_token(amts[-1])
            if amt is None:
                continue
                
            amt = -abs(amt)
            vendor = ln.replace(date_raw, '').replace(amts[-1], '').strip()
            
            txs.append({"date": date_norm, "amount": amt, "vendor": vendor, "raw": ln})
        return txs

with open("amex_raw_full.txt", "r", encoding="utf-16", errors="replace") as f:
    lines = f.read().splitlines()

p = CreditCardParser()
res = p.parse(lines)
print(f"Total extracted: {len(res)}")
