import re
from typing import List, Dict, Any, Optional, Tuple

class Transaction:
    def __init__(self, date, transaction_type, vendor, amount, description, raw_line, section, needs_review=False):
        self.date = date
        self.transaction_type = transaction_type
        self.vendor = vendor
        self.amount = amount
        self.description = description
        self.raw_line = raw_line
        self.section = section
        self.needs_review = needs_review

    def __repr__(self):
        return f"Transaction(date='{self.date}', vendor='{self.vendor}', amount={self.amount}, description='{self.description}')"

class MockParser:
    def __init__(self, extract_check_memos=True):
        self.extract_check_memos = extract_check_memos

    def _parse_amount(self, amt_raw):
        return float(amt_raw.replace(",", ""))

    def _parse_date(self, date_raw):
        return f"2024-{date_raw.split('/')[0].zfill(2)}-{date_raw.split('/')[1].zfill(2)}"

    def _finalize_check_tx(self, check_data: Dict[str, Any]) -> Transaction:
        date_norm = ""
        if check_data["date"]:
            date_norm = self._parse_date(check_data["date"]) or ""

        desc = f"Check #{check_data['check_no']}"
        if check_data["memo"]:
            desc = f"Check #{check_data['check_no']} - {check_data['memo']}"

        return Transaction(
            date=date_norm,
            transaction_type="withdrawal",
            vendor=f"Check #{check_data['check_no']}",
            amount=-abs(check_data["amount"]) if check_data["amount"] else 0.0,
            description=desc,
            raw_line="\n".join(check_data["raw_lines"]),
            section="CHECKS",
            needs_review=False
        )

    def _parse_checks_section(self, lines: List[str]) -> List[Transaction]:
        txs = []
        current_check = None

        for ln in lines:
            ln_stripped = ln.strip()
            if not ln_stripped:
                continue

            m = re.match(r'^(\d{3,6})\s+(.*?)(\d{1,3}(?:,\d{3})*\.\d{2})$', ln_stripped)
            
            if m:
                if current_check:
                    txs.append(self._finalize_check_tx(current_check))
                
                check_no = m.group(1)
                middle_text = m.group(2).strip()
                amt_raw = m.group(3)
                
                date_paid = ""
                memo = middle_text
                date_m = re.search(r'(\d{1,2}/\d{1,2})', middle_text)
                if date_m:
                    date_paid = date_m.group(1)
                    memo = middle_text.replace(date_paid, "").replace("^", "").replace("*", "").strip()
                else:
                    memo = middle_text.replace("^", "").replace("*", "").strip()

                current_check = {
                    "check_no": check_no,
                    "date": date_paid,
                    "amount": self._parse_amount(amt_raw),
                    "memo": memo,
                    "raw_lines": [ln]
                }
            elif current_check and self.extract_check_memos:
                cleaned_extra = ln_stripped.replace("^", "").replace("*", "").strip()
                if cleaned_extra:
                    if current_check["memo"]:
                        current_check["memo"] += " " + cleaned_extra
                    else:
                        current_check["memo"] = cleaned_extra
                current_check["raw_lines"].append(ln)

        if current_check:
            txs.append(self._finalize_check_tx(current_check))

        return txs

# Test cases
sample_lines = [
    "157 ^ 12/31 1,800.00",
    "158 ^ RENT FOR JAN 12/11 1,000.00",
    "225 *^ 12/06 500.00",
    "226 ^ 12/16 1,235.25",
    "Memo for 226 line 2",
    "More memo for 226",
    "227 ^ 12/16 1,235.25"
]

parser = MockParser(extract_check_memos=True)
results = parser._parse_checks_section(sample_lines)

print("--- Results with extract_check_memos=True ---")
for r in results:
    print(r)

parser_off = MockParser(extract_check_memos=False)
results_off = parser_off._parse_checks_section(sample_lines)

print("\n--- Results with extract_check_memos=False ---")
for r in results_off:
    print(r)
