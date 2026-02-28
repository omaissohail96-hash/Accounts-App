import dateparser
from datetime import datetime
import re

def parse_date(date_str, year=None):
    try:
        date_str = date_str.strip()
        if re.match(r'^\d{1,2}/\d{1,2}$', date_str):
            if year:
                date_str = f"{date_str}/{year}"
            else:
                date_str = f"{date_str}/{datetime.now().year}"
        
        parsed = dateparser.parse(date_str, settings={'PREFER_DATES_FROM': 'past', 'DATE_ORDER': 'MDY'})
        return parsed
    except Exception as e:
        return None

dates = ["12/22/25", "12/22/2025", "12/22"]
for d in dates:
    p = parse_date(d, 2025)
    print(f"Input: {d} | Parsed: {p}")
