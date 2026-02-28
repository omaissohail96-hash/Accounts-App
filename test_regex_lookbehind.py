import re

DATE_RE = re.compile(
    r'(?<!\d)(?<!\d-)(\d{1,2}\s+[A-Za-z]{3,9},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)(?!\d|-)'
)

lines = [
    '11/24/25 38.12 MICROSOFT MSBILL.INFO',
    '12/01/25      52.80 GOOGLE *WORKSPACE',
    '11/24 24.00 VENDOR',
    '01/01/2026 123.45 Test',
    '800-922-0204'
]

for ln in lines:
    m = DATE_RE.search(ln)
    print(f"{ln!r} -> {m.group(1) if m else None}")
