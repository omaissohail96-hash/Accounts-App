import sys
import os
sys.path.append(os.getcwd())

# Mock streamlit to allow importing bank_data_analysis
from unittest.mock import MagicMock
sys.modules["streamlit"] = MagicMock()

from bank_statement_parser import BankStatementParser
import pdfplumber

pdf_path = r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App\bank statments\2CED020D-078D-423B-A7BE-CFE61B680C1D-list.pdf"

print("=== TESTING CHASE STATEMENT PARSER ===\n")

# Extract text from PDF
with pdfplumber.open(pdf_path) as pdf:
    all_lines = []
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            all_lines.extend(text.split('\n'))

# Parse using our parser
parser = BankStatementParser()
transactions, metadata = parser.parse_statement(all_lines)

print(f"Total transactions parsed: {len(transactions)}\n")

# Calculate summary from parsed transactions
deposits = [t for t in transactions if t.amount > 0]
checks = [t for t in transactions if t.amount < 0 and "check" in t.description.lower()]
atm = [t for t in transactions if t.amount < 0 and "atm" in t.description.lower()]
electronic = [t for t in transactions if t.amount < 0 and "electronic" in t.transaction_type.lower()]
fees = [t for t in transactions if t.amount < 0 and "fee" in t.description.lower()]

print("=== PARSED SUMMARY ===")
print(f"Deposits: {len(deposits)} transactions, ${sum(t.amount for t in deposits):,.2f}")
print(f"Checks: {len(checks)} transactions, ${sum(t.amount for t in checks):,.2f}")
print(f"ATM: {len(atm)} transactions, ${sum(t.amount for t in atm):,.2f}")
print(f"Electronic: {len(electronic)} transactions, ${sum(t.amount for t in electronic):,.2f}")
print(f"Fees: {len(fees)} transactions, ${sum(t.amount for t in fees):,.2f}")

print("\n=== EXPECTED FROM CHASE STATEMENT ===")
print("Deposits: 52 transactions, $48,894.70")
print("Checks: 6 transactions, $-6,270.50")
print("ATM: 5 transactions, $-1,920.00")
print("Electronic: 17 transactions, $-49,146.42")
print("Fees: 1 transactions, $-2.50")

# Show sample transactions
print("\n=== SAMPLE PARSED TRANSACTIONS ===")
for i, t in enumerate(transactions[:10]):
    print(f"{i+1}. {t.date} | {t.amount:>10,.2f} | {t.description[:60]}")
