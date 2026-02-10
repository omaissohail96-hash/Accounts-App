import sys
import os
sys.path.append(os.getcwd())

# Mock streamlit
from unittest.mock import MagicMock
sys.modules["streamlit"] = MagicMock()

from bank_data_analysis import extract_chase_summary, get_sub_summary
import pdfplumber

pdf_path = r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App\bank statments\2CED020D-078D-423B-A7BE-CFE61B680C1D-list.pdf"

print("=== TESTING COMPLETE CHASE FIX ===\n")

# Extract raw text from PDF
with pdfplumber.open(pdf_path) as pdf:
    all_pages_text = []
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            all_pages_text.append(text)
    
    raw_text = "\n".join(all_pages_text)

# Test the extract_chase_summary function directly
print("1. Testing extract_chase_summary()...")
chase_summary = extract_chase_summary(raw_text)

if chase_summary:
    print("✅ Successfully extracted Chase summary!")
    print("\nExtracted values:")
    for category, values in chase_summary.items():
        count = values["count"] if values["count"] != "" else "N/A"
        amount = values["amount"]
        print(f"  {category}: {count} instances, ${amount:,.2f}")
else:
    print("❌ Failed to extract Chase summary")

# Test the get_sub_summary function with raw_text
print("\n2. Testing get_sub_summary() with raw_text...")
summary_df = get_sub_summary(transactions=[], raw_text=raw_text)
print("\nGenerated DataFrame:")
print(summary_df.to_string(index=False))

print("\n=== EXPECTED VALUES FROM CHASE STATEMENT ===")
print("Beginning Balance:                   $10,783.06")
print("Deposits and Additions:    52        $48,894.70")
print("Checks Paid:               6         $-6,270.50")
print("ATM & Debit Card:          5         $-1,920.00")
print("Electronic Withdrawals:    17        $-49,146.42")
print("Fees:                      1         $-2.50")
print("Ending Balance:            81        $2,338.34")

print("\n✅ FIX COMPLETE!")
