import re
import pdfplumber

pdf_path = r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App\bank statments\2CED020D-078D-423B-A7BE-CFE61B680C1D-list.pdf"

print("=== CHASE SUMMARY METRICS VERIFICATION ===\n")

# Extract raw text from PDF
with pdfplumber.open(pdf_path) as pdf:
    all_pages_text = []
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            all_pages_text.append(text)
    
    raw_text = "\n".join(all_pages_text)

# Extract Chase summary
summary_match = re.search(r'\*start\*summary.*?\*end\*summary', raw_text, re.DOTALL | re.IGNORECASE)

if summary_match:
    summary_section = summary_match.group(0)
    
    # Initialize result structure
    result = {
        "Beginning Balance": {"count": "", "amount": 0.0},
        "Deposits and Additions": {"count": 0, "amount": 0.0},
        "Checks Paid": {"count": 0, "amount": 0.0},
        "ATM & Debit Card Withdrawals": {"count": 0, "amount": 0.0},
        "Electronic Withdrawals": {"count": 0, "amount": 0.0},
        "Fees": {"count": 0, "amount": 0.0},
        "Ending Balance": {"count": "", "amount": 0.0}
    }
    
    # Parse summary
    lines = summary_section.split('\n')
    for line in lines:
        if any(skip in line for skip in ['*start*', '*end*', 'CHECKING SUMMARY', 'INSTANCES AMOUNT', 'Chase Business']):
            continue
        if not line.strip():
            continue
        
        match = re.search(r'^(.+?)\s+(\d+)?\s*([\$\-]?[\d,]+\.?\d*)$', line.strip())
        if match:
            category_raw = match.group(1).strip()
            instances_str = match.group(2) if match.group(2) else ""
            amount_str = match.group(3).strip().replace('$', '').replace(',', '')
            
            category_mapping = {
                "beginning balance": "Beginning Balance",
                "deposits and additions": "Deposits and Additions",
                "checks paid": "Checks Paid",
                "atm & debit card withdrawals": "ATM & Debit Card Withdrawals",
                "electronic withdrawals": "Electronic Withdrawals",
                "fees": "Fees",
                "ending balance": "Ending Balance"
            }
            
            category_key = category_mapping.get(category_raw.lower())
            if category_key:
                try:
                    amount = float(amount_str)
                    instances = int(instances_str) if instances_str else ""
                    result[category_key] = {"count": instances, "amount": amount}
                except ValueError:
                    continue
    
    # Calculate Summary Metrics (as they would appear in the UI)
    deposits_amount = result["Deposits and Additions"]["amount"]
    withdrawals_amount = (abs(result["Checks Paid"]["amount"]) + 
                         abs(result["ATM & Debit Card Withdrawals"]["amount"]) + 
                         abs(result["Electronic Withdrawals"]["amount"]) + 
                         abs(result["Fees"]["amount"]))
    net_income = deposits_amount - withdrawals_amount
    
    deposits_count = result["Deposits and Additions"]["count"]
    withdrawals_count = (result["Checks Paid"]["count"] + 
                        result["ATM & Debit Card Withdrawals"]["count"] + 
                        result["Electronic Withdrawals"]["count"] + 
                        result["Fees"]["count"])
    total_transactions = result["Ending Balance"]["count"]  # Chase puts total count in ending balance
    
    print("SUMMARY METRICS (should match UI after fix):")
    print("=" * 70)
    print(f"Total Deposits:     USD {deposits_amount:,.2f}  ({deposits_count} tx)")
    print(f"Total Withdrawals:  USD {withdrawals_amount:,.2f}  ({withdrawals_count} tx)")
    print(f"Net Income:         USD {net_income:,.2f}")
    print(f"Transactions:       {total_transactions}")
    print("=" * 70)
    
    print("\nBREAKDOWN:")
    print(f"  Checks Paid:        ${abs(result['Checks Paid']['amount']):,.2f} ({result['Checks Paid']['count']} tx)")
    print(f"  ATM Withdrawals:    ${abs(result['ATM & Debit Card Withdrawals']['amount']):,.2f} ({result['ATM & Debit Card Withdrawals']['count']} tx)")
    print(f"  Electronic:         ${abs(result['Electronic Withdrawals']['amount']):,.2f} ({result['Electronic Withdrawals']['count']} tx)")
    print(f"  Fees:               ${abs(result['Fees']['amount']):,.2f} ({result['Fees']['count']} tx)")
    print(f"  ---")
    print(f"  Total Withdrawals:  ${withdrawals_amount:,.2f} ({withdrawals_count} tx)")
    
    print("\n✅ The Summary metrics should now show these correct values!")
    print("\nPreviously showed (INCORRECT):")
    print("  Total Deposits:     USD 59,677.76")
    print("  Total Withdrawals:  USD 56,104.17")

else:
    print("❌ Chase summary section not found!")
