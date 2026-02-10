import re
import pdfplumber

pdf_path = r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App\bank statments\2CED020D-078D-423B-A7BE-CFE61B680C1D-list.pdf"

print("=== CHASE SUMMARY EXTRACTION TEST ===\n")

# Extract raw text from PDF
with pdfplumber.open(pdf_path) as pdf:
    all_pages_text = []
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            all_pages_text.append(text)
    
    raw_text = "\n".join(all_pages_text)

# Look for Chase summary section
summary_match = re.search(r'\*start\*summary.*?\*end\*summary', raw_text, re.DOTALL | re.IGNORECASE)

if not summary_match:
    print("❌ Chase summary section not found!")
else:
    print("✅ Found Chase summary section!\n")
    
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
    
    # Parse each line in the summary
    lines = summary_section.split('\n')
    for line in lines:
        # Skip markers and headers
        if any(skip in line for skip in ['*start*', '*end*', 'CHECKING SUMMARY', 'INSTANCES AMOUNT', 'Chase Business']):
            continue
        if not line.strip():
            continue
        
        # Try to extract: Category [instances] [amount]
        match = re.search(r'^(.+?)\s+(\d+)?\s*([\$\-]?[\d,]+\.?\d*)$', line.strip())
        if match:
            category_raw = match.group(1).strip()
            instances_str = match.group(2) if match.group(2) else ""
            amount_str = match.group(3).strip().replace('$', '').replace(',', '')
            
            # Map category names
            category_mapping = {
                "beginning balance": "Beginning Balance",
                "deposits and additions": "Deposits and Additions",
                "checks paid": "Checks Paid",
                "atm & debit card withdrawals": "ATM & Debit Card Withdrawals",
                "atm and debit card withdrawals": "ATM & Debit Card Withdrawals",
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
    
    print("EXTRACTED SUMMARY:")
    print("-" * 70)
    print(f"{'Category':<35} {'Instances':>10} {'Amount':>20}")
    print("-" * 70)
    for category, values in result.items():
        count_str = str(values["count"]) if values["count"] != "" else "-"
        print(f"{category:<35} {count_str:>10} ${values['amount']:>18,.2f}")
    print("\n" + "=" * 70)
    
    print("\nEXPECTED VALUES FROM CHASE STATEMENT:")
    print("-" * 70)
    print("Beginning Balance                                      $10,783.06")
    print("Deposits and Additions                    52           $48,894.70")
    print("Checks Paid                               6            $-6,270.50")
    print("ATM & Debit Card Withdrawals              5            $-1,920.00")
    print("Electronic Withdrawals                    17          $-49,146.42")
    print("Fees                                      1               $-2.50")
    print("Ending Balance                            81            $2,338.34")
    print("\n✅ TEST COMPLETE!")
