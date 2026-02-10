import re

# Sample Chase summary section
chase_summary_text = """
*start*summary        
Chase Business Complete Checking
CHECKING SUMMARY      
INSTANCES AMOUNT      
Beginning Balance $10,783.06
Deposits and Additions 52 48,894.70
Checks Paid 6 -6,270.50
ATM & Debit Card Withdrawals 5 -1,920.00    
Electronic Withdrawals 17 -49,146.42        
Fees 1 -2.50
Ending Balance 81 $2,338.34
*end*summary
"""

print("=== EXTRACTING CHASE SUMMARY ===\n")

#Extract summary section
summary_match = re.search(r'\*start\*summary.*?\*end\*summary', chase_summary_text, re.DOTALL)
if summary_match:
    summary_section = summary_match.group(0)
    print("Found summary section:")
    print(summary_section)
    print("\n")
    
    # Parse the rows
    lines = summary_section.split('\n')
    for line in lines:
        # Skip markers and headers
        if '*start*' in line or '*end*' in line or 'CHECKING SUMMARY' in line or 'INSTANCES AMOUNT' in line:
            continue
        if not line.strip():
            continue
            
        print(f"Line: {repr(line)}")
        
        # Try to extract: Category [instances] [amount]
        # Pattern: text followed by optional number, then amount with $ or -
        match = re.search(r'^(.+?)\s+(\d+)?\s*([\$\-]?[\d,]+\.?\d*)$', line.strip())
        if match:
            category = match.group(1).strip()
            instances = match.group(2) if match.group(2) else ""
            amount = match.group(3).strip()
            print(f"  → Category: {category}, Instances: {instances}, Amount: {amount}\n")
