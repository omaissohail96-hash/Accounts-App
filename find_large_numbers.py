import re

def find_large_numbers():
    with open(r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App\ocr_dumps\BoA.txt", 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Extract deposit section
    sections = re.split(r'(?i)(Deposits and other credits|Withdrawals and other debits|Daily ledger balances)', text)
    deposit_text = ""
    for i in range(len(sections)):
        if "Deposits and other credits" in sections[i]:
            if i+1 < len(sections):
                deposit_text += sections[i+1]
    
    # Find all numbers with commas or spaces or dots
    # Look for patterns like 4,100.00 or 4100.00 or 4 100
    numbers = re.findall(r'[\d, ]+\.\d{2}|(?<=\s)\d{4,}(?=\s|$)', deposit_text)
    
    print("Potential Large Amounts in Deposit Section:")
    for num_str in numbers:
        try:
            val = float(re.sub(r'[^\d.]', '', num_str.replace(' ', '')))
            if val >= 500:
                print(f"  Value: {val:10,.2f} | Raw: '{num_str.strip()}'")
        except:
            continue

if __name__ == "__main__":
    find_large_numbers()
