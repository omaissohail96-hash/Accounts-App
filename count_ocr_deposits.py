import re

def count_ocr_deposits():
    with open(r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App\ocr_dumps\BoA.txt", 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 1. Count all 375.00
    z375 = re.findall(r'375\.00', text)
    print(f"Total occurrences of '375.00' in OCR: {len(z375)}")
    
    # 2. Extract deposit section
    sections = re.split(r'(?i)(Deposits and other credits|Withdrawals and other debits|Daily ledger balances)', text)
    deposit_text = ""
    for i in range(len(sections)):
        if "Deposits and other credits" in sections[i]:
            if i+1 < len(sections):
                deposit_text += sections[i+1]
    
    # 3. Count lines starting with date in deposit sections
    date_lines = []
    for line in deposit_text.split('\n'):
        line = line.strip()
        if re.match(r'^\d{1,2}/\d{1,2}', line):
            date_lines.append(line)
    
    print(f"Total lines starting with a date in OCR deposit section: {len(date_lines)}")
    for i, line in enumerate(date_lines):
        print(f"{i+1:3}: {line}")

if __name__ == "__main__":
    count_ocr_deposits()
