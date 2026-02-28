import re

def hunt_everything():
    path = r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App\ocr_dumps\BoA.txt"
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print("Potential Transaction Lines (Date + Money):")
    for i, line in enumerate(lines):
        line = line.strip()
        has_date = re.search(r'\d{1,2}/\d{1,2}', line)
        has_money = re.search(r'[\d, ]+\.\d{2}', line)
        
        if has_date and has_money:
            print(f"{i+1:3}: {line}")
        elif has_money and any(k in line.upper() for k in ["ZELLE", "TRANSFER", "CHECK", "MID"]):
            print(f"{i+1:3}: [AMT ONLY?] {line}")

if __name__ == "__main__":
    hunt_everything()
