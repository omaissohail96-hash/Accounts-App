from bank_statement_parser import ChaseParser, BANK_LAYOUTS, BankName, TransactionCategory, TransactionType

parser = ChaseParser()
layout = BANK_LAYOUTS[BankName.CHASE]
year = 2025

lines_to_test = [
    ("CHECK", "268 ^ 11/03 $500.00"),
    ("ATM", "11/10 ATM Withdrawal 11/10 4790 W University DR Prosper TX Card 3387 $400.00"),
]

for section, line in lines_to_test:
    print(f"\n--- Testing Line: {line}")
    if section == "CHECK":
        info = (TransactionCategory.CHECK, TransactionType.DEBIT)
    else:
        info = (TransactionCategory.ATM, TransactionType.DEBIT)
        
    res = parser._parse_line_with_layout(line, info, year, layout)
    print(f"Result: {res}")
