
import sys
import os
import re
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.getcwd())

import bank_statement_parser as bsp

def test_detection():
    print("=== Testing Bank Detection ===")
    test_cases = [
        ("JPMORGAN CHASE BANK\nACCOUNT SUMMARY", bsp.BankName.CHASE),
        ("BANK OF AMERICA\nPreferred Rewards", bsp.BankName.BOA),
        ("BofA Business Advantage", bsp.BankName.BOA),
        ("BMO Harris Bank\nBanking Summary", bsp.BankName.BMO),
        ("AMERICAN EXPRESS\nMembership Rewards", bsp.BankName.AMEX),
        ("FIFTH THIRD BANK\nWithdrawals / Debits", bsp.BankName.FIFTH_THIRD),
        ("U.S. BANK\nElectronic Deposits", bsp.BankName.US_BANK),
    ]
    
    for text, expected in test_cases:
        detected = bsp.detect_bank(text)
        status = "PASS" if detected == expected else f"FAIL (got {detected})"
        clean_text = text[:30].replace('\n', ' ')
        print(f"Text: {clean_text}... -> {status}")

def test_amex_parsing():
    print("\n=== Testing AMEX Parsing ===")
    amex_text = """
    AMERICAN EXPRESS
    PAYMENTS AND CREDITS
    01/15/24  PAYMENT RECEIVED - THANK YOU  -$500.00
    NEW CHARGES
    01/16/24  AMAZON.COM*AMZN      $50.00
    01/17/24  UBER TRIP            $15.25
    01/18/24  STARBUCKS            $5.50
    """
    result = bsp.parse_bank_statement(amex_text, bsp.BankName.AMEX)
    print(f"AMEX: Extracted {len(result.transactions)} transactions")
    for tx in result.transactions:
        print(f"  {tx.date.strftime('%m/%d/%y')} {tx.description[:20]:<20} {tx.amount:>8.2f} {tx.type}")

def test_boa_parsing():
    print("\n=== Testing BOA Parsing ===")
    boa_text = """
    BANK OF AMERICA
    DEPOSITS AND OTHER CREDITS
    01/15/2024  Online Banking transfer from CHECKING  1,500.00
    WITHDRAWALS AND OTHER DEBITS
    01/20  ATM WITHDRAWAL  -200.00
    01/21  CHECK #123      -100.00
    """
    result = bsp.parse_bank_statement(boa_text, bsp.BankName.BOA)
    print(f"BOA: Extracted {len(result.transactions)} transactions")
    for tx in result.transactions:
        print(f"  {tx.date.strftime('%m/%d/%y')} {tx.description[:20]:<20} {tx.amount:>8.2f} {tx.type}")

def test_fifth_third_parsing():
    print("\n=== Testing Fifth Third Parsing ===")
    ft_text = """
    FIFTH THIRD BANK
    WITHDRAWALS / DEBITS
    01/15  AMAZON MARKETPLACE  50.00
    DEPOSITS / CREDITS
    01/20  PAYROLL DEPOSIT     1,500.00
    """
    result = bsp.parse_bank_statement(ft_text, bsp.BankName.FIFTH_THIRD)
    print(f"Fifth Third: Extracted {len(result.transactions)} transactions")
    for tx in result.transactions:
        print(f"  {tx.date.strftime('%m/%d/%y')} {tx.description[:20]:<20} {tx.amount:>8.2f} {tx.type}")

if __name__ == "__main__":
    test_detection()
    test_amex_parsing()
    test_boa_parsing()
    test_fifth_third_parsing()
