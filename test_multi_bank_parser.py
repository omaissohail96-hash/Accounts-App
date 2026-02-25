"""
Test suite for multi-bank statement parser
Tests each bank parser with sample data
"""
import sys
from datetime import datetime
from bank_statement_parser import (
    parse_bank_statement,
    detect_bank,
    BankName,
    TransactionType,
    ChaseParser,
    BMOParser,
    BoAParser,
    FifthThirdParser,
    USBankParser,
    AmexParser
)


def test_chase_parser():
    """Test Chase bank statement parsing"""
    print("\n=== TESTING CHASE PARSER ===")
    
    chase_text = """
JPMorgan Chase Bank, N.A.
Chase Business Complete Checking
Account Number: 123456789
Statement Period: 01/01/2024 TO 01/31/2024

BEGINNING BALANCE                                    $5,000.00

DEPOSITS AND ADDITIONS
DATE      DESCRIPTION                                 AMOUNT
01/05     ACH CREDIT PAYROLL INC                     2,500.00
01/15     MOBILE DEPOSIT                             1,200.50
01/20     WIRE TRANSFER RECEIVED                     3,000.00

CHECKS PAID
DATE      CHECK#    DESCRIPTION                       AMOUNT
01/08     1001      CHECK                             500.00
01/12     1002      CHECK                             250.00

ATM & DEBIT CARD WITHDRAWALS
DATE      DESCRIPTION                                 AMOUNT
01/10     DEBIT CARD PURCHASE #1234                  125.75
01/18     ATM WITHDRAWAL #5678                       100.00

ELECTRONIC WITHDRAWALS
DATE      DESCRIPTION                                 AMOUNT
01/06     ACH DEBIT UTILITY COMPANY                  150.00
01/22     ACH DEBIT INTERNET SERVICE                  80.00

FEES
DATE      DESCRIPTION                                 AMOUNT
01/31     MONTHLY SERVICE FEE                         15.00

ENDING BALANCE                                      $10,580.75
"""
    
    # Test bank detection
    detected = detect_bank(chase_text)
    assert detected == BankName.CHASE, f"Expected CHASE, got {detected}"
    print("✓ Bank detection: CHASE")
    
    # Test parsing
    statement = parse_bank_statement(chase_text)
    assert statement.bank_name == BankName.CHASE.value
    print(f"✓ Parsed {len(statement.transactions)} transactions")
    
    # Verify transaction types
    deposits = [t for t in statement.transactions if t.amount > 0]
    withdrawals = [t for t in statement.transactions if t.amount < 0]
    print(f"  - Deposits: {len(deposits)}")
    print(f"  - Withdrawals: {len(withdrawals)}")
    
    # Check specific transactions
    assert len(deposits) == 3, f"Expected 3 deposits, got {len(deposits)}"
    assert len(withdrawals) >= 5, f"Expected at least 5 withdrawals, got {len(withdrawals)}"
    
    # Verify amounts
    assert any(abs(t.amount - 2500.00) < 0.01 for t in deposits), "Missing payroll deposit"
    assert any(abs(t.amount + 500.00) < 0.01 for t in withdrawals), "Missing check payment"
    
    print("✓ Chase parser: PASSED")
    return True


def test_bmo_parser():
    """Test BMO bank statement parsing"""
    print("\n=== TESTING BMO PARSER ===")
    
    bmo_text = """
BMO
BMO ELITE BUSINESS CKG
Account Number: 987654321
Statement Period: January 2024

BANKING SUMMARY
Beginning Balance                                    $8,500.00

MONTHLY ACTIVITY DETAILS
Date         Transaction description              Withdrawal    Deposit      Balance
Jan 03       DIRECT DEPOSIT PAYROLL                              $3,000.00   $11,500.00
Jan 05       DEBIT CARD PURCHASE OFFICE            $125.50                   $11,374.50
Jan 10       CHECK #2001                           $450.00                   $10,924.50
Jan 15       ACH CREDIT VENDOR PAYMENT                           $1,500.00   $12,424.50
Jan 20       MONTHLY SERVICE FEE                    $25.00                   $12,399.50

MONTHLY ACTIVITY DETAILS (CONT'D)
Date         Transaction description              Withdrawal    Deposit      Balance
Jan 25       WIRE TRANSFER OUT                     $2,000.00                 $10,399.50

Ending Balance                                      $10,399.50
"""
    
    # Test bank detection
    detected = detect_bank(bmo_text)
    assert detected == BankName.BMO, f"Expected BMO, got {detected}"
    print("✓ Bank detection: BMO")
    
    # Test parsing
    statement = parse_bank_statement(bmo_text)
    assert statement.bank_name == BankName.BMO.value
    print(f"✓ Parsed {len(statement.transactions)} transactions")
    
    # Verify transactions
    deposits = [t for t in statement.transactions if t.amount > 0]
    withdrawals = [t for t in statement.transactions if t.amount < 0]
    print(f"  - Deposits: {len(deposits)}")
    print(f"  - Withdrawals: {len(withdrawals)}")
    
    assert len(deposits) >= 2, f"Expected at least 2 deposits, got {len(deposits)}"
    assert len(withdrawals) >= 3, f"Expected at least 3 withdrawals, got {len(withdrawals)}"
    
    print("✓ BMO parser: PASSED")
    return True


def test_boa_parser():
    """Test Bank of America statement parsing"""
    print("\n=== TESTING BANK OF AMERICA PARSER ===")
    
    boa_text = """
BANK OF AMERICA
Your checking account summary

Account Number: 555-12345
Statement Period: 01/01/2024 - 01/31/2024

Beginning Balance                                    $7,200.00

DEPOSITS AND OTHER CREDITS
Date      Description                                Amount
01/05     DIRECT DEPOSIT EMPLOYER                   2,800.00
01/12     MOBILE CHECK DEPOSIT                      1,150.00
01/20     ACH CREDIT REFUND                          350.00

WITHDRAWALS AND OTHER DEBITS
Date      Description                                Amount
01/07     DEBIT CARD PURCHASE AMAZON                 125.99
01/14     ACH DEBIT UTILITIES                        200.00
01/28     MONTHLY ACCOUNT FEE                         12.00

CHECKS
Date      Check#   Description                       Amount
01/10     3001     CHECK                              450.00
01/22     3002     CHECK                              275.50

Ending Balance                                      $10,436.51
"""
    
    # Test bank detection
    detected = detect_bank(boa_text)
    assert detected == BankName.BOA, f"Expected BOA, got {detected}"
    print("✓ Bank detection: BOA")
    
    # Test parsing
    statement = parse_bank_statement(boa_text)
    assert statement.bank_name == BankName.BOA.value
    print(f"✓ Parsed {len(statement.transactions)} transactions")
    
    deposits = [t for t in statement.transactions if t.amount > 0]
    withdrawals = [t for t in statement.transactions if t.amount < 0]
    print(f"  - Deposits: {len(deposits)}")
    print(f"  - Withdrawals: {len(withdrawals)}")
    
    print("✓ BoA parser: PASSED")
    return True


def test_fifth_third_parser():
    """Test Fifth Third Bank statement parsing"""
    print("\n=== TESTING FIFTH THIRD BANK PARSER ===")
    
    fifth_third_text = """
FIFTH THIRD BANK
Business Checking Account

Account Number: FT-987654
Statement Period: January 2024

Beginning Balance: $6,500.00

DEPOSITS / CREDITS
Date      Description                                Amount
01/04     ACH DEPOSIT CUSTOMER PAYMENT              3,200.00
01/15     WIRE TRANSFER RECEIVED                    1,800.00
01/25     MOBILE DEPOSIT                             975.00

WITHDRAWALS / DEBITS
Date      Description                                Amount
01/08     DEBIT CARD PURCHASE                        189.50
01/17     ACH DEBIT RENT PAYMENT                   2,000.00
01/30     SERVICE CHARGE                              18.00

Ending Balance: $10,267.50
"""
    
    # Test bank detection
    detected = detect_bank(fifth_third_text)
    assert detected == BankName.FIFTH_THIRD, f"Expected FIFTH_THIRD, got {detected}"
    print("✓ Bank detection: FIFTH_THIRD")
    
    # Test parsing
    statement = parse_bank_statement(fifth_third_text)
    assert statement.bank_name == BankName.FIFTH_THIRD.value
    print(f"✓ Parsed {len(statement.transactions)} transactions")
    
    deposits = [t for t in statement.transactions if t.amount > 0]
    withdrawals = [t for t in statement.transactions if t.amount < 0]
    print(f"  - Deposits: {len(deposits)}")
    print(f"  - Withdrawals: {len(withdrawals)}")
    
    print("✓ Fifth Third parser: PASSED")
    return True


def test_us_bank_parser():
    """Test US Bank statement parsing"""
    print("\n=== TESTING US BANK PARSER ===")
    
    us_bank_text = """
U.S. Bank Silver Business Checking

Account Number: USB-123456
Statement Period: 01/01/2024 - 01/31/2024

Beginning Balance                                    $9,100.00

OTHER DEPOSITS
Date      Description                                Amount
01/06     ACH CREDIT BUSINESS REVENUE               4,500.00
01/18     DEPOSIT                                   2,300.00

OTHER WITHDRAWALS
Date      Description                                Amount
01/09     DEBIT CARD PURCHASE SUPPLIES               425.00
01/15     ACH PAYMENT VENDOR                       1,200.00
01/29     MONTHLY FEE                                 25.00

ENDING BALANCE                                      $14,250.00
"""
    
    # Test bank detection
    detected = detect_bank(us_bank_text)
    assert detected == BankName.US_BANK, f"Expected US_BANK, got {detected}"
    print("✓ Bank detection: US_BANK")
    
    # Test parsing
    statement = parse_bank_statement(us_bank_text)
    assert statement.bank_name == BankName.US_BANK.value
    print(f"✓ Parsed {len(statement.transactions)} transactions")
    
    deposits = [t for t in statement.transactions if t.amount > 0]
    withdrawals = [t for t in statement.transactions if t.amount < 0]
    print(f"  - Deposits: {len(deposits)}")
    print(f"  - Withdrawals: {len(withdrawals)}")
    
    print("✓ US Bank parser: PASSED")
    return True


def test_amex_parser():
    """Test American Express statement parsing"""
    print("\n=== TESTING AMERICAN EXPRESS PARSER ===")
    
    amex_text = """
AMERICAN EXPRESS
Business Card Statement

Account Number: XXXX-XXXXX-12345
Statement Period: 01/01/2024 - 01/31/2024
PAYMENT DUE DATE: 02/22/2024

Previous Balance                                     $2,500.00

PAYMENTS AND CREDITS
Date      Description                                Amount
01/05     PAYMENT RECEIVED - THANK YOU              2,500.00
01/20     CREDIT ADJUSTMENT REFUND                   150.00

PURCHASES
Date      Description                                Amount
01/08     AMAZON.COM OFFICE SUPPLIES                 245.75
01/12     DELTA AIRLINES BUSINESS TRAVEL           1,200.00
01/18     STAPLES BUSINESS SUPPLIES                  185.50

FEES
Date      Description                                Amount
01/31     LATE FEE                                    35.00

NEW BALANCE                                          $916.25
"""
    
    # Test bank detection
    detected = detect_bank(amex_text)
    assert detected == BankName.AMEX, f"Expected AMEX, got {detected}"
    print("✓ Bank detection: AMEX")
    
    # Test parsing
    statement = parse_bank_statement(amex_text)
    assert statement.bank_name == BankName.AMEX.value
    print(f"✓ Parsed {len(statement.transactions)} transactions")
    
    credits = [t for t in statement.transactions if t.amount > 0]
    charges = [t for t in statement.transactions if t.amount < 0]
    print(f"  - Credits/Payments: {len(credits)}")
    print(f"  - Charges/Fees: {len(charges)}")
    
    print("✓ Amex parser: PASSED")
    return True


def test_unknown_bank():
    """Test handling of unknown bank"""
    print("\n=== TESTING UNKNOWN BANK DETECTION ===")
    
    unknown_text = """
Some Random Bank
Account Statement

This bank is not in our supported list.
"""
    
    detected = detect_bank(unknown_text)
    assert detected == BankName.UNKNOWN, f"Expected UNKNOWN, got {detected}"
    print("✓ Unknown bank detected correctly")
    
    statement = parse_bank_statement(unknown_text)
    assert statement.bank_name == BankName.UNKNOWN.value
    assert "no_parser_available" in statement.errors
    print("✓ Unknown bank handled gracefully")
    
    return True


def run_all_tests():
    """Run all bank parser tests"""
    print("="*60)
    print("MULTI-BANK STATEMENT PARSER TEST SUITE")
    print("="*60)
    
    tests = [
        test_chase_parser,
        test_bmo_parser,
        test_boa_parser,
        test_fifth_third_parser,
        test_us_bank_parser,
        test_amex_parser,
        test_unknown_bank
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
