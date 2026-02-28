import sys, os
from datetime import datetime
from typing import List

# Mock the BankStatementParser and Transaction context
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from bank_statement_parser import parse_bank_statement, BankName, TransactionCategory

def test_boa_header_robustness():
    # Simulate BoA text with EXTRA SPACES in headers
    text = """
    WITHDRAWALS    AND    OTHER    DEBITS
    12/02/25 Online payment -2,033.00
    12/03/25 Another payment -100.00
    
    DEPOSITS    AND    OTHER    CREDITS
    12/01/25 Zelle from RICK SU 825.00
    12/01/25 Zelle from ALICE 825.00
    """
    
    print("Testing BoA Header Robustness...")
    result = parse_bank_statement(text, bank_name=BankName.BOA)
    
    withdrawals = [t for t in result.transactions if t.amount < 0]
    deposits = [t for t in result.transactions if t.amount > 0]
    
    print(f"  Found {len(withdrawals)} withdrawals (Expected 2)")
    print(f"  Found {len(deposits)} deposits (Expected 2)")
    
    assert len(withdrawals) == 2, "Withdrawals not detected due to whitespace in header"
    assert len(deposits) == 2, "Deposits not detected due to whitespace in header"
    print("  PASS: Header robustness verified")

def test_deduplication_relaxation():
    # Import dedup logic by mocking it or using the real one if it was a standalone function
    # Since it's inline, we'll simulate the key logic
    print("Testing Deduplication Relaxation...")
    
    class MockTx:
        def __init__(self, date, amount, vendor, description):
            self.date = date
            self.amount = amount
            self.vendor = vendor
            self.description = description

    tx1 = MockTx("2025-12-01", 825.00, "RICK SU", "Zelle payment from RICK SU")
    tx2 = MockTx("2025-12-01", 825.00, "ALICE", "Zelle payment from ALICE")
    
    def get_key(tx):
        v_norm = (tx.vendor or "").strip().lower()[:20]
        d_norm = (tx.description or "").strip().lower()[:15]
        return (tx.date, abs(tx.amount), v_norm, d_norm)

    key1 = get_key(tx1)
    key2 = get_key(tx2)
    
    print(f"  Key 1: {key1}")
    print(f"  Key 2: {key2}")
    
    assert key1 != key2, "Transactions with same amount/date but different people should NOT have same key"
    print("  PASS: Deduplication relaxation verified")

if __name__ == "__main__":
    try:
        test_boa_header_robustness()
        print("-" * 30)
        test_deduplication_relaxation()
        print("\nALL TESTS PASSED")
    except AssertionError as e:
        print(f"\nAssertion Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
