
import sys
import os
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from datetime import datetime

# Mock streamlit before any imports from bank_data_analysis
mock_st = MagicMock()
mock_st.session_state = MagicMock()
mock_st.session_state.user_id = 'test_user'
mock_st.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
sys.modules['streamlit'] = mock_st

# Mock json.dump to avoid MagicMock serialization errors in signup_user
import json
json.dump = MagicMock()

# Define necessary classes for mocking since we can't easily import them without side effects
@dataclass
class NewTransaction:
    date: datetime
    description: str
    amount: float
    type: str
    bank_name: str
    category: Optional[str] = None
    raw_line: Optional[str] = None

@dataclass
class StatementPeriod:
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None

@dataclass
class ParsedStatement:
    bank_name: str
    transactions: List[NewTransaction] = field(default_factory=list)
    account_number: Optional[str] = None
    account_type: Optional[str] = None
    statement_period: Optional[StatementPeriod] = None
    beginning_balance: Optional[float] = None
    ending_balance: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

# Mock bank_statement_parser
mock_bsp = MagicMock()
mock_bsp.TransactionType = MagicMock()
mock_bsp.TransactionCategory = MagicMock()
sys.modules['bank_statement_parser'] = mock_bsp

import bank_data_analysis

def verify_fix():
    print("Starting verification of ParsedStatement fix...")
    
    # 1. Prepare mock data
    tx = NewTransaction(
        date=datetime(2024, 12, 17),
        description="Test Merchant 123-456",
        amount=-42.50,
        type="debit",
        bank_name="amex"
    )
    
    period = StatementPeriod(
        from_date=datetime(2024, 12, 1),
        to_date=datetime(2024, 12, 31)
    )
    
    parsed_result = ParsedStatement(
        bank_name="amex",
        transactions=[tx],
        statement_period=period,
        beginning_balance=1000.0,
        ending_balance=957.5,
        errors=[]
    )
    
    # Mock the parser function
    bank_data_analysis.parse_bank_statement = MagicMock(return_value=parsed_result)
    
    # 2. Run the function under test
    print("Calling parse_with_multi_bank_parser...")
    txs, meta = bank_data_analysis.parse_with_multi_bank_parser(["line 1"], "test.pdf")
    
    if txs is None:
        print("FAILED: Parser returned None. An exception likely occurred.")
        return False
        
    print(f"SUCCESS: Parser returned {len(txs)} transactions.")
    
    # 3. Verify metadata mapping
    print("\nVerifying metadata mapping:")
    expected_period = "12/01/2024 to 12/31/2024"
    print(f"  Bank: {meta.get('bank')} (Expected: amex)")
    print(f"  Period: {meta.get('statement_period')} (Expected: {expected_period})")
    print(f"  Opening Balance: {meta.get('opening_balance')} (Expected: 1000.0)")
    print(f"  Closing Balance: {meta.get('closing_balance')} (Expected: 957.5)")
    
    if meta.get('bank') == "amex" and \
       meta.get('statement_period') == expected_period and \
       meta.get('opening_balance') == 1000.0 and \
       meta.get('closing_balance') == 957.5:
        print("\nALL METADATA MAPPINGS CORRECT!")
        return True
    else:
        print("\nMETADATA MAPPING MISMATCH!")
        return False

if __name__ == "__main__":
    if verify_fix():
        print("\n=== VERIFICATION PASSED ===")
        sys.exit(0)
    else:
        print("\n=== VERIFICATION FAILED ===")
        sys.exit(1)
