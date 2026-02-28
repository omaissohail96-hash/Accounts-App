
import sys
from unittest.mock import MagicMock

# Mock streamlit
mock_st = MagicMock()
mock_st.session_state = MagicMock()
mock_st.session_state.user_id = "test_user"
mock_st.tabs.return_value = (MagicMock(), MagicMock())
mock_st.columns.return_value = (MagicMock(), MagicMock())
sys.modules["streamlit"] = mock_st

from bank_data_analysis import parse_with_multi_bank_parser, Transaction
from bank_statement_parser import ParsedStatement, Transaction as NewTransaction, TransactionType, StatementPeriod
from datetime import datetime

def reproduce_error():
    # Create a mock ParsedStatement
    tx = NewTransaction(
        date=datetime(2024, 1, 1),
        description="Test transaction",
        amount=-100.0,
        type=TransactionType.DEBIT,
        bank_name="amex"
    )
    
    parsed_result = ParsedStatement(
        bank_name="amex",
        transactions=[tx],
        beginning_balance=1000.0,
        ending_balance=900.0
    )
    
    # We need to mock parse_bank_statement to return our parsed_result
    import bank_data_analysis
    import bank_statement_parser
    bank_statement_parser.parse_bank_statement = MagicMock(return_value=parsed_result)
    
    print("Running parse_with_multi_bank_parser...")
    try:
        txs, meta = parse_with_multi_bank_parser(["some text"], "test.pdf")
        if txs is None:
            print("Parser returned None (likely due to caught exception)")
        else:
            print("Parser succeeded unexpectedly")
    except AttributeError as e:
        print(f"Caught expected AttributeError: {e}")
    except Exception as e:
        print(f"Caught unexpected Exception: {e}")

if __name__ == "__main__":
    reproduce_error()
