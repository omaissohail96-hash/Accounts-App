
import pandas as pd
from dataclasses import dataclass
import sys
import os
from unittest.mock import MagicMock

# Create a mock for streamlit
class MockStreamlit:
    def __init__(self):
        self.session_state = MagicMock()
        self.session_state.user_id = "test_user_id"
        self.session_state.active_business = "Test Business"
        self.session_state.user = {"name": "Test User", "id": "test_user_id"}
        # Allow dictionary access and attribute access
        self.session_state.__getitem__ = lambda s, k: getattr(s, k)
        self.session_state.__setitem__ = lambda s, k, v: setattr(s, k, v)
        self.session_state.get = lambda k, d=None: getattr(self.session_state, k, d)
        
        # Mock common streamlit functions
        self.set_page_config = MagicMock()
        self.columns = lambda *args, **kwargs: [MagicMock() for _ in range(args[0] if isinstance(args[0], int) else len(args[0]))]
        self.sidebar = MagicMock()
        self.markdown = MagicMock()
        self.title = MagicMock()
        self.header = MagicMock()
        self.subheader = MagicMock()
        self.write = MagicMock()
        self.button = MagicMock(return_value=False)
        self.text_input = MagicMock(return_value="Test Input")
        self.selectbox = MagicMock(return_value="Option 1")
        self.file_uploader = MagicMock(return_value=None)
        self.success = MagicMock()
        self.error = MagicMock()
        self.warning = MagicMock()
        self.info = MagicMock()
        self.stop = MagicMock()
        self.rerun = MagicMock()
        self.divider = MagicMock()
        self.expander = MagicMock()
        self.tabs = MagicMock(return_value=[MagicMock(), MagicMock()])
        self.dataframe = MagicMock()
        self.image = MagicMock()
        self.balloons = MagicMock()
        self.snow = MagicMock()
        self.form = MagicMock()
        self.empty = MagicMock()
        self.container = MagicMock()
        self.metric = MagicMock()
        self.caption = MagicMock()
        self.code = MagicMock()
        self.status = MagicMock()
        self.progress = MagicMock()
        self.spinner = MagicMock()
        self.toast = MagicMock()
        
        # Add any other attributes accessed dynamically via __getattr__
        self.__getattr__ = MagicMock()

# Inject the mock into sys.modules
sys.modules["streamlit"] = MockStreamlit()


# Now import the module
sys.path.append(os.getcwd())
if "streamlit" in sys.modules and not isinstance(sys.modules["streamlit"], MockStreamlit):
    del sys.modules["streamlit"]
    sys.modules["streamlit"] = MockStreamlit()

try:
    from bank_data_analysis import get_sub_summary
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

@dataclass
class MockTransaction:
    date: pd.Timestamp
    amount: float
    description: str
    section: str = ""
    category: str = ""
    source: str = "BANK"
    vendor: str = ""

def test_summary():
    transactions = [
        MockTransaction(
            date=pd.Timestamp("2023-01-01"), 
            amount=1000.0, 
            description="Opening Deposit"
        ), 
        MockTransaction(
            date=pd.Timestamp("2023-01-02"),
            amount=500.0, 
            description="Deposit 1"
        ),
        MockTransaction(
            date=pd.Timestamp("2023-01-03"),
            amount=-100.0, 
            description="Check 101", 
            section="CHECKS"
        ),
        MockTransaction(
            date=pd.Timestamp("2023-01-04"),
            amount=-50.0, 
            description="ATM Withdrawal"
        ),
        MockTransaction(
            date=pd.Timestamp("2023-01-05"),
            amount=-25.0, 
            description="Monthly Maintenance Fee"
        ),
        MockTransaction(
            date=pd.Timestamp("2023-01-06"),
            amount=-10.0, 
            description="Netflix", 
            category="Subscription"
        ),
    ]
    
    opening_balance = 1000.0
    
    df = get_sub_summary(transactions, opening_balance)
    
    print("\nGenerated Summary Table:")
    print(df)
    
    # Assertions
    assert "Type" in df.columns
    # Check if updated columns exist
    # Note: If verify fails here, it implies get_sub_summary update wasn't applied correctly or I'm checking wrong keys
    # The updated get_sub_summary should have 'INSTANCES' and 'AMOUNT'
    
    rows = df.set_index("Type").to_dict(orient="index")
    
    # Verify values
    
    # Beginning Balance
    assert rows["Beginning Balance"]["AMOUNT"] == "$1,000.00", f"Expected $1,000.00, got {rows['Beginning Balance']['AMOUNT']}"
    
    # Deposits
    assert rows["Deposits and Additions"]["INSTANCES"] == 1
    assert rows["Deposits and Additions"]["AMOUNT"] == "$500.00"
    
    # Checks
    assert rows["Checks Paid"]["INSTANCES"] == 1
    assert rows["Checks Paid"]["AMOUNT"] == "$-100.00"
    
    # ATM
    # Note: 'ATM & Debit Card Withdrawals' is the expected key
    assert rows["ATM & Debit Card Withdrawals"]["INSTANCES"] == 1
    assert rows["ATM & Debit Card Withdrawals"]["AMOUNT"] == "$-50.00"
    
    # Fees - 25.00
    assert rows["Fees"]["INSTANCES"] == 1
    assert rows["Fees"]["AMOUNT"] == "$-25.00"
    
    # Electronic - 10.00
    assert rows["Electronic Withdrawals"]["INSTANCES"] == 1
    assert rows["Electronic Withdrawals"]["AMOUNT"] == "$-10.00"
    
    # Ending Balance
    assert rows["Ending Balance"]["AMOUNT"] == "$1,315.00"
    
    print("\n✅ Verification Successful!")

if __name__ == "__main__":
    try:
        test_summary()
    except Exception as e:
        print(f"\n❌ Verification Failed: {e}")
        import traceback
        traceback.print_exc()
