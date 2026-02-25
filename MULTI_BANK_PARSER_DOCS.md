# Multi-Bank Statement Parser - Documentation

## Overview

The bank statement parser has been refactored into a clean, extensible multi-bank architecture that supports parsing statements from multiple financial institutions while maintaining backward compatibility with existing Chase parsing functionality.

## Supported Banks

The parser currently supports the following banks:

1. **Chase** - JPMorgan Chase Business Complete Checking
2. **American Express (Amex)** - Credit card statements
3. **BMO** - BMO Elite Business Checking
4. **Bank of America (BoA)** - Business checking accounts
5. **Fifth Third Bank** - Business accounts
6. **US Bank** - Silver Business Checking

## Architecture

### Core Components

#### 1. Unified Data Models

**Transaction Model**
```python
@dataclass
class Transaction:
    date: datetime                          # Transaction date
    description: str                        # Transaction description
    amount: float                           # Amount (negative = debit, positive = credit)
    type: TransactionType                   # DEBIT or CREDIT
    bank_name: str                          # Bank identifier
    account_type: Optional[str]             # Account type (e.g., "checking", "credit_card")
    category: Optional[TransactionCategory] # Category (deposit, withdrawal, check, etc.)
    running_balance: Optional[float]        # Running balance after transaction
    raw_line: str                           # Original line from statement
    check_number: Optional[str]             # Check number if applicable
    needs_review: bool                      # Flag for manual review
```

**ParsedStatement Model**
```python
@dataclass
class ParsedStatement:
    bank_name: str                          # Bank identifier
    transactions: List[Transaction]         # List of all transactions
    account_number: Optional[str]           # Account number
    account_type: Optional[str]             # Account type
    statement_period: Optional[StatementPeriod]  # Statement date range
    beginning_balance: Optional[float]      # Opening balance
    ending_balance: Optional[float]         # Closing balance
    errors: List[str]                       # Any parsing errors
    metadata: Dict                          # Additional metadata
```

#### 2. Bank Detection

The `detect_bank()` function automatically identifies the bank from statement text:

```python
def detect_bank(text: str) -> BankName:
    """
    Detect bank from statement text by inspecting first page.
    Returns: BankName enum value (CHASE, AMEX, BMO, BOA, FIFTH_THIRD, US_BANK, UNKNOWN)
    """
```

**Detection Keywords:**
- **Chase**: "JPMORGAN CHASE" or "CHASE BUSINESS COMPLETE"
- **Amex**: "AMERICAN EXPRESS" or "PAYMENT DUE DATE"
- **BMO**: "BMO" + ("BMO ELITE BUSINESS" or "BANKING SUMMARY")
- **BoA**: "BANK OF AMERICA" + "YOUR CHECKING ACCOUNT"
- **Fifth Third**: "FIFTH THIRD BANK"
- **US Bank**: "U.S. BANK SILVER" or "USBANK"

#### 3. Parser Interface

All bank parsers inherit from the `BankStatementParser` abstract base class:

```python
class BankStatementParser(ABC):
    @abstractmethod
    def parse(self, text: str) -> ParsedStatement:
        """Parse statement text into structured data"""
        pass
```

**Helper Methods Available to All Parsers:**
- `_parse_date(date_str, year)` - Parse date with various formats
- `_parse_amount(amount_str)` - Parse monetary amounts
- `_clean_description(desc)` - Normalize transaction descriptions

#### 4. Bank-Specific Parsers

Each bank has its own dedicated parser class:

- `ChaseParser` - Handles Chase statements with sections like "DEPOSITS AND ADDITIONS", "CHECKS PAID", etc.
- `BMOParser` - Parses BMO's "MONTHLY ACTIVITY DETAILS" format
- `BoAParser` - Handles Bank of America's multi-section format
- `FifthThirdParser` - Parses "DEPOSITS / CREDITS" and "WITHDRAWALS / DEBITS" sections
- `USBankParser` - Handles "OTHER DEPOSITS" and "OTHER WITHDRAWALS"
- `AmexParser` - Parses credit card statements with "PURCHASES", "PAYMENTS AND CREDITS"

## Usage

### Basic Usage

```python
from bank_statement_parser import parse_bank_statement

# Read PDF text (using pdfplumber or similar)
with pdfplumber.open("statement.pdf") as pdf:
    text = "\n".join(page.extract_text() for page in pdf.pages)

# Parse statement (auto-detects bank)
statement = parse_bank_statement(text)

# Access parsed data
print(f"Bank: {statement.bank_name}")
print(f"Transactions: {len(statement.transactions)}")
print(f"Beginning Balance: ${statement.beginning_balance}")
print(f"Ending Balance: ${statement.ending_balance}")

# Iterate through transactions
for trans in statement.transactions:
    print(f"{trans.date} | {trans.description} | ${trans.amount}")
```

### Specify Bank Manually

```python
from bank_statement_parser import parse_bank_statement, BankName

# Force specific parser
statement = parse_bank_statement(text, bank_name=BankName.CHASE)
```

### Legacy Compatibility

For backward compatibility with existing code:

```python
from bank_statement_parser import LegacyBankStatementParser

parser = LegacyBankStatementParser()
transactions, metadata = parser.parse_statement(lines)

# Returns old format:
# transactions = [{'date': '2024-01-01', 'description': '...', 'amount': 100.0, ...}]
# metadata = {'total_lines': 100, 'total_transactions': 50, ...}
```

## Configuration

Bank-specific configuration is stored in `BANK_CONFIG`:

```python
BANK_CONFIG = {
    BankName.CHASE: {
        "sections": [
            "DEPOSITS AND ADDITIONS",
            "CHECKS PAID",
            "ATM & DEBIT CARD WITHDRAWALS",
            "ELECTRONIC WITHDRAWALS",
            "OTHER WITHDRAWALS",
            "FEES"
        ],
        "transaction_regex": r'^(\d{2}/\d{2})\s+(.+?)\s+([\d,]+\.?\d*)\s*$',
        "balance_section": "DAILY LEDGER BALANCES"
    },
    # ... other banks ...
}
```

## Error Handling

The parser includes comprehensive error handling:

```python
statement = parse_bank_statement(text)

# Check for errors
if statement.errors:
    print("Parsing errors:")
    for error in statement.errors:
        print(f"  - {error}")

# Check for transactions needing review
needs_review = [t for t in statement.transactions if t.needs_review]
if needs_review:
    print(f"{len(needs_review)} transactions need manual review")
```

**Common Error Codes:**
- `no_transactions_parsed` - No valid transactions found in statement
- `no_parser_available` - Unknown bank, no parser implemented
- `parse_error: <message>` - Exception during parsing

## Adding a New Bank

To add support for a new bank:

1. **Create a new parser class:**

```python
class NewBankParser(BankStatementParser):
    def __init__(self):
        super().__init__(BankName.NEW_BANK)
    
    def parse(self, text: str) -> ParsedStatement:
        statement = ParsedStatement(bank_name=BankName.NEW_BANK.value)
        lines = text.split('\n')
        
        # Implement parsing logic here
        # ...
        
        return statement
```

2. **Add bank to BankName enum:**

```python
class BankName(Enum):
    # ... existing banks ...
    NEW_BANK = "new_bank"
```

3. **Add detection logic:**

```python
def detect_bank(text: str) -> BankName:
    first_page = text[:3000].upper()
    
    # ... existing detection ...
    
    # New bank detection
    if "NEW BANK IDENTIFIER" in first_page:
        return BankName.NEW_BANK
```

4. **Register parser:**

```python
BANK_PARSERS: Dict[BankName, BankStatementParser] = {
    # ... existing parsers ...
    BankName.NEW_BANK: NewBankParser(),
}
```

5. **Add configuration:**

```python
BANK_CONFIG = {
    # ... existing config ...
    BankName.NEW_BANK: {
        "sections": ["SECTION HEADER 1", "SECTION HEADER 2"],
        "transaction_regex": r'^(\d{2}/\d{2})\s+(.+?)\s+([\d,]+\.\d{2})\s*$',
    }
}
```

6. **Write tests:**

```python
def test_new_bank_parser():
    text = """
    NEW BANK STATEMENT
    ...
    """
    statement = parse_bank_statement(text)
    assert statement.bank_name == BankName.NEW_BANK.value
    assert len(statement.transactions) > 0
```

## Testing

Run the comprehensive test suite:

```bash
python3 test_multi_bank_parser.py
```

This will test:
- Bank detection for all supported banks
- Transaction parsing for each bank
- Proper categorization (deposits vs withdrawals)
- Amount parsing and sign convention
- Error handling for unknown banks

## Transaction Sign Convention

**Checking/Savings Accounts:**
- Deposits, credits, incoming transfers: **positive** amounts
- Withdrawals, debits, fees, outgoing payments: **negative** amounts

**Credit Card Accounts:**
- Payments, credits, refunds: **positive** amounts
- Purchases, fees, interest charges: **negative** amounts

## Notes

- **Date Parsing**: Handles multiple date formats (MM/DD, MM/DD/YYYY, MMM DD)
- **Amount Parsing**: Supports currency symbols, commas, parentheses for negative amounts
- **Description Cleaning**: Removes extra whitespace and common prefixes
- **Section Detection**: Each bank parser identifies transaction sections by their specific headers
- **Backward Compatible**: Legacy `BankStatementParser` class maintained for existing integrations

## Troubleshooting

**Problem**: No transactions parsed
- Check if bank is correctly detected (`detect_bank()`)
- Verify statement format matches expected sections
- Check if regex patterns match the actual line format

**Problem**: Wrong transaction amounts or signs
- Verify section headers are correctly identified
- Check if transaction type (debit/credit) is properly set
- Review amount parsing logic for special formats

**Problem**: Unknown bank detected
- Add bank-specific keywords to `detect_bank()`
- Implement new parser class
- Register parser in `BANK_PARSERS` dictionary

## Future Enhancements

Potential improvements:
- Support for international banks and currencies
- OCR pre-processing for scanned statements
- Machine learning for transaction categorization
- Support for multi-currency transactions
- Enhanced duplicate detection
- Statement validation (verify balances match)
