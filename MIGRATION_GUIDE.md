# Migration Guide: Old Parser → Multi-Bank Parser

## Overview

This guide helps you migrate from the old generic bank statement parser to the new multi-bank parser architecture.

## What Changed?

### Before (Old Parser)
- Single generic parser tried to handle all banks
- Basic section detection with common headers
- Limited bank-specific logic
- Simple Transaction model

### After (New Parser)
- Dedicated parser for each supported bank
- Automatic bank detection
- Bank-specific parsing logic
- Enhanced Transaction model with more metadata
- Better error handling

## Breaking Changes

### 1. Transaction Model Changes

**Old Transaction Model:**
```python
@dataclass
class Transaction:
    date: Optional[str]              # String format
    description: str
    amount: float
    transaction_type: str            # 'deposit' or 'withdrawal'
    vendor: Optional[str]
    category: Optional[str]
    needs_review: bool
    raw_line: str
    line_number: Optional[int]
    account_code: Optional[str]
```

**New Transaction Model:**
```python
@dataclass
class Transaction:
    date: datetime                   # datetime object (CHANGED)
    description: str
    amount: float
    type: TransactionType            # Enum: DEBIT or CREDIT (CHANGED)
    bank_name: str                   # NEW: Bank identifier
    account_type: Optional[str]      # NEW
    category: Optional[TransactionCategory]  # NEW: Enum
    running_balance: Optional[float] # NEW
    raw_line: str
    check_number: Optional[str]      # NEW
    needs_review: bool
```

**Key Differences:**
- `date` is now a `datetime` object instead of string
- `transaction_type` renamed to `type` and uses enum
- Added `bank_name`, `account_type`, `running_balance`, `check_number`
- `category` is now an enum instead of string
- Removed `line_number`, `vendor`, `account_code`

### 2. Parser Interface Changes

**Old Way:**
```python
parser = BankStatementParser()
transactions, metadata = parser.parse_statement(lines)  # Pass lines array
```

**New Way:**
```python
from bank_statement_parser import parse_bank_statement

statement = parse_bank_statement(text)  # Pass full text
# statement.transactions # List of Transaction objects
# statement.metadata     # Dict of metadata
```

## Migration Strategies

### Strategy 1: Use Legacy Compatibility Layer (Recommended for Quick Migration)

The new parser includes a `LegacyBankStatementParser` class that maintains the old API:

```python
from bank_statement_parser import LegacyBankStatementParser

# Your existing code
parser = LegacyBankStatementParser()  # Drop-in replacement
transactions, metadata = parser.parse_statement(lines)

# Output format matches old parser
# transactions = [{'date': '2024-01-01', 'description': '...', 'amount': 100.0, ...}]
```

**Pros:**
- Minimal code changes
- Works with existing infrastructure
- Can migrate gradually

**Cons:**
- Doesn't take advantage of new features
- Date is returned as string, not datetime
- Missing new metadata fields

### Strategy 2: Full Migration (Recommended for New Features)

Migrate to the new API to access all features:

**Before:**
```python
from bank_statement_parser import BankStatementParser

parser = BankStatementParser()
transactions, metadata = parser.parse_statement(lines)

for trans in transactions:
    date_str = trans.date  # String
    amount = trans.amount
    trans_type = trans.transaction_type  # 'deposit' or 'withdrawal'
```

**After:**
```python
from bank_statement_parser import parse_bank_statement

# Join lines back to text if you're starting from lines
text = '\n'.join(lines)

statement = parse_bank_statement(text)  # Auto-detects bank

for trans in statement.transactions:
    date_obj = trans.date  # datetime object
    date_str = trans.date.strftime('%Y-%m-%d')  # Convert to string if needed
    amount = trans.amount
    trans_type = trans.type.value  # 'debit' or 'credit'
    bank = trans.bank_name  # NEW: which bank
```

## Common Migration Patterns

### Pattern 1: Converting Date Strings to datetime

**Old Code:**
```python
date_str = transaction.date  # '2024-01-15'
parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
```

**New Code:**
```python
date_obj = transaction.date  # Already a datetime object
# Use directly for comparisons, sorting, etc.
date_str = date_obj.strftime('%Y-%m-%d')  # Convert to string if needed
```

### Pattern 2: Handling Transaction Types

**Old Code:**
```python
if transaction.transaction_type == 'deposit':
    # Handle deposit
elif transaction.transaction_type == 'withdrawal':
    # Handle withdrawal
```

**New Code:**
```python
from bank_statement_parser import TransactionType

if transaction.type == TransactionType.CREDIT:
    # Handle credit (was 'deposit')
elif transaction.type == TransactionType.DEBIT:
    # Handle debit (was 'withdrawal')

# Or use the value:
if transaction.type.value == 'credit':
    # Handle credit
```

### Pattern 3: Accessing Metadata

**Old Code:**
```python
transactions, metadata = parser.parse_statement(lines)
total_lines = metadata['total_lines']
total_trans = metadata['total_transactions']
```

**New Code:**
```python
statement = parse_bank_statement(text)
total_trans = len(statement.transactions)
bank = statement.bank_name
beginning = statement.beginning_balance
ending = statement.ending_balance
errors = statement.errors
```

### Pattern 4: Filtering Transactions

**Old Code:**
```python
deposits = [t for t in transactions if t.transaction_type == 'deposit']
withdrawals = [t for t in transactions if t.transaction_type == 'withdrawal']
```

**New Code:**
```python
credits = [t for t in statement.transactions if t.amount > 0]
debits = [t for t in statement.transactions if t.amount < 0]

# Or using type:
from bank_statement_parser import TransactionType
credits = [t for t in statement.transactions if t.type == TransactionType.CREDIT]
debits = [t for t in statement.transactions if t.type == TransactionType.DEBIT]
```

## Step-by-Step Migration Example

### Scenario: Migrating a PDF Processing Function

**Old Code:**
```python
def process_bank_statement(pdf_path):
    # Extract text
    with pdfplumber.open(pdf_path) as pdf:
        lines = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines.extend(text.split('\n'))
    
    # Parse with old parser
    from bank_statement_parser import BankStatementParser
    parser = BankStatementParser()
    transactions, metadata = parser.parse_statement(lines)
    
    # Process results
    deposits_total = sum(t.amount for t in transactions if t.transaction_type == 'deposit')
    withdrawals_total = sum(abs(t.amount) for t in transactions if t.transaction_type == 'withdrawal')
    
    return {
        'transactions': transactions,
        'deposits': deposits_total,
        'withdrawals': withdrawals_total,
        'count': metadata['total_transactions']
    }
```

**New Code:**
```python
def process_bank_statement(pdf_path):
    # Extract text (combine pages into single text)
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    
    # Parse with new parser
    from bank_statement_parser import parse_bank_statement
    statement = parse_bank_statement(full_text)  # Auto-detects bank
    
    # Process results
    credits_total = sum(t.amount for t in statement.transactions if t.amount > 0)
    debits_total = sum(abs(t.amount) for t in statement.transactions if t.amount < 0)
    
    return {
        'transactions': statement.transactions,
        'bank': statement.bank_name,  # NEW: Know which bank
        'deposits': credits_total,
        'withdrawals': debits_total,
        'count': len(statement.transactions),
        'beginning_balance': statement.beginning_balance,  # NEW
        'ending_balance': statement.ending_balance,        # NEW
        'errors': statement.errors                         # NEW
    }
```

## Testing Your Migration

### 1. Side-by-Side Comparison

Run both parsers on the same data and compare results:

```python
# Old parser
from bank_statement_parser import LegacyBankStatementParser
old_parser = LegacyBankStatementParser()
old_transactions, old_metadata = old_parser.parse_statement(lines)

# New parser
from bank_statement_parser import parse_bank_statement
text = '\n'.join(lines)
statement = parse_bank_statement(text)

# Compare
print(f"Old: {len(old_transactions)} transactions")
print(f"New: {len(statement.transactions)} transactions")
```

### 2. Validation Checklist

- [ ] Same number of transactions parsed
- [ ] Transaction amounts match
- [ ] Transaction dates match (allowing for format differences)
- [ ] Deposits/credits properly identified
- [ ] Withdrawals/debits properly identified
- [ ] No new parsing errors introduced

## Troubleshooting

### Issue: "No transactions parsed"

**Possible causes:**
- Bank not detected correctly
- Text extraction quality issues
- Section headers not matching expected format

**Solution:**
```python
from bank_statement_parser import detect_bank

# Check which bank was detected
bank = detect_bank(text)
print(f"Detected: {bank.value}")

# Force specific bank if detection fails
from bank_statement_parser import BankName
statement = parse_bank_statement(text, bank_name=BankName.CHASE)
```

### Issue: "Wrong transaction types"

**Possible causes:**
- Bank-specific format differences
- Section headers changed

**Solution:**
- Check `statement.errors` for parsing errors
- Review raw_line in transactions
- May need to adjust bank-specific parser

### Issue: "Dates are wrong"

**Possible causes:**
- Year not detected correctly for MM/DD format
- Date format not recognized

**Solution:**
- Check first few lines of statement for year
- Verify date format matches expected patterns

## Need Help?

1. Review the full documentation: `MULTI_BANK_PARSER_DOCS.md`
2. Check examples: `example_multi_bank_integration.py`
3. Run tests: `python3 test_multi_bank_parser.py`
4. Enable debug logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

## Benefits of Migrating

- ✅ **Better Accuracy**: Bank-specific parsers understand each bank's format
- ✅ **More Metadata**: Access to account numbers, statement periods, balances
- ✅ **Auto-Detection**: Automatically identifies which bank
- ✅ **Multiple Banks**: Process statements from different banks seamlessly
- ✅ **Better Error Handling**: Clear error messages when parsing fails
- ✅ **Future-Proof**: Easy to add new banks without modifying existing code
