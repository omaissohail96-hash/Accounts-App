# Multi-Bank Statement Parser - Quick Start

## 🎯 What's New?

The bank statement parser has been **completely refactored** to support multiple banks with automatic bank detection and bank-specific parsing logic.

### Supported Banks
- ✅ **Chase** - Business Complete Checking
- ✅ **American Express (Amex)** - Credit Cards
- ✅ **BMO** - Elite Business Checking  
- ✅ **Bank of America** - Business Checking
- ✅ **Fifth Third Bank** - Business Accounts
- ✅ **US Bank** - Silver Business Checking

## 🚀 Quick Start

### Basic Usage

```python
from bank_statement_parser import parse_bank_statement
import pdfplumber

# Extract text from PDF
with pdfplumber.open("statement.pdf") as pdf:
    text = "\n".join(page.extract_text() for page in pdf.pages)

# Parse (auto-detects bank)
statement = parse_bank_statement(text)

# Use the data
print(f"Bank: {statement.bank_name}")
print(f"Transactions: {len(statement.transactions)}")
print(f"Beginning Balance: ${statement.beginning_balance:,.2f}")
print(f"Ending Balance: ${statement.ending_balance:,.2f}")

# Process transactions
for trans in statement.transactions:
    print(f"{trans.date.strftime('%Y-%m-%d')} | {trans.description} | ${trans.amount:,.2f}")
```

### Run Tests

```bash
python3 test_multi_bank_parser.py
```

Expected output:
```
============================================================
MULTI-BANK STATEMENT PARSER TEST SUITE
============================================================

=== TESTING CHASE PARSER ===
✓ Bank detection: CHASE
✓ Parsed 10 transactions
✓ Chase parser: PASSED

=== TESTING BMO PARSER ===
✓ Bank detection: BMO
✓ Parsed 6 transactions
✓ BMO parser: PASSED

... (all banks tested)

RESULTS: 7 passed, 0 failed
============================================================
```

## 📁 Project Structure

### Core Files

| File | Purpose |
|------|---------|
| `bank_statement_parser.py` | **Main refactored parser** - Contains all bank parsers, models, and detection logic |
| `test_multi_bank_parser.py` | Comprehensive test suite for all supported banks |
| `example_multi_bank_integration.py` | Examples showing PDF processing, CSV export, analysis |

### Documentation

| File | Description |
|------|-------------|
| `MULTI_BANK_PARSER_DOCS.md` | **Complete documentation** - Architecture, usage, adding new banks |
| `MIGRATION_GUIDE.md` | **Migration guide** - How to upgrade from old parser |
| `MULTI_BANK_QUICK_START.md` | **This file** - Quick reference and overview |

### Legacy Files (Still Present)

| File | Status |
|------|--------|
| `bank_data_analysis.py` | Contains `extract_chase_summary()` - used by existing code |
| `test_chase_*.py` | Old Chase-specific tests (may need updating) |

## 🏗️ Architecture Overview

### 1. Unified Models

```python
@dataclass
class Transaction:
    date: datetime                    # When it happened
    description: str                  # What it was
    amount: float                     # How much (negative = debit, positive = credit)
    type: TransactionType             # DEBIT or CREDIT
    bank_name: str                    # Which bank
    category: Optional[TransactionCategory]  # deposit, withdrawal, check, etc.
    running_balance: Optional[float]  # Balance after this transaction
    # ... more fields

@dataclass
class ParsedStatement:
    bank_name: str
    transactions: List[Transaction]
    beginning_balance: Optional[float]
    ending_balance: Optional[float]
    account_number: Optional[str]
    errors: List[str]
    # ... more fields
```

### 2. Auto Bank Detection

```python
def detect_bank(text: str) -> BankName:
    """Automatically detects which bank from statement text"""
    # Looks for bank-specific keywords in first page
    # Returns: CHASE, AMEX, BMO, BOA, FIFTH_THIRD, US_BANK, or UNKNOWN
```

### 3. Bank-Specific Parsers

Each bank has its own parser class:
- `ChaseParser` - Handles Chase-specific format
- `BMOParser` - Handles BMO-specific format
- `BoAParser` - Handles Bank of America format
- `FifthThirdParser` - Handles Fifth Third format
- `USBankParser` - Handles US Bank format
- `AmexParser` - Handles Amex credit card statements

### 4. Factory Pattern

```python
BANK_PARSERS = {
    BankName.CHASE: ChaseParser(),
    BankName.BMO: BMOParser(),
    # ... other banks
}

def parse_bank_statement(text: str, bank_name: Optional[BankName] = None):
    if bank_name is None:
        bank_name = detect_bank(text)  # Auto-detect
    
    parser = BANK_PARSERS.get(bank_name)
    return parser.parse(text)
```

## 🔄 Migration from Old Parser

### Option 1: Legacy Compatibility (Quick)

```python
# Old code still works with LegacyBankStatementParser
from bank_statement_parser import LegacyBankStatementParser

parser = LegacyBankStatementParser()
transactions, metadata = parser.parse_statement(lines)  # Same API as before
```

### Option 2: Full Migration (Recommended)

```python
# New way - more powerful
from bank_statement_parser import parse_bank_statement

text = '\n'.join(lines)  # Convert lines to text
statement = parse_bank_statement(text)  # Auto-detects bank

# Access enhanced data
for trans in statement.transactions:
    print(f"{trans.bank_name} | {trans.date} | {trans.description} | ${trans.amount}")
```

See `MIGRATION_GUIDE.md` for detailed migration instructions.

## 📊 Examples

### Example 1: Process Single PDF

```python
from bank_statement_parser import parse_bank_statement
import pdfplumber

with pdfplumber.open("chase_statement.pdf") as pdf:
    text = "\n".join(page.extract_text() for page in pdf.pages)

statement = parse_bank_statement(text)

# Analyze
credits = sum(t.amount for t in statement.transactions if t.amount > 0)
debits = sum(abs(t.amount) for t in statement.transactions if t.amount < 0)

print(f"Total Credits: ${credits:,.2f}")
print(f"Total Debits: ${debits:,.2f}")
print(f"Net: ${credits - debits:,.2f}")
```

### Example 2: Process Multiple Banks

```python
from bank_statement_parser import parse_bank_statement
from pathlib import Path

statements = []
for pdf_file in Path("statements").glob("*.pdf"):
    # Extract text
    with pdfplumber.open(pdf_file) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages)
    
    # Parse (auto-detects bank)
    statement = parse_bank_statement(text)
    statements.append(statement)
    
    print(f"{pdf_file.name} → {statement.bank_name} ({len(statement.transactions)} txns)")

# Combine all transactions
all_transactions = []
for stmt in statements:
    all_transactions.extend(stmt.transactions)

print(f"\nTotal transactions across all banks: {len(all_transactions)}")
```

### Example 3: Export to CSV

```python
import csv
from bank_statement_parser import parse_bank_statement

statement = parse_bank_statement(text)

with open('transactions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Date', 'Bank', 'Description', 'Amount', 'Balance'])
    
    for trans in statement.transactions:
        writer.writerow([
            trans.date.strftime('%Y-%m-%d'),
            trans.bank_name,
            trans.description,
            f"{trans.amount:.2f}",
            f"{trans.running_balance:.2f}" if trans.running_balance else ""
        ])

print(f"Exported {len(statement.transactions)} transactions")
```

See `example_multi_bank_integration.py` for more examples.

## 🔧 Adding a New Bank

1. **Add to BankName enum**
   ```python
   class BankName(Enum):
       NEW_BANK = "new_bank"
   ```

2. **Add detection logic**
   ```python
   def detect_bank(text: str) -> BankName:
       if "NEW BANK IDENTIFIER" in text[:3000].upper():
           return BankName.NEW_BANK
   ```

3. **Create parser class**
   ```python
   class NewBankParser(BankStatementParser):
       def parse(self, text: str) -> ParsedStatement:
           # Implement parsing logic
           pass
   ```

4. **Register parser**
   ```python
   BANK_PARSERS[BankName.NEW_BANK] = NewBankParser()
   ```

5. **Add tests**
   ```python
   def test_new_bank_parser():
       # Test with sample statement
       pass
   ```

See `MULTI_BANK_PARSER_DOCS.md` for detailed instructions.

## ✅ Testing

### Run All Tests
```bash
python3 test_multi_bank_parser.py
```

### Test Specific Bank
```python
from bank_statement_parser import parse_bank_statement, BankName

# Force specific bank parser
statement = parse_bank_statement(chase_text, bank_name=BankName.CHASE)
assert len(statement.transactions) > 0
```

### Validate Results
```python
# Check for errors
if statement.errors:
    print("Errors:", statement.errors)

# Check transactions needing review
needs_review = [t for t in statement.transactions if t.needs_review]
if needs_review:
    print(f"{len(needs_review)} transactions need manual review")
```

## 🐛 Troubleshooting

### No transactions parsed?
```python
from bank_statement_parser import detect_bank

# Check bank detection
bank = detect_bank(text)
print(f"Detected: {bank.value}")

# Check for errors
statement = parse_bank_statement(text)
print(f"Errors: {statement.errors}")
```

### Wrong bank detected?
```python
# Force specific bank
from bank_statement_parser import BankName
statement = parse_bank_statement(text, bank_name=BankName.CHASE)
```

### Transaction amounts wrong?
```python
# Check transaction details
for trans in statement.transactions[:5]:  # First 5
    print(f"Raw: {trans.raw_line}")
    print(f"Parsed: {trans.amount} | {trans.type.value}")
    print()
```

## 📚 Documentation

- **`MULTI_BANK_PARSER_DOCS.md`** - Complete technical documentation
- **`MIGRATION_GUIDE.md`** - Upgrading from old parser
- **`example_multi_bank_integration.py`** - Working code examples

## 🎯 Key Features

✅ **Automatic Bank Detection** - No manual configuration needed  
✅ **Bank-Specific Parsers** - Accurate parsing for each bank's format  
✅ **Enhanced Metadata** - Account numbers, balances, statement periods  
✅ **Better Error Handling** - Clear error messages and flagging  
✅ **Backward Compatible** - Legacy API still available  
✅ **Well Tested** - Comprehensive test suite included  
✅ **Easy to Extend** - Simple process to add new banks  

## 🚫 What Didn't Change

- PDF extraction (still use `pdfplumber` or similar)
- Transaction categorization logic (still in `transaction_categorizer.py`)
- The existing `bank_data_analysis.py` functionality
- Your existing PDF files and data

## 💡 Best Practices

1. **Always check for errors**: `if statement.errors: ...`
2. **Validate balances**: Compare beginning + transactions = ending
3. **Review flagged transactions**: `if trans.needs_review: ...`
4. **Sort by date**: `sorted(transactions, key=lambda t: t.date)`
5. **Enable logging for debugging**: `logging.basicConfig(level=logging.DEBUG)`

## 🤝 Contributing

To add support for a new bank:
1. Create a parser class in `bank_statement_parser.py`
2. Add detection logic to `detect_bank()`
3. Register in `BANK_PARSERS`
4. Add tests in `test_multi_bank_parser.py`
5. Update documentation

## 📝 License

Same as the main project.

---

**Questions?** Check the full documentation in `MULTI_BANK_PARSER_DOCS.md` or review examples in `example_multi_bank_integration.py`.
