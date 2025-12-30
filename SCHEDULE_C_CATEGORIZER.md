# Schedule C Categorizer

Intelligent Schedule C categorization system for bank transactions that matches IRS Schedule C structure for tax preparation.

## Overview

The Schedule C categorizer automatically categorizes parsed bank transactions into the proper IRS Schedule C structure:
- **Part I**: Income (Gross receipts/sales, Returns & allowances, Other income)
- **Part II**: Expenses (Advertising, Bank fees, Contract labor, Wages, Rent, Utilities, Supplies, Travel, Legal & accounting, Insurance, Repairs & maintenance, Subscriptions, Other expenses)
- **Part III**: Cost of Goods Sold (COGS) - Inventory purchases, freight-in, direct materials, customs/duties
- **Part IV**: Vehicle expenses (business-related only)
- **Part V**: Other expenses

## Exclusion Rules

The categorizer automatically excludes:
- **ATM withdrawals** → Marked as Owner draw (NOT a business expense)
- **Personal spending** → Excluded from business expenses
- **Transfers between accounts** → Excluded (not income or expense)

## Usage

### Basic Usage

```python
from bank_statement_parser import Transaction
from schedule_c_categorizer import ScheduleCCategorizer

# Create categorizer
categorizer = ScheduleCCategorizer()

# Categorize transactions
transactions = [
    Transaction(
        date="2024-01-15",
        transaction_type="deposit",
        vendor="Shopify",
        amount=1500.00,
        description="Shopify payout for sales",
        raw_line=""
    ),
    Transaction(
        date="2024-01-12",
        transaction_type="withdrawal",
        vendor="Google Ads",
        amount=-200.00,
        description="Google Ads spend",
        raw_line=""
    ),
]

# Categorize
categorized = categorizer.categorize_transactions(transactions)

# Each item is a tuple: (Transaction, ScheduleCCategory)
for transaction, category in categorized:
    print(f"{transaction.vendor}: {category.category_name} ({category.line_number})")
```

### Integration with Bank Statement Parser

```python
from bank_statement_parser import BankStatementParser
from schedule_c_categorizer import ScheduleCCategorizer

# Parse bank statement
parser = BankStatementParser()
# ... parse your statement to get transactions ...

# Categorize for Schedule C
categorizer = ScheduleCCategorizer()
categorized = categorizer.categorize_transactions(transactions)

# Generate summary
summary = categorizer.generate_schedule_c_summary(categorized)

# Generate human-readable report
report = categorizer.generate_schedule_c_report(categorized)
print(report)
```

### Export for Bookkeeping

The categorized transactions can be exported in formats suitable for bookkeeping software (QuickBooks, Xero, etc.):

```python
# See example_schedule_c_integration.py for CSV/JSON export functions
```

## Schedule C Category Structure

### Part I - Income

| Line | Category | Tax Code | Keywords |
|------|----------|----------|----------|
| Line 1 | Gross receipts or sales | GROSS | shopify, ebay, stripe, paypal, tiktok, payment received, etc. |
| Line 2 | Returns and allowances | RETURNS | refund, return, chargeback, reversal |
| Line 6 | Other income | OTHER_INCOME | interest income, dividend, royalty |

### Part II - Expenses

| Line | Category | Tax Code | Keywords |
|------|----------|----------|----------|
| Line 8 | Advertising | ADVERTISING | google ads, facebook ads, meta ads, advertising, marketing |
| Line 11 | Contract labor | CONTRACT_LABOR | upwork, fiverr, freelancer, contractor, consultant |
| Line 15 | Insurance | INSURANCE | insurance, premium, coverage |
| Line 17 | Legal and professional services | LEGAL_ACCOUNTING | legal, attorney, cpa, accounting, bookkeeping |
| Line 18 | Office expense | BANK_FEES | bank fee, service charge, platform fee, payment processing |
| Line 20 | Rent or lease | RENT | rent, lease, landlord |
| Line 21 | Repairs and maintenance | REPAIRS | repair, maintenance, fix, upkeep |
| Line 22 | Supplies | SUPPLIES | office supplies, store supplies, stationery |
| Line 24a | Travel | TRAVEL | airline, flight, hotel, uber, lyft, travel |
| Line 25 | Utilities | UTILITIES | electric, internet, phone, mobile, utilities |
| Line 26 | Wages and salaries | WAGES | payroll, salary, wage, employee, payroll tax |
| Line 27a | Other expenses | SUBSCRIPTIONS | subscription, saas, software, membership, dues |

### Part III - Cost of Goods Sold (COGS)

| Line | Category | Tax Code | Keywords |
|------|----------|----------|----------|
| Line 35 | Purchases | COGS_INVENTORY | inventory, stock, product purchase, merchandise, wholesale |
| Line 36 | Cost of labor | COGS_FREIGHT | freight, shipping in, freight in, inbound shipping |
| Line 38 | Materials and supplies | COGS_MATERIALS | raw materials, direct materials, component, part |
| Line 39 | Other costs | COGS_CUSTOMS | customs, duty, import tax, tariff, import fee |

### Part IV - Vehicle Expenses

| Line | Category | Tax Code | Keywords |
|------|----------|----------|----------|
| Line 9 | Car and truck expenses | VEHICLE | gas, fuel, shell, chevron, parking, toll, vehicle maintenance |

## Output Structure

### ScheduleCCategory Object

Each categorized transaction returns a `ScheduleCCategory` object with:
- `part`: "Part I", "Part II", "Part III", "Part IV", "Part V", or "Excluded"
- `line_number`: IRS Schedule C line number (e.g., "Line 1", "Line 8")
- `category_name`: Human-readable category name
- `tax_code`: Machine-readable tax code for automation
- `is_excluded`: Boolean indicating if transaction is excluded
- `exclusion_reason`: Reason for exclusion (if applicable)
- `is_owner_draw`: Boolean indicating if this is an owner draw

### Summary Dictionary

The `generate_schedule_c_summary()` method returns a structured dictionary:

```python
{
    "Part I - Income": {
        "Line 1 - Gross receipts or sales": {
            "tax_code": "GROSS",
            "line_number": "Line 1",
            "category_name": "Gross receipts or sales",
            "total_amount": 1500.00,
            "transaction_count": 1,
            "transactions": [...]
        },
        ...
    },
    "Part II - Expenses": {...},
    "Part III - COGS": {...},
    "Part IV - Vehicle": {...},
    "Part V - Other Expenses": {...},
    "Excluded": {...},
    "Owner Draws": [...]
}
```

## Customization

To add custom categorization rules, modify the keyword lists in `schedule_c_categorizer.py`:

```python
# In ScheduleCCategorizer._build_categorization_rules()
self.income_gross_receipts = [
    # Add your custom keywords here
    "your-platform", "your-payment-processor"
]
```

## Testing

Run the test suite:

```bash
python3 test_schedule_c_categorizer.py
```

This will verify that transactions are correctly categorized into the proper Schedule C categories.

## Requirements

- Python 3.7+
- `bank_statement_parser` module (for Transaction dataclass)

## Notes

- The categorizer uses rule-based and keyword-based logic for deterministic categorization
- Transactions are matched based on vendor name and description text
- All matching is case-insensitive
- If no match is found, transactions default to "Part V - Other Expenses" (Line 27a)
- Owner draws (ATM withdrawals) are tracked separately and not included in expense totals


