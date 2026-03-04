# Bank Fee Parser - Quick Reference Guide

## Overview

The robust bank fee parser accurately detects and extracts bank fee amounts from transaction descriptions, ensuring fees are properly categorized without misreading total transaction amounts.

## ✅ What It Does

### 1. **Accurate Fee Amount Extraction**
- Extracts the **actual fee amount** from descriptions, not transaction totals
- Handles complex descriptions with multiple amounts
- Example: `"ACH Fee Qty 3 7.50 Total Fees 7.50 Transfer USD 32,563.65"` 
  - ✅ Extracts: **$7.50** (the fee)
  - ❌ Not: $32,563.65 (the transfer amount)

### 2. **Smart Detection**
Detects bank fees using keywords:
- `fee`, `charge`, `service`, `monthly`, `maintenance`
- `transaction`, `processing`, `nsf`, `overdraft`, `atm`
- `wire`, `international`, `platform`, `ach`

### 3. **Amount Validation**
- Validates fees are realistic (< $500 by default)
- Flags suspiciously large fees for review
- Sets `needs_review` flag when amounts are unclear

### 4. **Multiple Extraction Strategies**

**Priority Order:**
1. **"Total Fees" pattern**: `Qty 3 7.50 Total Fees 7.50` → $7.50
2. **Amount after "fee"**: `Service Fee 12.50` → $12.50
3. **Amount before "fee"**: `12.50 Service Fee` → $12.50
4. **Smallest amount** (when multiple found): Assumes fees are smaller

## 📋 Integration Examples

### Basic Usage

```python
from bank_fee_parser import parse_bank_fee

# Parse a transaction description
result = parse_bank_fee(
    description="Wire Transfer Fee $25.00 Sent Amount USD 10,000",
    transaction_amount=-10025.00
)

if result.is_bank_fee:
    print(f"Fee Amount: ${result.fee_amount}")  # $25.00
    print(f"Needs Review: {result.needs_review}")  # False
    print(f"Reason: {result.reason}")
```

### With Schedule C Categorizer

```python
from bank_statement_parser import Transaction
from schedule_c_categorizer import ScheduleCCategorizer

# Create transaction
transaction = Transaction(
    date="2024-01-15",
    description="Monthly Service Fee 15.00",
    amount=-15.00,
    transaction_type="withdrawal"
)

# Categorize (automatically uses robust fee parser)
categorizer = ScheduleCCategorizer()
category = categorizer.categorize_transaction(transaction)

print(f"Category: {category.category_name}")  # Office expense
print(f"Tax Code: {category.tax_code}")  # BANK_FEES
print(f"Amount: ${transaction.amount}")  # -15.00
print(f"Needs Review: {transaction.needs_review}")  # False
```

### With Bank Data Analysis

The parser is automatically integrated into `bank_data_analysis.py`. When processing statements:

```python
# Parser is initialized in FallbackStatementParser.__init__()
# Automatically used when parsing fee transactions
# No manual integration needed!
```

## 🎯 Test Results

**All 13 core tests passing:**

✅ ACH Fee with Qty and Total  
✅ Simple Service Fee  
✅ ATM Fee  
✅ Wire Transfer Fee  
✅ NSF Fee  
✅ Processing Fee with Multiple Amounts  
✅ Overdraft Fee  
✅ Suspicious Large Fee (flagged for review)  
✅ Not a Bank Fee (correctly rejected)  
✅ Platform Fee  
✅ Maintenance Fee  
✅ Service Charge  
✅ International Transaction Fee  

**Edge cases:**
- Empty descriptions ✅
- Missing amounts ✅
- Multiple fees ✅
- Large fees flagged ✅
- Zero fees flagged ✅

## 🔍 Real-World Example

**Input Transaction:**
```
Description: "Standard ACH Pmnts Initial Fee Qty 3 7.50 Total Fees 7.50 Transfer USD 32,563.65"
Amount: -$32,563.65
```

**Output:**
```python
{
    'is_bank_fee': True,
    'fee_amount': 7.50,           # ✅ Correct fee amount
    'needs_review': False,
    'vendor': 'Bank Fees',
    'reason': 'Extracted from Total Fees label',
    'original_amount': -32563.65  # Preserved for reference
}
```

**Categorization:**
- **Category**: Office expense
- **Schedule C**: Part II, Line 18
- **Tax Code**: BANK_FEES
- **Amount**: -$7.50 (corrected)

## ⚠️ Review Flags

Transactions are flagged for review when:

1. **No clear fee amount found** in description
2. **Fee exceeds $500** (configurable via `MAX_FEE_AMOUNT`)
3. **Fee is $0.00** (likely waived or error)
4. **Multiple amounts** found but unclear which is the fee

## 🔧 Configuration

### Adjust Maximum Fee Amount

```python
from bank_fee_parser import BankFeeParser

parser = BankFeeParser()
parser.MAX_FEE_AMOUNT = 1000.0  # Increase limit to $1000
```

### Custom Keywords

Modify in `bank_fee_parser.py`:
```python
BANK_FEE_KEYWORDS = [
    # Add your custom keywords here
    "custom fee", "special charge"
]
```

## 📊 Files Modified

1. **`bank_fee_parser.py`** - New robust parser module
2. **`schedule_c_categorizer.py`** - Integrated parser for categorization
3. **`bank_data_analysis.py`** - Integrated parser for statement parsing
4. **`test_bank_fee_parser.py`** - Comprehensive test suite

## 🚀 Running Tests

```bash
python3 test_bank_fee_parser.py
```

Expected output:
```
SUMMARY: 13 passed, 0 failed out of 13 tests
```

## 📝 Key Benefits

✅ **Accuracy**: Extracts correct fee amounts from complex descriptions  
✅ **Safety**: Validates amounts and flags suspicious transactions  
✅ **Transparency**: Provides clear reasons for amount choices  
✅ **Integration**: Works seamlessly with existing categorization logic  
✅ **Tested**: Comprehensive test suite ensures reliability  

---

**Questions or Issues?**

Run the test suite to see all scenarios:
```bash
python3 test_bank_fee_parser.py
```
