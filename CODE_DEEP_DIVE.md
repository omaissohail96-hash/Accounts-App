# Custom Rules Filter - Code Deep Dive

## File Location
**File:** [bank_data_analysis.py](bank_data_analysis.py#L1520-L1620)

---

## Current Filter Implementation (Lines 1523-1607)

### Main Function: `reapply_custom_rules()`

```python
def reapply_custom_rules():
    """
    Apply custom business rules to categorize transactions.
    Rules can include keyword matching, amount filters, and pattern matching.
    """
    
    # Load business rules from JSON file
    business_rules = load_business_rules(user_id, business_name) or []
    
    # Normalize text for comparison (lowercase, trimmed)
    def _norm(s: str) -> str:
        return " ".join((s or "").lower().strip().split())
    
    # Sort rules by priority (lower number = higher priority)
    sorted_rules = sorted(business_rules, key=lambda r: r.get("priority", 999))
```

---

## Filter Execution Pipeline

### 1. FILTER 1: Primary Keyword Match

```python
# Line 1556-1559: PRIMARY KEYWORD FILTER
for rule in sorted_rules:
    # 1. Check primary keyword (required)
    kw = _norm(rule.get("keyword", ""))
    if not kw or kw not in text:
        continue  # Skip this rule if keyword doesn't match
```

**Logic:**
- Normalize both rule keyword and transaction text to lowercase
- Check if keyword exists anywhere in text
- If not found, skip to next rule

**Example:**
```python
rule_keyword = "CHECK"
transaction_text = "Check #502 Payment to Vendor"
normalized_text = "check #502 payment to vendor"

match = "check" in normalized_text  # ✓ True
```

---

### 2. FILTER 2: Amount Range Checking

```python
# Line 1563-1570: AMOUNT RANGE FILTER
# Use absolute value of amount (withdrawals are negative)
tx_amount = abs(getattr(tx, "amount", 0.0))

# Check minimum amount (if specified)
min_amt = rule.get("min_amount")
if min_amt is not None and tx_amount < min_amt:
    continue  # Amount too small for this rule

# Check maximum amount (if specified)
max_amt = rule.get("max_amount")
if max_amt is not None and tx_amount > max_amt:
    continue  # Amount too large for this rule
```

**Logic:**
- Convert amount to absolute value (deposits positive, withdrawals negative)
- Compare against min_amount and max_amount boundaries
- Both boundaries are INCLUSIVE (≤ and ≥)
- If NOT specified in rule, skip this check

**Examples:**

```python
# Example 1: Amount in range
tx_amount = 750
min_amt = 500
max_amt = 2000

check_min = (min_amt is None) or (750 >= 500)  # ✓ True
check_max = (max_amt is None) or (750 <= 2000)  # ✓ True
result = "PASS" ✓

# Example 2: Amount too small
tx_amount = 250
min_amt = 500
max_amt = 2000

check_min = (min_amt is None) or (250 >= 500)  # ✗ False
result = "FAIL - Too small"

# Example 3: Amount too large
tx_amount = 3500
min_amt = 500
max_amt = 2000

check_max = (max_amt is None) or (3500 <= 2000)  # ✗ False
result = "FAIL - Too large"
```

---

### 3. FILTER 3: Exclusion Keywords

```python
# Line 1572-1575: EXCLUSION FILTER
exclude_kws = rule.get("exclude_keywords", [])
if exclude_kws and any(_norm(excl) in text for excl in exclude_kws):
    continue  # Excluded by keyword
```

**Logic:**
- Check if ANY exclusion keyword appears in text
- If found, skip this rule (don't apply)
- Uses OR logic: any match causes failure
- If no exclusion keywords specified, this filter passes

**Examples:**

```python
# Example 1: Check WITHOUT exclusion keywords
rule_exclude = []
text = "Stripe Payment"
result = any(excl in text for excl in [])  # ✓ Passes

# Example 2: Stripe refund (excluded)
rule_exclude = ["REFUND", "CHARGEBACK"]
text = "Stripe Refund Transaction"
normalized_excl = ["refund", "chargeback"]
normalized_text = "stripe refund transaction"

result = any(excl in normalized_text for excl in normalized_excl)
# "refund" in "stripe refund transaction" → ✓ Found
# Action: SKIP this rule ✗

# Example 3: Stripe payment (not excluded)
rule_exclude = ["REFUND", "CHARGEBACK"]
text = "Stripe Payment Charge"
normalized_text = "stripe payment charge"

result = any(excl in normalized_text for excl in normalized_excl)
# "refund" in "stripe payment charge" → ✗ Not found
# "chargeback" in "stripe payment charge" → ✗ Not found
# Action: PASS - continue to next filter ✓
```

---

### 4. FILTER 4: Additional Keywords

```python
# Line 1577-1580: ADDITIONAL KEYWORDS FILTER
additional_kws = rule.get("additional_keywords", [])
if additional_kws and not all(_norm(kw_add) in text for kw_add in additional_kws):
    continue  # Not all additional keywords match
```

**Logic:**
- Check if ALL additional keywords are present in text
- ALL must match (AND logic)
- If any keyword is missing, skip this rule
- If no additional keywords specified, this filter passes

**Examples:**

```python
# Example 1: Rent payment (both keywords present)
rule_additional = ["RENT", "PAYMENT"]
text = "Landlord Payment for Rent"
normalized_text = "landlord payment for rent"

check1 = "rent" in normalized_text  # ✓ True
check2 = "payment" in normalized_text  # ✓ True
result = all([check1, check2])  # ✓ Both present - PASS

# Example 2: Rent-only (missing PAYMENT keyword)
rule_additional = ["RENT", "PAYMENT"]
text = "Monthly Rent Charge"
normalized_text = "monthly rent charge"

check1 = "rent" in normalized_text  # ✓ True
check2 = "payment" in normalized_text  # ✗ False
result = all([check1, check2])  # ✗ Missing PAYMENT - SKIP

# Example 3: No additional keywords required
rule_additional = []
text = "Any transaction"

result = all([])  # ✓ Empty list = all() returns True - PASS
```

---

### 5. FILTER 5: Rule Application

```python
# Line 1582-1587: APPLY RULE IF ALL FILTERS PASSED
# 5. All conditions met - apply the rule
tx.account_code = rule.get("account_code")
tx._mapped_by_rule = True
matched = True
break  # First matching rule wins (priority-based)
```

**Logic:**
- Assign account code to transaction
- Mark transaction as mapped by rule
- Stop processing (don't check other rules)
- Only first matching rule is applied

---

## Rule Data Structure

### JSON Structure (Stored in data/rules/)

```json
{
  "keyword": "CHECK",
  "account_code": "801",
  "account_display": "801 · SALARIES-OFFICERS",
  "min_amount": 500,
  "max_amount": 2000,
  "priority": 1,
  "exclude_keywords": ["REFUND", "CANCELLED"],
  "additional_keywords": []
}
```

### Field Definitions

| Field | Type | Required | Example | Description |
|-------|------|----------|---------|-------------|
| `keyword` | string | YES | "CHECK" | Primary keyword to match |
| `account_code` | string | YES | "801" | Target account code |
| `account_display` | string | YES | "801 · SALARIES-OFFICERS" | Display text for UI |
| `min_amount` | float | NO | 500 | Minimum transaction amount |
| `max_amount` | float | NO | 2000 | Maximum transaction amount |
| `priority` | int | NO | 1 | Processing priority (lower = first) |
| `exclude_keywords` | array | NO | ["REFUND"] | Keywords to exclude |
| `additional_keywords` | array | NO | ["PAYMENT"] | Additional required keywords |

---

## Filter Decision Tree

```
┌─ START ──────────────────────────────────────────────────────────────┐
│ Transaction: "Check #502 - Payroll Payment"                         │
│ Amount: $750                                                         │
│ Rules: [Rule1: CHECK $500-2k, Rule2: CHECK $2k-10k, Rule3: STRIPE] │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
        ╔════════════════════════════╗
        ║ Sort rules by priority    ║
        ║ Rule1(P:1), Rule2(P:2),   ║
        ║ Rule3(P:999)              ║
        ╚════════════╤═══════════════╝
                     │
        ┌────────────▼───────────────┐
        │ FOR EACH RULE (in order)  │
        └────────────┬───────────────┘
                     │
             ┌───────▼────────┐
             │ Rule 1: CHECK  │
             │ $500-$2000     │
             └───────┬────────┘
                     │
        ┌────────────▼──────────────┐
        │ FILTER 1: Keyword match?  │
        │ "check" in text?          │
        │ YES ─────────┐            │
        │              ▼            │
        │        Continue to F2     │
        └───────────────────────────┘
                     │
        ┌────────────▼──────────────┐
        │ FILTER 2: Amount range?   │
        │ 500 ≤ 750 ≤ 2000?        │
        │ YES ─────────┐            │
        │              ▼            │
        │        Continue to F3     │
        └───────────────────────────┘
                     │
        ┌────────────▼──────────────┐
        │ FILTER 3: Exclusions?     │
        │ Has REFUND keyword?       │
        │ NO ──────────┐            │
        │              ▼            │
        │        Continue to F4     │
        └───────────────────────────┘
                     │
        ┌────────────▼──────────────┐
        │ FILTER 4: Additions?      │
        │ (if specified)            │
        │ NO requirements           │
        │ YES (pass) ──┐            │
        │              ▼            │
        │        All passed!        │
        └───────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────┐
        │ ✅ APPLY RULE                │
        │ account_code = "801"         │
        │ STOP PROCESSING (break)      │
        └──────────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────┐
        │ RESULT:                      │
        │ Check $750 → 801 SALARIES    │
        │ ✅ CATEGORIZED CORRECTLY     │
        └──────────────────────────────┘
```

---

## Code Performance Analysis

### Time Complexity

```
For N transactions and M rules:
- Time: O(N × M × K) where K = average text length
- Rules sorted once: O(M log M)
- Per transaction: O(M) rule checks
- Per rule: O(K) text matching

Example:
- 1000 transactions
- 10 rules
- Average text length 50 chars
= ~500,000 string comparison operations
= Fast (< 1 second)
```

### Space Complexity

```
- Rules storage: O(M) for M rules
- Per transaction: O(1) for account_code assignment
- Text normalization: O(K) temporary space

Example:
- 10 rules × ~500 bytes each = 5 KB
- Very efficient
```

---

## Example Scenarios with Code

### Scenario 1: $750 Check to Salaries

```python
# Input
transaction = {
    "vendor": "Check #502",
    "description": "Payroll Payment",
    "amount": -750.00  # Negative = withdrawal
}

rule = {
    "keyword": "CHECK",
    "account_code": "801",
    "min_amount": 500,
    "max_amount": 2000,
    "priority": 1
}

# Processing
text = "check #502 payroll payment"  # Normalized
tx_amount = abs(-750.00) = 750

# Filter 1: Keyword
"check" in text  # ✓ True

# Filter 2: Amount
750 >= 500  # ✓ True (min check)
750 <= 2000  # ✓ True (max check)

# Filter 3: Exclusions
exclude_kws = []  # No exclusions
any([]) = False  # ✓ Pass

# Filter 4: Additions
additional_kws = []  # No additional keywords
all([]) = True  # ✓ Pass

# Result
tx.account_code = "801"
tx._mapped_by_rule = True
break  # Stop processing

# Output
account_code = "801 · SALARIES-OFFICERS"  # ✅ CORRECT
```

---

### Scenario 2: Stripe Refund (Excluded)

```python
# Input
transaction = {
    "vendor": "Stripe Inc",
    "description": "Refund - Transaction Reversal",
    "amount": 150.00
}

rule = {
    "keyword": "STRIPE",
    "account_code": "601",
    "exclude_keywords": ["REFUND", "CHARGEBACK"],
    "priority": 10
}

# Processing
text = "stripe inc refund transaction reversal"  # Normalized
tx_amount = abs(150.00) = 150

# Filter 1: Keyword
"stripe" in text  # ✓ True

# Filter 2: Amount (no min/max)
skip

# Filter 3: Exclusions
exclude_kws = ["refund", "chargeback"]
any(excl in text for excl in exclude_kws)
# "refund" in "stripe inc refund ..." → ✓ Found!
continue  # Skip this rule ✗

# Result
# Rule not applied
# Falls through to next rule or default mapper

# Output
account_code = "999 · OTHER EXPENSES"  # Falls to default
```

---

### Scenario 3: Rent Payment (Multiple Keywords)

```python
# Input
transaction = {
    "vendor": "Property Manager",
    "description": "Rent payment for building",
    "amount": -1500.00
}

rule = {
    "keyword": "PAYMENT",
    "account_code": "808",
    "additional_keywords": ["RENT", "BUILDING"],
    "priority": 5
}

# Processing
text = "property manager rent payment for building"  # Normalized
tx_amount = abs(-1500.00) = 1500

# Filter 1: Keyword
"payment" in text  # ✓ True

# Filter 2: Amount (no min/max)
skip

# Filter 3: Exclusions
exclude_kws = []  # No exclusions
any([]) = False  # ✓ Pass

# Filter 4: Additional Keywords
additional_kws = ["rent", "building"]
all(kw in text for kw in additional_kws)
# "rent" in text → ✓ True
# "building" in text → ✓ True
all([True, True]) = True  # ✓ Pass

# Result
tx.account_code = "808"
tx._mapped_by_rule = True
break

# Output
account_code = "808 · RENT"  # ✅ CORRECT
```

---

## Key Implementation Details

### 1. Priority Sorting
```python
# Lower number = higher priority (processed first)
sorted_rules = sorted(business_rules, key=lambda r: r.get("priority", 999))

# Default priority if not specified: 999 (lowest)
# This puts new rules at the end unless explicit priority set
```

### 2. Text Normalization
```python
def _norm(s: str) -> str:
    """Convert to lowercase and remove extra whitespace"""
    return " ".join((s or "").lower().strip().split())

# Examples:
_norm("CHECK #502")  # → "check #502"
_norm("  STRIPE   ")  # → "stripe"
_norm(None)  # → ""
```

### 3. Absolute Value for Amounts
```python
# Transactions can be positive (deposits) or negative (withdrawals)
# For filtering purposes, use absolute value
tx_amount = abs(getattr(tx, "amount", 0.0))

# Example:
abs(-750.00) = 750.00  # Enables comparison with min/max
abs(150.00) = 150.00   # Same logic for deposits
```

### 4. First Match Wins
```python
# When a rule matches, stop immediately
for rule in sorted_rules:
    # ... all filters pass ...
    tx.account_code = rule.get("account_code")
    matched = True
    break  # Stop processing other rules

# Only first matching rule is applied
# Other rules are ignored for this transaction
```

---

## Common Filter Combinations

### Combination 1: Simple Keyword
```python
{
    "keyword": "PAYPAL",
    "account_code": "601",
    "priority": 50
}
# Matches: Any transaction with "paypal"
# Fails: Transactions without "paypal"
```

### Combination 2: Keyword + Amount Range
```python
{
    "keyword": "CHECK",
    "account_code": "801",
    "min_amount": 500,
    "max_amount": 2000,
    "priority": 1
}
# Matches: "check" keyword AND $500-$2000
# Fails: "check" but wrong amount
```

### Combination 3: Keyword + Exclusions
```python
{
    "keyword": "STRIPE",
    "account_code": "601",
    "exclude_keywords": ["REFUND", "DISPUTE"],
    "priority": 10
}
# Matches: "stripe" keyword WITHOUT "refund" or "dispute"
# Fails: "stripe" + "refund"
```

### Combination 4: Keyword + Additions + Exclusions
```python
{
    "keyword": "PAYMENT",
    "account_code": "808",
    "additional_keywords": ["RENT", "LANDLORD"],
    "exclude_keywords": ["REFUND"],
    "min_amount": 100,
    "max_amount": 5000,
    "priority": 2
}
# Matches: "payment" AND ("rent" AND "landlord") AND NOT "refund" AND $100-$5000
# Fails: If any condition is false
```

---

## Error Handling

### Current Implementation
```python
# Safe defaults
tx_amount = abs(getattr(tx, "amount", 0.0))  # Defaults to 0 if missing
kw = _norm(rule.get("keyword", ""))          # Defaults to "" if missing
exclude_kws = rule.get("exclude_keywords", [])  # Defaults to [] if missing

# If any field is missing, filter is skipped gracefully
if not kw:  # Empty keyword = skip filter
    continue
```

### Potential Issues
```python
# Issue 1: What if amount is None?
getattr(tx, "amount", 0.0)  # ✓ Handled (defaults to 0.0)

# Issue 2: What if rule keyword is None?
_norm(None)  # → ""
if not kw  # → True (skip)  ✓ Handled

# Issue 3: What if exclude_keywords has None?
exclude_kws = [None, "REFUND"]
_norm(None) in text  # → "" in text → Usually False ✓ Safe

# Issue 4: Circular rule references?
# Not applicable - rules are independent
```

---

## Future Enhancements

### Potential Improvements

```python
# 1. Regex Support for Keywords
pattern = re.compile(r"check\s*#?\d+", re.I)
if pattern.search(text):  # More flexible than substring match

# 2. Case Sensitivity Option
{
    "keyword": "CHECK",
    "case_sensitive": False  # Current: always false
}

# 3. Amount Condition Operators
{
    "amount_operator": ">=",  # Instead of min/max
    "amount_value": 500
}

# 4. Rule Combinations (AND/OR logic)
{
    "conditions": [
        {"type": "keyword", "value": "CHECK"},
        {"type": "keyword", "value": "SALARY"},
        {"operator": "AND"}  # Must match both
    ]
}

# 5. Rule Weights/Confidence Scores
{
    "keyword": "CHECK",
    "confidence": 0.95  # How confident is this rule?
}
```

---

## Summary

The custom rules filter system provides:

✅ **Flexible Matching**: Keywords, amounts, patterns  
✅ **Priority-Based Processing**: Control rule evaluation order  
✅ **Safe Filtering**: All checks are optional and safe  
✅ **Efficient Execution**: O(N×M) time complexity  
✅ **Persistent Storage**: Rules saved to JSON  
✅ **Easy to Extend**: Add new filters without code changes  

The system works correctly; you just need to define appropriate rules with amount filters to fix the $500 check categorization issue.
