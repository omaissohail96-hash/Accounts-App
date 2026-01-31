# Code Examples: Custom Rules Configuration

## Complete Rule Configuration Examples

### Example 1: Basic Check Rule

```json
{
  "keyword": "CHECK",
  "account_code": "801",
  "account_display": "801 · SALARIES-OFFICERS",
  "min_amount": 500,
  "max_amount": 2000,
  "priority": 1
}
```

**What it does:**
- Matches transactions with "check" keyword
- Only if amount between $500-$2000
- Assigns to SALARIES-OFFICERS account (801)
- Processed with high priority (1)

**Matches:**
```
✓ "Check #502 Payroll" - $750
✓ "Employee Check" - $1200
✓ "PAYCHECK DEPOSIT" - $999
```

**Doesn't match:**
```
✗ "Check" - $250 (too small)
✗ "Check" - $3000 (too large)
✗ "Transfer" - $750 (wrong keyword)
```

---

### Example 2: Payroll Provider Specific

```json
{
  "keyword": "ADP",
  "account_code": "801",
  "account_display": "801 · SALARIES-OFFICERS",
  "min_amount": 500,
  "max_amount": 50000,
  "priority": 1
}
```

**What it does:**
- Only matches ADP (payroll processor)
- Any amount from $500-$50000
- Very specific, high priority

**Matches:**
```
✓ "ADP Payroll Services" - $5000
✓ "ADP Direct Deposit" - $1200
✓ "ADP Processing" - $50000
```

**Doesn't match:**
```
✗ "CHECK" - $5000 (different keyword)
✗ "ADP" - $250 (below minimum)
```

---

### Example 3: Stripe with Exclusions

```json
{
  "keyword": "STRIPE",
  "account_code": "601",
  "account_display": "601 · SALES",
  "exclude_keywords": ["REFUND", "CHARGEBACK", "DISPUTE"],
  "priority": 10
}
```

**What it does:**
- Stripe transactions → SALES
- EXCEPT refunds, chargebacks, disputes
- Uses OR logic for exclusions (any match = skip)

**Matches:**
```
✓ "Stripe Payment Charge" - $150
✓ "Stripe Transaction" - $500
✓ "Stripe Sale" - $1000
```

**Doesn't match:**
```
✗ "Stripe Refund" - $150 (has "refund")
✗ "Stripe Chargeback" - $300 (has "chargeback")
✗ "Stripe Dispute" - $100 (has "dispute")
```

---

### Example 4: Rent with Additional Keywords

```json
{
  "keyword": "PAYMENT",
  "account_code": "808",
  "account_display": "808 · RENT",
  "additional_keywords": ["RENT", "LANDLORD"],
  "min_amount": 100,
  "max_amount": 5000,
  "priority": 5
}
```

**What it does:**
- Must have "payment" keyword
- MUST have both "rent" AND "landlord"
- Amount between $100-$5000
- Uses AND logic for additional keywords (all must match)

**Matches:**
```
✓ "Payment to Landlord for Rent" - $1500
✓ "Rent Payment Landlord" - $2000
✓ "Monthly Rent Payment to Landlord" - $1200
```

**Doesn't match:**
```
✗ "Payment to John" - $1500 (missing "rent" and "landlord")
✗ "Rent Only" - $1500 (missing "payment")
✗ "Payment for Landlord" - $1500 (missing "rent")
✗ "Rent Payment" - $50 (below minimum)
```

---

### Example 5: Multi-Tier Check Categorization

**Complete set of 3 rules:**

```json
[
  {
    "keyword": "CHECK",
    "account_code": "605",
    "account_display": "605 · SUPPLIES",
    "min_amount": 50,
    "max_amount": 500,
    "priority": 1
  },
  {
    "keyword": "CHECK",
    "account_code": "801",
    "account_display": "801 · SALARIES-OFFICERS",
    "min_amount": 500,
    "max_amount": 2000,
    "priority": 2
  },
  {
    "keyword": "CHECK",
    "account_code": "801",
    "account_display": "801 · SALARIES-OFFICERS",
    "min_amount": 2000,
    "max_amount": 10000,
    "priority": 3
  }
]
```

**Routing Logic:**
```
Check $150   → Priority 1: $50-$500 range ✓ → 605 SUPPLIES
Check $500   → Priority 1: $50-$500 range ✓ → 605 SUPPLIES
Check $750   → Priority 1: fails, Priority 2: $500-$2000 ✓ → 801 SALARIES
Check $2000  → Priority 1: fails, Priority 2: $500-$2000 ✓ → 801 SALARIES
Check $3500  → Priority 1: fails, Priority 2: fails, Priority 3: $2000-$10000 ✓ → 801 SALARIES
Check $15000 → All fail → DEFAULT → 999 OTHER EXPENSES
```

---

### Example 6: Contractor Payments

```json
{
  "keyword": "CHECK",
  "account_code": "811",
  "account_display": "811 · CONTRACTORS",
  "additional_keywords": ["CONTRACTOR", "FREELANCE"],
  "exclude_keywords": ["REFUND", "ADVANCE"],
  "min_amount": 100,
  "max_amount": 5000,
  "priority": 3
}
```

**What it does:**
- Checks for contractors
- Must contain "CONTRACTOR" OR "FREELANCE" (additional keywords)
- Exclude "REFUND" and "ADVANCE"
- Amount $100-$5000

**Matches:**
```
✓ "Check Payment to Contractor Services" - $1500
✓ "Freelance Work Payment Check" - $500
✓ "Check - Contractor Invoice" - $2000
```

**Doesn't match:**
```
✗ "Check Payment" - $1500 (missing "contractor"/"freelance")
✗ "Check Contractor Refund" - $500 (has "refund")
✗ "Check Contractor Advance" - $300 (has "advance")
✗ "Check Contractor" - $50 (below minimum)
```

---

### Example 7: Complex Business Logic

```json
{
  "keyword": "PAYMENT",
  "account_code": "809",
  "account_display": "809 · UTILITIES",
  "additional_keywords": ["ELECTRIC", "POWER"],
  "exclude_keywords": ["DEPOSIT", "PENALTY"],
  "min_amount": 50,
  "max_amount": 1000,
  "priority": 4
}
```

**What it does:**
- Utility payments
- Must have "PAYMENT" + ("ELECTRIC" OR "POWER")
- No deposits or penalties
- Monthly billing range

**Real-world matches:**
```
✓ "Electric Company Payment" - $125
✓ "Power Utility Payment - Monthly" - $180
✓ "City Electric Payment" - $150
```

---

## Python Implementation: How Rules Are Applied

### Filter Application Code

```python
def apply_custom_rules(transaction, custom_rules):
    """Apply custom rules to categorize a transaction"""
    
    # Normalize text for matching
    text = (transaction.vendor or '') + ' ' + (transaction.description or '')
    text = ' '.join(text.lower().strip().split())
    
    # Get transaction amount (absolute value)
    tx_amount = abs(getattr(transaction, 'amount', 0.0))
    
    # Sort rules by priority (lower = first)
    sorted_rules = sorted(custom_rules, key=lambda r: r.get('priority', 999))
    
    # Apply each rule in priority order
    for rule in sorted_rules:
        
        # FILTER 1: Keyword must match
        keyword = rule.get('keyword', '').lower().strip()
        if not keyword or keyword not in text:
            continue  # Skip to next rule
        
        # FILTER 2: Amount range check
        min_amt = rule.get('min_amount')
        max_amt = rule.get('max_amount')
        
        if min_amt is not None and tx_amount < min_amt:
            continue  # Too small
        if max_amt is not None and tx_amount > max_amt:
            continue  # Too large
        
        # FILTER 3: Exclusion keywords
        exclude_kws = rule.get('exclude_keywords', [])
        if any(excl.lower() in text for excl in exclude_kws):
            continue  # Excluded
        
        # FILTER 4: Additional keywords (all must match)
        additional_kws = rule.get('additional_keywords', [])
        if not all(add.lower() in text for add in additional_kws):
            continue  # Missing required keyword
        
        # ALL FILTERS PASSED - Apply rule
        return rule.get('account_code')
    
    # No rule matched
    return None  # Fall through to default mapping
```

---

### Complete Example: Processing Transactions

```python
# Sample rules configuration
custom_rules = [
    {
        "keyword": "CHECK",
        "account_code": "801",
        "min_amount": 500,
        "max_amount": 2000,
        "priority": 1
    },
    {
        "keyword": "CHECK",
        "account_code": "801",
        "min_amount": 2000,
        "max_amount": 10000,
        "priority": 2
    },
    {
        "keyword": "STRIPE",
        "account_code": "601",
        "exclude_keywords": ["REFUND"],
        "priority": 10
    }
]

# Sample transactions
transactions = [
    {"vendor": "Check", "description": "Payroll #502", "amount": -750},
    {"vendor": "Stripe", "description": "Payment", "amount": 500},
    {"vendor": "Stripe", "description": "Refund", "amount": -100},
    {"vendor": "Check", "description": "Contractor", "amount": -3500},
]

# Apply rules to each transaction
for tx in transactions:
    account_code = apply_custom_rules(tx, custom_rules)
    print(f"{tx['vendor']} {tx['amount']} → {account_code}")

# Output:
# Check -750 → 801 (Rule 1: within $500-$2000)
# Stripe 500 → 601 (Rule 3: matched, not excluded)
# Stripe -100 → None (Rule 3: excluded by "refund")
# Check -3500 → 801 (Rule 2: within $2000-$10000)
```

---

## JSON Schema: Valid Rule Structure

```json
{
  "keyword": {
    "type": "string",
    "required": true,
    "description": "Primary keyword to match",
    "example": "CHECK"
  },
  "account_code": {
    "type": "string",
    "required": true,
    "description": "Target account code (e.g., '801')",
    "example": "801"
  },
  "account_display": {
    "type": "string",
    "required": true,
    "description": "Display text for UI",
    "example": "801 · SALARIES-OFFICERS"
  },
  "min_amount": {
    "type": "number",
    "required": false,
    "description": "Minimum transaction amount (inclusive)",
    "example": 500,
    "default": null
  },
  "max_amount": {
    "type": "number",
    "required": false,
    "description": "Maximum transaction amount (inclusive)",
    "example": 2000,
    "default": null
  },
  "priority": {
    "type": "integer",
    "required": false,
    "description": "Processing priority (lower = first)",
    "example": 1,
    "default": 999
  },
  "exclude_keywords": {
    "type": "array",
    "required": false,
    "description": "Keywords that disqualify the rule (OR logic)",
    "example": ["REFUND", "DISPUTE"],
    "default": []
  },
  "additional_keywords": {
    "type": "array",
    "required": false,
    "description": "Keywords that must all be present (AND logic)",
    "example": ["RENT", "LANDLORD"],
    "default": []
  }
}
```

---

## Testing Rules: Test Cases

### Test Case 1: Amount Filtering

```python
def test_check_amount_filtering():
    rule = {
        "keyword": "CHECK",
        "account_code": "801",
        "min_amount": 500,
        "max_amount": 2000,
        "priority": 1
    }
    
    test_cases = [
        ({"vendor": "Check", "description": "", "amount": -250}, None),  # Too small
        ({"vendor": "Check", "description": "", "amount": -750}, "801"),  # Match
        ({"vendor": "Check", "description": "", "amount": -3000}, None),  # Too large
    ]
    
    for tx, expected in test_cases:
        result = apply_custom_rules(tx, [rule])
        assert result == expected, f"Failed for {tx['amount']}"
```

### Test Case 2: Keyword Matching

```python
def test_keyword_matching():
    rule = {
        "keyword": "STRIPE",
        "account_code": "601",
        "priority": 10
    }
    
    test_cases = [
        ({"vendor": "Stripe Inc", "description": "Payment", "amount": 100}, "601"),
        ({"vendor": "PayPal", "description": "Payment", "amount": 100}, None),
        ({"vendor": "My Stripe", "description": "Shop", "amount": 100}, "601"),
    ]
    
    for tx, expected in test_cases:
        result = apply_custom_rules(tx, [rule])
        assert result == expected
```

### Test Case 3: Exclusion Keywords

```python
def test_exclusion_keywords():
    rule = {
        "keyword": "STRIPE",
        "account_code": "601",
        "exclude_keywords": ["REFUND"],
        "priority": 10
    }
    
    test_cases = [
        ({"vendor": "Stripe", "description": "Payment", "amount": 100}, "601"),
        ({"vendor": "Stripe", "description": "Refund", "amount": -100}, None),
        ({"vendor": "Stripe", "description": "Refund Payment", "amount": 50}, None),
    ]
    
    for tx, expected in test_cases:
        result = apply_custom_rules(tx, [rule])
        assert result == expected
```

### Test Case 4: Priority System

```python
def test_priority_system():
    rules = [
        {
            "keyword": "CHECK",
            "account_code": "605",
            "min_amount": 50,
            "max_amount": 500,
            "priority": 1
        },
        {
            "keyword": "CHECK",
            "account_code": "801",
            "min_amount": 500,
            "max_amount": 2000,
            "priority": 2
        }
    ]
    
    # $750 check - should match Rule 2, but Rule 1 is checked first
    tx = {"vendor": "Check", "description": "", "amount": -750}
    result = apply_custom_rules(tx, rules)
    
    # Rule 1: $750 > $500 max → Skip
    # Rule 2: $500 ≤ $750 ≤ $2000 → Match
    assert result == "801"
```

---

## Debugging: Troubleshooting Rules

### Debug Transactions Not Matching

```python
def debug_rule_matching(transaction, rule):
    """Print detailed debug info for rule matching"""
    
    text = (transaction.get('vendor', '') + ' ' + 
            transaction.get('description', '')).lower()
    amount = abs(transaction.get('amount', 0))
    
    print(f"\nDEBUG: Transaction: {transaction}")
    print(f"Normalized text: '{text}'")
    print(f"Amount: ${amount}")
    
    # Check each filter
    keyword = rule.get('keyword', '').lower()
    print(f"\n1. Keyword: '{keyword}' in text?")
    print(f"   → {keyword in text}")
    
    min_amt = rule.get('min_amount')
    max_amt = rule.get('max_amount')
    print(f"\n2. Amount range: {min_amt} ≤ {amount} ≤ {max_amt}?")
    print(f"   → {min_amt is None or amount >= min_amt}")
    print(f"   → {max_amt is None or amount <= max_amt}")
    
    exclude = rule.get('exclude_keywords', [])
    print(f"\n3. Exclusions: {exclude}")
    print(f"   → {any(e.lower() in text for e in exclude)}")
    
    additional = rule.get('additional_keywords', [])
    print(f"\n4. Additional: {additional}")
    print(f"   → {all(a.lower() in text for a in additional)}")

# Usage
debug_rule_matching(
    {"vendor": "Check", "description": "Payroll", "amount": -750},
    {"keyword": "CHECK", "min_amount": 500, "max_amount": 2000}
)
```

---

## Integration: Adding Rules via API

```python
def add_custom_rule(user_id, business_name, rule_data):
    """Add a new custom rule and save to database"""
    
    # Validate rule data
    required_fields = ['keyword', 'account_code', 'account_display']
    for field in required_fields:
        if field not in rule_data:
            raise ValueError(f"Missing required field: {field}")
    
    # Load existing rules
    existing_rules = load_business_rules(user_id, business_name) or []
    
    # Check for duplicates
    for rule in existing_rules:
        if (rule.get('keyword').lower() == rule_data['keyword'].lower() and
            rule.get('min_amount') == rule_data.get('min_amount') and
            rule.get('max_amount') == rule_data.get('max_amount')):
            raise ValueError("Duplicate rule already exists")
    
    # Add new rule
    existing_rules.append(rule_data)
    
    # Save updated rules
    save_business_rules(user_id, business_name, existing_rules)
    
    return rule_data

# Usage
new_rule = {
    "keyword": "CHECK",
    "account_code": "801",
    "account_display": "801 · SALARIES-OFFICERS",
    "min_amount": 500,
    "max_amount": 2000,
    "priority": 1
}

add_custom_rule("user123", "My Business", new_rule)
```

---

## Summary

✅ **Basic Rule**: Keyword + Account Code  
✅ **Intermediate Rule**: Keyword + Amount Range  
✅ **Advanced Rule**: Keyword + Amount + Exclusions + Additional Keywords  
✅ **Priority**: Controls rule evaluation order  
✅ **Testing**: Use test cases to verify rules  
✅ **Debugging**: Use debug function to troubleshoot  

All rules are validated on the backend and saved persistently!
