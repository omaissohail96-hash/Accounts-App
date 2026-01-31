# Custom Rules Filter Analysis & Enhancement

## Current Issue
**Problem**: Transactions with checks of $500 are falling into "Other Expenses" instead of "Salaries" category when new bank statements are uploaded.

---

## Current Filter Logic (Lines 1523-1607)

### How the Custom Rules System Works:

```python
def reapply_custom_rules():
    # 1. Load business rules from database
    # 2. Normalize vendor + description text to lowercase
    # 3. For each transaction, apply rules with these checks:
```

### Current Filter Checks (Priority Order):

1. **Primary Keyword Match** (Required)
   - Checks if keyword exists in normalized transaction text
   - `if not kw or kw not in text: continue`

2. **Amount Range Filters** (Optional)
   ```python
   min_amt = rule.get("min_amount")
   max_amt = rule.get("max_amount")
   
   if min_amt is not None and tx_amount < min_amt:
       continue  # Amount too small for this rule
   if max_amt is not None and tx_amount > max_amt:
       continue  # Amount too large for this rule
   ```

3. **Exclusion Keywords** (Optional - AND logic)
   - Skips rule if ANY exclusion keyword matches
   - `if any(_norm(excl) in text for excl in exclude_kws): continue`

4. **Additional Keywords** (Optional - AND logic)
   - ALL additional keywords must match
   - `if not all(_norm(kw_add) in text for kw_add in additional_kws): continue`

---

## Root Cause Analysis

### Why $500 Checks Fall into "Other Expenses":

1. **No Check Amount Rule Defined**: The system has no rule like:
   ```json
   {
     "keyword": "CHECK",
     "min_amount": 500,
     "max_amount": 10000,
     "account_code": "801",
     "account_display": "801 · SALARIES-OFFICERS"
   }
   ```

2. **Fallback Behavior**: When no custom rule matches:
   - Transaction gets mapped via `AccountCodeMapper.get_account_code()`
   - This likely defaults to "999 OTHER EXPENSES"

3. **No Priority Differentiation**: Multiple check amounts aren't distinguished:
   - Small checks (<$500) → Could be "Supplies"
   - Medium checks ($500-$2000) → Should be "Salaries"
   - Large checks (>$2000) → Could be special category

---

## Enhanced Filter Solution

### 1. **Create Check Amount Rules**

Add these rules to your Custom Rules tab:

| Rule | Keyword | Min Amount | Max Amount | Account Code | Purpose |
|------|---------|-----------|-----------|--------------|---------|
| 1 | CHECK | 500 | 2000 | 801 | Salary checks in typical range |
| 2 | CHECK | 2000 | 10000 | 801 | Large salary checks |
| 3 | CHECK | 100 | 500 | 605 | Small check payments (Supplies) |

### 2. **Priority System**

Lower priority = processed first

```python
# Execution order:
1. Priority: 1   → Checks $500-$2000 (SALARIES)
2. Priority: 2   → Checks $2000-$10000 (SALARIES)
3. Priority: 3   → Checks $100-$500 (SUPPLIES)
4. Priority: 999 → Default fallback rules
```

### 3. **Enhanced Filter Logic Pseudocode**

```python
def apply_custom_rules_enhanced(transaction):
    # Step 1: Normalize transaction text
    text = normalize(vendor + description)
    tx_amount = abs(transaction.amount)
    
    # Step 2: Load and sort rules by priority
    sorted_rules = sort_by_priority(business_rules)
    
    for rule in sorted_rules:
        # FILTER 1: Primary keyword MUST match
        if rule.keyword NOT in text:
            continue
        
        # FILTER 2: Amount must be within range
        if tx_amount < rule.min_amount:
            continue  # Too small
        if tx_amount > rule.max_amount:
            continue  # Too large
        
        # FILTER 3: Exclusion keywords (skip if ANY match)
        if rule.has_exclusion_keywords():
            if any(excl in text for excl in rule.exclude_keywords):
                continue  # Excluded
        
        # FILTER 4: Additional keywords (ALL must match)
        if rule.has_additional_keywords():
            if not all(add in text for add in rule.additional_keywords):
                continue  # Missing required keywords
        
        # ✅ ALL FILTERS PASSED - APPLY RULE
        transaction.account_code = rule.account_code
        transaction._mapped_by_rule = True
        break  # First matching rule wins (priority-based)
    
    # Step 3: If no rule matched, clear previous rule
    if no_rule_matched and transaction._mapped_by_rule:
        transaction.account_code = None
        transaction._mapped_by_rule = False
```

---

## Filter Functionality Breakdown

### Amount Range Checks (%) Examples

**Scenario**: Different checks of varying amounts

```
CHECK #001: $150.00
├─ Keyword "CHECK" ✓
├─ $150 < $500 min ✗ → Skip SALARIES rule
└─ Falls through to default → "999 OTHER EXPENSES"

CHECK #002: $750.00
├─ Keyword "CHECK" ✓
├─ $500 ≤ $750 ≤ $2000 ✓
└─ Matches SALARIES rule → "801 SALARIES-OFFICERS" ✓

CHECK #003: $3500.00
├─ Keyword "CHECK" ✓
├─ $500 ≤ $3500 ≤ $2000 ✗ → Skip medium rule
├─ $3500 > $2000 but < $10000 ✓
└─ Matches large SALARIES rule → "801 SALARIES-OFFICERS" ✓
```

### Filter Combination Examples

**Exclude Refunds from Sales Rule:**
```json
{
  "keyword": "STRIPE",
  "account_code": "601",
  "exclude_keywords": ["REFUND", "CHARGEBACK"],
  "additional_keywords": ["PAYMENT", "CHARGE"]
}
```

**Effect:**
- ✓ "STRIPE PAYMENT CHARGE" → Sales
- ✗ "STRIPE REFUND" → Skip this rule
- ✗ "STRIPE ONLY" (missing PAYMENT/CHARGE) → Skip this rule

---

## Implementation Steps

### Step 1: Define Check Amount Rules

In **⚙️ Custom Rules** tab, add:

1. **Rule 1: Medium Salary Checks**
   - Keyword: `CHECK`
   - Min Amount: `500`
   - Max Amount: `2000`
   - Account: `801 · SALARIES-OFFICERS`
   - Priority: `1` (highest)

2. **Rule 2: Large Salary Checks**
   - Keyword: `CHECK`
   - Min Amount: `2000`
   - Max Amount: `10000`
   - Account: `801 · SALARIES-OFFICERS`
   - Priority: `2`

3. **Rule 3: Small Checks (Optional)**
   - Keyword: `CHECK`
   - Min Amount: `50`
   - Max Amount: `500`
   - Account: `605 · SUPPLIES`
   - Priority: `3`

### Step 2: Test the Rules

1. Upload a bank statement with mixed check amounts
2. Verify:
   - Checks $500-$2000 → SALARIES ✓
   - Checks $2000+ → SALARIES ✓
   - Checks <$500 → SUPPLIES ✓

### Step 3: Add Exclusions (Optional)

Refine rules to exclude certain types:
```
Rule: CHECK + exclude_keywords: ["TRANSFER", "REFUND"]
Effect: Only categorizes real check payments, not transfers
```

---

## Code Changes Required

### Current Code (Lines 1548-1585)

The current filter implementation is **already correct** for amount filtering. However, you need to:

1. ✅ Add custom rules with proper min/max amounts in the UI
2. ✅ Set appropriate priority values (lower = higher priority)
3. ✅ Test with actual transaction data

### No code changes needed in the filter logic itself

The enhancement is **configuration-based**, not code-based.

---

## Enhanced Features Explanation

| Feature | How It Works | Use Case |
|---------|------------|----------|
| **Amount Filtering** | Matches min ≤ amount ≤ max | Differentiate check amounts |
| **Priority System** | Lower priority # checked first | More specific rules process first |
| **Exclusion Keywords** | Skip if ANY keyword matches | Avoid misclassification |
| **Additional Keywords** | Match if ALL keywords present | Precise pattern matching |

---

## Prevention Strategy

### Why Transactions Fall to "Other Expenses":

1. **No matching custom rule** → Falls through to default mapper
2. **Account code mapper doesn't have check logic** → Defaults to 999
3. **No persistent account code** → Resets on new upload

### Solution:

✅ Create persistent custom rules with amount filters  
✅ Set high priority for check rules  
✅ Test with sample transactions before production upload

---

## Testing Recommendations

```python
# Test data
test_transactions = [
    ("Check #001", 150.00),    # Expected: SUPPLIES (605)
    ("Check #002", 750.00),    # Expected: SALARIES (801)
    ("Check #003", 3500.00),   # Expected: SALARIES (801)
    ("Check #004", 50.00),     # Expected: SUPPLIES (605)
]

# For each: verify account_code matches expected value
```

---

## Summary

The **custom rules filter is working correctly**. The issue is **missing rule definitions** for check transactions with amount thresholds. 

**Action Required:**
1. Open **⚙️ Custom Rules** tab
2. Add check amount rules (see "Implementation Steps")
3. Reupload bank statement
4. Verify checks $500+ now map to SALARIES (801)
