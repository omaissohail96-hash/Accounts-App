# Custom Rules Filter Functionality - Visual Flowchart

## 1. FILTER EXECUTION FLOW

```
┌─────────────────────────────────────────────────────────────────┐
│ NEW TRANSACTION UPLOADED                                        │
│ Vendor: "Check #502 - John Doe"                                │
│ Description: "Payroll check"                                    │
│ Amount: $750.00                                                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ NORMALIZE & EXTRACT                                             │
│ Text = "check john doe payroll check"                          │
│ Amount = 750.00 (absolute value)                               │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ LOAD & SORT CUSTOM RULES                                       │
│ Rule 1: CHECK, min=$500, max=$2000, priority=1 (SALARIES)     │
│ Rule 2: CHECK, min=$2000, max=$10000, priority=2 (SALARIES)   │
│ Rule 3: STRIPE, priority=999 (SALES)                           │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
        ╔════════════════════════════╗
        ║  FOR EACH RULE (by priority)║
        ╚════╤═══════════════════════╝
             │
             ▼
    ┌─────────────────────────┐
    │ FILTER 1: KEYWORD MATCH │
    │ "CHECK" in "check..." ? │◄─── YES ──┐
    │ NO ────►  SKIP RULE     │           │
    └─────────────────────────┘           │
                                          ▼
                           ┌──────────────────────────┐
                           │ FILTER 2: AMOUNT RANGE   │
                           │ 500 ≤ 750 ≤ 2000 ?       │◄─── YES ──┐
                           │ NO ──► SKIP RULE         │           │
                           └──────────────────────────┘           │
                                                                  ▼
                                   ┌───────────────────────────┐
                                   │ FILTER 3: EXCLUSIONS      │
                                   │ Has exclude keywords?     │
                                   │ (check for REFUND, etc)   │◄─── NO ──┐
                                   │ YES ──► SKIP RULE         │          │
                                   └───────────────────────────┘          │
                                                                          ▼
                                      ┌──────────────────────────┐
                                      │ FILTER 4: ADDITIONS      │
                                      │ Has additional keywords? │◄─── NO ──┐
                                      │ (ALL must match)         │          │
                                      │ NO ──► Skip if present   │          │
                                      └──────────────────────────┘          │
                                                                            ▼
                                           ✅ ALL FILTERS PASSED
                                           
                                           transaction.account_code = "801"
                                           transaction._mapped_by_rule = True
                                           STOP (first rule wins)
```

---

## 2. AMOUNT FILTER VISUALIZATION

### Check Transaction Amounts & Categorization

```
Amount Scale ($)
0─────50────100────150────200────300────500────750──1000──1500──2000──3000──5000──10000
│     │      │      │      │      │      │     │   │     │     │     │     │     │
                                        ▲     ▲   ▲      ▲     ▲
                                        │     │   │      │     │
                                      MIN    │   │      │     └─ Rule 2 MAX
                                            │   │      │
                                     Check #1   │      │
                                    ($750)    Check   Check
                                               #2    #3
                                              ($50)  ($3500)

RULES:
┌──────────────────────────────────────────────────────────────┐
│ RULE 1: SALARY CHECKS ($500-$2000)                          │
│ ────────────────────────────────                            │
│ Min: $500 │████████████────────│ Max: $2000                │
│ Account: 801 · SALARIES-OFFICERS                            │
│ Priority: 1                                                   │
│                                                               │
│ ✓ Check #1 ($750) matches   ✓✓✓                            │
│ ✗ Check #2 ($50) doesn't match  (too small)                │
│ ✗ Check #3 ($3500) doesn't match  (too large)              │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ RULE 2: LARGE SALARY CHECKS ($2000-$10000)                  │
│ ──────────────────────────────────────                       │
│ Min: $2000 │──────────────────────│ Max: $10000            │
│ Account: 801 · SALARIES-OFFICERS                            │
│ Priority: 2                                                   │
│                                                               │
│ ✗ Check #1 ($750) doesn't match  (too small)               │
│ ✗ Check #2 ($50) doesn't match  (too small)                │
│ ✓ Check #3 ($3500) matches   ✓✓✓                           │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ RULE 3: SMALL CHECKS ($50-$500)                             │
│ ─────────────────────────────                               │
│ Min: $50 │████│ Max: $500                                   │
│ Account: 605 · SUPPLIES                                      │
│ Priority: 3                                                   │
│                                                               │
│ ✗ Check #1 ($750) doesn't match  (too large)               │
│ ✓ Check #2 ($50) matches   ✓✓✓                             │
│ ✗ Check #3 ($3500) doesn't match  (too large)              │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. FILTER COMBINATION EXAMPLES

### Example 1: Amount Range (%)

```
TRANSACTION:
├─ Vendor: "Payroll Services Inc"
├─ Description: "Check #0523 - Employee Salary"
└─ Amount: $1,250.00

RULES TO CHECK:
┌─────────────────────────────────────────────┐
│ RULE: Salary Checks                         │
├─────────────────────────────────────────────┤
│ Keyword: "CHECK"        ✓ MATCHED           │
│ Min Amount: $500        ✓ ($1,250 ≥ $500)   │
│ Max Amount: $2,000      ✓ ($1,250 ≤ $2,000) │
├─────────────────────────────────────────────┤
│ ✅ RESULT: Map to 801 · SALARIES-OFFICERS  │
└─────────────────────────────────────────────┘
```

### Example 2: Exclusion Keywords

```
TRANSACTION:
├─ Vendor: "Stripe"
├─ Description: "Stripe Refund - Transaction"
└─ Amount: $-150.00

RULES TO CHECK:
┌──────────────────────────────────────────────┐
│ RULE: Stripe Sales                           │
├──────────────────────────────────────────────┤
│ Keyword: "STRIPE"              ✓ MATCHED     │
│ Exclude Keywords: ["REFUND"]   ✗ FOUND       │
├──────────────────────────────────────────────┤
│ ❌ RESULT: SKIP - Rule excluded due to       │
│           "REFUND" keyword                   │
│                                              │
│ Falls through to next rule or default mapper │
└──────────────────────────────────────────────┘
```

### Example 3: Additional Keywords (ALL must match)

```
TRANSACTION:
├─ Vendor: "Property Management LLC"
├─ Description: "Payment for Rent"
└─ Amount: $-1,500.00

RULES TO CHECK:
┌──────────────────────────────────────────────┐
│ RULE: Rent Payment                           │
├──────────────────────────────────────────────┤
│ Keyword: "PAYMENT"             ✓ MATCHED     │
│ Additional Keywords:                         │
│   - "RENT"                     ✓ MATCHED     │
│   - "LANDLORD"                 ✗ NOT FOUND   │
├──────────────────────────────────────────────┤
│ ❌ RESULT: SKIP - Not all additional        │
│           keywords present                   │
└──────────────────────────────────────────────┘

ALTERNATIVE RULE:
┌──────────────────────────────────────────────┐
│ RULE: Rent (Simple)                          │
├──────────────────────────────────────────────┤
│ Keyword: "RENT"                ✓ MATCHED     │
│ Additional Keywords: NONE                    │
├──────────────────────────────────────────────┤
│ ✅ RESULT: Map to 808 · RENT                │
└──────────────────────────────────────────────┘
```

### Example 4: Priority System

```
TRANSACTION:
├─ Vendor: "Acme Corp"
├─ Description: "Check for Contractor Services"
└─ Amount: $-1,200.00

MULTIPLE MATCHING RULES (by priority):
┌──────────────────────────────────────────────┐
│ Priority 1: CHECK (min $500, max $2000)     │
│            ✓ KEYWORD MATCHES                 │
│            ✓ AMOUNT IN RANGE                 │
│            → Account: 801 (SALARIES)         │
│                                              │
│ 🏁 FIRST MATCH WINS - STOP HERE             │
│                                              │
│ Priority 2: CHECK (min $2000, max $10000)   │
│            (Would match, but skipped)       │
│                                              │
│ Priority 3: CONTRACTOR                      │
│            (Would match, but skipped)       │
└──────────────────────────────────────────────┘

✅ FINAL RESULT: Account 801 · SALARIES
   (Higher priority rule wins)
```

---

## 4. FILTER LOGIC PSEUDOCODE

```python
def apply_filters_to_transaction(transaction, custom_rules):
    """
    Apply custom rules filters with priority-based matching
    Returns: account_code if matched, None otherwise
    """
    
    # Normalize transaction text (case-insensitive)
    text = normalize(transaction.vendor + transaction.description)
    tx_amount = abs(transaction.amount)
    
    # Sort rules by priority (lower number = higher priority)
    sorted_rules = sort_by(custom_rules, 'priority')
    
    for rule in sorted_rules:
        
        # ──────────────────────────────────────────────
        # FILTER 1: PRIMARY KEYWORD (REQUIRED)
        # ──────────────────────────────────────────────
        if rule.keyword not in text:
            continue  # Skip to next rule
        
        
        # ──────────────────────────────────────────────
        # FILTER 2: AMOUNT RANGE (OPTIONAL)
        # ──────────────────────────────────────────────
        if rule.has('min_amount'):
            if tx_amount < rule.min_amount:
                continue  # Amount too small
        
        if rule.has('max_amount'):
            if tx_amount > rule.max_amount:
                continue  # Amount too large
        
        
        # ──────────────────────────────────────────────
        # FILTER 3: EXCLUSION KEYWORDS (ANY match = skip)
        # ──────────────────────────────────────────────
        if rule.has('exclude_keywords'):
            for excl_keyword in rule.exclude_keywords:
                if excl_keyword in text:
                    continue(to_next_rule)  # Skip this rule
        
        
        # ──────────────────────────────────────────────
        # FILTER 4: ADDITIONAL KEYWORDS (ALL must match)
        # ──────────────────────────────────────────────
        if rule.has('additional_keywords'):
            all_found = True
            for add_keyword in rule.additional_keywords:
                if add_keyword not in text:
                    all_found = False
                    break
            
            if not all_found:
                continue  # Not all additional keywords found
        
        
        # ✅ ALL FILTERS PASSED
        # ──────────────────────────────────────────────
        return rule.account_code  # MATCH! Stop processing
    
    
    # No rule matched
    return None  # Falls through to default mapper
```

---

## 5. WHY $500 CHECKS FALL TO "OTHER EXPENSES"

### Root Cause Chain

```
Step 1: Upload Bank Statement
  └─ Transaction: "Check #502", $750

Step 2: Check Custom Rules
  └─ NO RULE for "CHECK" keyword with amount filter
  └─ Result: No custom rule matches

Step 3: Fall Through to Default Mapper
  └─ AccountCodeMapper.get_account_code()
  └─ No specific logic for checks
  └─ Result: Defaults to "999 OTHER EXPENSES"

Step 4: Transaction Categorized Incorrectly
  └─ $750 check → 999 OTHER EXPENSES ❌
  └─ Should be → 801 SALARIES ✓

Step 5: Next Bank Upload
  └─ Transaction NOT persisted in custom rules
  └─ Same rule miss occurs
  └─ Repeats to "OTHER EXPENSES" ❌
```

### Solution

```
CREATE RULE:
┌─────────────────────────────────────────────┐
│ Keyword: CHECK                              │
│ Min Amount: 500                             │
│ Max Amount: 2000                            │
│ Account Code: 801 · SALARIES-OFFICERS      │
│ Priority: 1                                 │
└─────────────────────────────────────────────┘

NEW FLOW:
Step 2: Check Custom Rules
  └─ ✓ RULE found: "CHECK" with $500-$2000 range
  └─ ✓ $750 falls in range
  └─ ✓ MATCH: Assign to 801 SALARIES

Step 4: Transaction Categorized Correctly
  └─ $750 check → 801 SALARIES ✓
```

---

## 6. AMOUNT PERCENTAGE EXAMPLES

### Different Check Scenarios

```
Scenario Analysis: Checks from $50 to $5,000

┌─────────────────┬──────────┬──────────────────────────────┬──────────────────┐
│ Check #         │ Amount   │ Rule Matching Process        │ Final Category   │
├─────────────────┼──────────┼──────────────────────────────┼──────────────────┤
│ 1001            │ $50.00   │ Rule1: $50 < $500 min ✗      │ DEFAULT MAPPER   │
│                 │          │ Rule2: $50 < $500 min ✗      │ → 999 OTHER EXP  │
│                 │          │ Rule3: $50 MATCHED ✓         │ → 605 SUPPLIES   │
├─────────────────┼──────────┼──────────────────────────────┼──────────────────┤
│ 1002            │ $250.00  │ Rule1: $250 < $500 min ✗     │ DEFAULT MAPPER   │
│                 │          │ Rule2: $250 < $500 min ✗     │ → 999 OTHER EXP  │
│                 │          │ Rule3: $50-$500 MATCH ✓      │ → 605 SUPPLIES   │
├─────────────────┼──────────┼──────────────────────────────┼──────────────────┤
│ 1003 (PROBLEM)  │ $500.00  │ Rule1: $500 IN $500-$2k ✓✓✓  │ 801 SALARIES     │
│                 │          │ MATCH! Stop here             │ ✅ CORRECT       │
├─────────────────┼──────────┼──────────────────────────────┼──────────────────┤
│ 1004            │ $750.00  │ Rule1: $750 IN $500-$2k ✓✓✓  │ 801 SALARIES     │
│                 │          │ MATCH! Stop here             │ ✅ CORRECT       │
├─────────────────┼──────────┼──────────────────────────────┼──────────────────┤
│ 1005            │ $1500.00 │ Rule1: $1500 IN $500-$2k ✓✓✓ │ 801 SALARIES     │
│                 │          │ MATCH! Stop here             │ ✅ CORRECT       │
├─────────────────┼──────────┼──────────────────────────────┼──────────────────┤
│ 1006            │ $2000.00 │ Rule1: $2000 IN $500-$2k ✓✓✓ │ 801 SALARIES     │
│                 │          │ MATCH! Stop here             │ ✅ CORRECT       │
├─────────────────┼──────────┼──────────────────────────────┼──────────────────┤
│ 1007            │ $3500.00 │ Rule1: $3500 > $2k max ✗     │ 801 SALARIES     │
│                 │          │ Rule2: $3500 IN $2k-$10k ✓✓✓ │ ✅ CORRECT       │
│                 │          │ MATCH! Stop here             │                  │
├─────────────────┼──────────┼──────────────────────────────┼──────────────────┤
│ 1008            │ $5000.00 │ Rule1: $5000 > $2k max ✗     │ 801 SALARIES     │
│                 │          │ Rule2: $5000 IN $2k-$10k ✓✓✓ │ ✅ CORRECT       │
│                 │          │ MATCH! Stop here             │                  │
└─────────────────┴──────────┴──────────────────────────────┴──────────────────┘

KEY INSIGHT:
With proper rules defined, ALL checks are categorized correctly
based on their amount ranges.
```

---

## 7. QUICK REFERENCE CARD

| Term | Definition | Example |
|------|-----------|---------|
| **Keyword Match** | Primary text to search for | "CHECK", "STRIPE", "RENT" |
| **Min Amount** | Lowest amount to match (inclusive) | $500 = match $500, $501, $502... |
| **Max Amount** | Highest amount to match (inclusive) | $2000 = match $1999, $2000 |
| **Priority** | Order of rule evaluation | 1 = highest (checked first), 999 = lowest |
| **Additional Keywords** | ALL must be present (AND logic) | ["RENT", "PAYMENT"] both needed |
| **Exclusion Keywords** | ANY present = skip rule | ["REFUND"] = skip if found |
| **% Checks** | Different checks by amount % | $50 (5%), $500 (50%), $5000 (500%) |

