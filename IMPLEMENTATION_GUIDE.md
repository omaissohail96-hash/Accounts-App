# Practical Implementation Guide: Fix $500 Check Categorization

## Issue Summary
**Problem**: Check transactions of $500 are being categorized as "999 OTHER EXPENSES" instead of "801 SALARIES-OFFICERS"

**Root Cause**: No custom rule exists to handle check amounts in the $500+ range

**Solution**: Add custom rules with amount filters to the Custom Rules tab

---

## Step-by-Step Implementation

### Step 1: Open the Application

1. Start the Streamlit app
2. Navigate to **⚙️ Custom Rules** tab
3. Look for **"➕ Add New Rule"** section

---

### Step 2: Add Rule #1 - Medium Salary Checks ($500-$2000)

**Form Fields:**

| Field | Value | Why |
|-------|-------|-----|
| **Keyword/Vendor Name** | `CHECK` | Matches all transactions with "check" keyword |
| **Account Code** | `801 · SALARIES-OFFICERS` | Target account for salary checks |
| **Minimum Amount** | `500` | Only match checks $500 or more |
| **Maximum Amount** | `2000` | Only match checks up to $2000 |
| **Priority** | `1` | Process this rule first (highest priority) |

**Click:** ➕ Add Rule

**Expected Result:**
```
✅ Rule added: 'CHECK' → 801 · SALARIES-OFFICERS ($500-$2000) [Priority: 1]
```

**What This Does:**
- Any transaction with "CHECK" in vendor/description
- Amount between $500-$2000 (inclusive)
- Gets assigned to account 801 (SALARIES-OFFICERS)
- Processed before any other check rules

---

### Step 3: Add Rule #2 - Large Salary Checks ($2000+)

**Form Fields:**

| Field | Value | Why |
|-------|-------|-----|
| **Keyword/Vendor Name** | `CHECK` | Same keyword as Rule #1 |
| **Account Code** | `801 · SALARIES-OFFICERS` | Same account |
| **Minimum Amount** | `2000` | Catch larger checks |
| **Maximum Amount** | `10000` | Upper limit for typical payroll |
| **Priority** | `2` | Process after Rule #1 (if not matched) |

**Click:** ➕ Add Rule

**Expected Result:**
```
✅ Rule added: 'CHECK' → 801 · SALARIES-OFFICERS ($2000-$10000) [Priority: 2]
```

**What This Does:**
- Catches checks between $2000-$10000
- Assigned to same SALARIES account
- Acts as fallback if transaction amount exceeds $2000 limit

---

### Step 3b (Optional): Add Rule #3 - Small Checks (<$500)

**Form Fields:**

| Field | Value | Why |
|-------|-------|-----|
| **Keyword/Vendor Name** | `CHECK` | Same keyword |
| **Account Code** | `605 · SUPPLIES` | Different category for small checks |
| **Minimum Amount** | `50` | Minimum transaction size |
| **Maximum Amount** | `500` | Upper limit for "small" checks |
| **Priority** | `3` | Process last (lower priority) |

**Click:** ➕ Add Rule

**Expected Result:**
```
✅ Rule added: 'CHECK' → 605 · SUPPLIES ($50-$500) [Priority: 3]
```

**What This Does:**
- Separates small check payments from salaries
- Allows different categorization by amount
- Provides flexibility for various check types

---

### Step 4: Verify Rules Are Added

In **📋 Active Rules** section, you should see:

```
Total rules: 3 (sorted by priority)

✓ 1. 🔍 CHECK [≥ $500 & ≤ $2000] [Priority: 1]
     → 801 · SALARIES-OFFICERS

✓ 2. 🔍 CHECK [≥ $2000 & ≤ $10000] [Priority: 2]
     → 801 · SALARIES-OFFICERS

✓ 3. 🔍 CHECK [≥ $50 & ≤ $500] [Priority: 3]
     → 605 · SUPPLIES
```

---

### Step 5: Test with Sample Bank Statement

1. **Upload a test bank statement** with various check amounts:
   - Check for $150
   - Check for $750 ← This was the problem case
   - Check for $2500
   - Check for $50

2. **Review** the categorized transactions:
   - $150 check → Should be 605 SUPPLIES ✅
   - $750 check → Should be 801 SALARIES ✅ (This fixes the issue!)
   - $2500 check → Should be 801 SALARIES ✅
   - $50 check → Should be 605 SUPPLIES ✅

3. **Go to** 📊 P&L (Account Codes) tab to verify categorization

4. **Check** the "💼 By Account Code" section:
   - All salary checks ($500+) should be under 801
   - Small checks (<$500) should be under 605

---

## Advanced Customization (Optional)

### Scenario A: Exclude Refunded Checks

**Add exclusion keywords to prevent misclassification:**

| Field | Value |
|-------|-------|
| **Keyword** | `CHECK` |
| **Account Code** | `801 · SALARIES-OFFICERS` |
| **Min/Max Amount** | $500-$2000 |
| **Exclusion Keywords** | `REFUND, REVERSAL, CANCELLED` |

**Effect:** Checks with "REFUND" won't match this rule

---

### Scenario B: Vendor-Specific Rules

**For payroll service checks:**

| Field | Value |
|-------|-------|
| **Keyword** | `ADP` (or your payroll provider) |
| **Account Code** | `801 · SALARIES-OFFICERS` |
| **Min Amount** | `500` |
| **Max Amount** | `0` (no upper limit) |
| **Priority** | `1` |

**Effect:** Any ADP check over $500 automatically categorized as salaries

---

### Scenario C: Multiple Keywords Required

**For rent payments that specify both vendor AND purpose:**

| Field | Value |
|-------|-------|
| **Keyword** | `PAYMENT` |
| **Account Code** | `808 · RENT` |
| **Additional Keywords** | `LANDLORD, PROPERTY` |
| **Priority** | `5` |

**Effect:** Only matches if "PAYMENT" AND ("LANDLORD" or "PROPERTY") are present

---

## Verification Checklist

After implementing the rules, verify:

- [ ] Rules appear in **📋 Active Rules** section
- [ ] Rules are sorted by priority (1 before 2 before 3)
- [ ] Amount ranges are correct ($500-$2000, $2000-$10000, etc.)
- [ ] Account codes match intended categories (801 for salaries, etc.)
- [ ] Test transaction $750 check → categorizes as 801 SALARIES
- [ ] Small $50 check → categorizes as 605 SUPPLIES (if Rule #3 added)
- [ ] Rules persist after page refresh

---

## Troubleshooting

### Problem: $500 Check Still Goes to "999 OTHER EXPENSES"

**Cause 1: Rule not saved**
- Check that rules appear in **📋 Active Rules**
- If not, click ➕ Add Rule again

**Cause 2: Amount range is wrong**
- Verify Min Amount ≤ $500
- Verify Max Amount ≥ $500
- Example: If Min=$1000, checks of $500 won't match

**Cause 3: Keyword doesn't match transaction**
- The transaction must contain word "CHECK"
- If vendor says "Paycheck" instead of "Check #502", rule won't match
- Solution: Add additional rule with keyword "PAYCHECK" or "PAY"

**Cause 4: Rule has lower priority than another rule**
- Check priority numbers
- Lower number = checked first
- If Rule #3 has priority=1, it might match before Rule #1

### Problem: Rules Not Persisting After Upload

**Cause:** Browser session state cleared
- Rules are saved to `data/rules/user_[ID]_rules.json`
- If file not found, rules reset
- Solution: Verify file exists in data/rules directory

### Problem: Too Many Rules Matching

**Cause:** Keyword too broad
- If keyword is just "PAY", many unrelated transactions match
- Solution: Use more specific keywords like "PAYCHECK" or "PAYROLL"

---

## Before & After Comparison

### BEFORE (No Rules):
```
Transaction: "Check #502 - Payroll"
Amount: $750

Processing:
1. Check custom rules → NO RULES DEFINED → Skip
2. Fall back to default mapper → No logic for checks
3. Result: 999 OTHER EXPENSES ❌

P&L Report:
999 OTHER EXPENSES    $750.00
```

### AFTER (With Rules):
```
Transaction: "Check #502 - Payroll"
Amount: $750

Processing:
1. Check custom rules:
   ✓ Rule 1: CHECK keyword found ✓
   ✓ Amount $750 in range $500-$2000 ✓
   → Apply Rule 1 → 801 SALARIES-OFFICERS ✅
2. Stop processing (first match wins)
3. Result: 801 SALARIES-OFFICERS ✓

P&L Report:
801 SALARIES-OFFICERS    $750.00
```

---

## FAQ

**Q: Will these rules apply to existing transactions?**
A: Yes! Rules are applied automatically to all transactions (including existing ones from previous uploads) when you click ➕ Add Rule.

**Q: What if a check matches multiple rules?**
A: The rule with the **lowest priority number** is applied (first match wins). Other rules are ignored for that transaction.

**Q: Can I delete a rule?**
A: Yes! Each rule has a 🗑️ Delete button. Click it to remove a rule. Transactions will be re-mapped to default categories.

**Q: What about checks that don't have a specific amount pattern?**
A: Use a rule with Min Amount = $0 and Max Amount = $0 (or leave blank) to match all checks regardless of amount.

**Q: How do I handle different types of payments (checks, ACH, wire)?**
A: Create separate rules for each type:
- Rule: keyword="CHECK" → SALARIES
- Rule: keyword="ACH" → SALARIES
- Rule: keyword="WIRE" → OTHER EXPENSES

**Q: Can I have different salary accounts for different employees?**
A: The current system maps to ONE account code per rule. For employee-specific rules, you'd need:
- Rule 1: keyword="CHECK" + additional_keyword="JOHN" → 801 or specific code
- Rule 2: keyword="CHECK" + additional_keyword="JANE" → 801 or specific code

---

## Performance Notes

- ✅ Rules are processed in priority order (fastest first)
- ✅ First matching rule stops evaluation (efficient)
- ✅ Amount filters reduce false matches
- ✅ All rules saved to JSON file (persistent)

---

## Example Rule Combinations for Different Scenarios

### Scenario 1: Restaurant Business
```
Rule 1: STRIPE (payment processor) → SALES (priority 1)
  Min: $50, Max: $5000
  Exclude: REFUND, CHARGEBACK

Rule 2: CHECK (supplier payments) → SUPPLIES (priority 2)
  Min: $100, Max: $500

Rule 3: PAYCHECK (employee salary) → SALARIES (priority 3)
  Min: $1000, Max: $3000
```

### Scenario 2: Professional Services
```
Rule 1: QUICKBOOKS (accounting) → OFFICE EXPENSES (priority 1)
  Min: $0, Max: $500

Rule 2: CLIENT CHECK (client payment) → INCOME (priority 2)
  Min: $1000, Max: $50000
  Additional: DEPOSIT, PAYMENT

Rule 3: CONTRACTOR → CONTRACTORS (priority 3)
  Min: $500, Max: $10000
```

---

## Summary

✅ **Implementation**: Add 2-3 rules with amount filters to handle check categorization  
✅ **Testing**: Upload sample transactions with various check amounts  
✅ **Verification**: Confirm $500+ checks now map to SALARIES (801)  
✅ **Done**: Issue resolved! Checks now categorized by amount automatically  

The system now differentiates between small checks (supplies) and salary checks (salaries) based on amount ranges.
