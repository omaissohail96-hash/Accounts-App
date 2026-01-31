# 🎯 Enhanced Automatic Filtering System - User Guide

## Overview

The **Enhanced Automatic Filtering System** allows you to set up sorting rules **once** that automatically categorize transactions for all future uploads. Think of it as creating custom filters that learn your categorization preferences.

---

## 🚀 Quick Start (3 Steps)

### Step 1: Identify the Pattern
Think about a common transaction type in your expenses (checks, PayPal, rent, utilities, etc.).

### Step 2: Create the Rule
In the **"Setup New Sorting Rule"** section:
1. Enter the **keyword/vendor** that appears in these transactions
2. Select the **target account category** where it should go
3. (Optional) Add filters for amount, keywords, or exclusions

### Step 3: Done ✅
The system immediately applies your rule to:
- All current transactions in your file
- Every future transaction you upload

---

## 📋 Two Configuration Methods

### Method 1: Step-by-Step (Recommended for Beginners)

Perfect if you prefer guided setup:

```
STEP 1: What to Look For?
↓
Keyword/Vendor Name: "Check"

STEP 2: Where Should It Go?
↓
Account Code: "801 · Salaries"

STEP 3: Any Additional Conditions?
↓
☐ Filter by Transaction Amount?
☐ Use Additional Keywords?
```

**Advantages:**
- Walks you through each decision
- Validates input at each step
- Creates clear, simple rules

---

### Method 2: Advanced (Full Control)

For users who want all options at once:

```
Primary Keyword: "Check"
Target Account Code: "801 · Salaries"
Min Amount: $500
Max Amount: $2000
Priority: 10
Additional Keywords: salary, paycheck
Exclusion Keywords: reversal, cancelled
```

**Advantages:**
- All parameters visible at once
- Fine-grained control
- Better for complex scenarios

---

## ⚙️ Understanding Each Parameter

### 1. **Primary Keyword** (Required)
**What it is:** The main search term that identifies these transactions

**Examples:**
- "Check" - for check payments
- "Stripe" - for Stripe transactions
- "Rent" - for rent payments
- "Utilities" - for utility bills

**How it works:**
- Case-insensitive matching
- Searches transaction description/vendor field
- Single keyword is required

---

### 2. **Target Account Code** (Required)
**What it is:** Where matching transactions should be categorized

**Available Categories:**
- 801 - SALARIES
- 802 - WAGES
- 803 - EMPLOYEE BENEFITS
- 804 - CONTRACTS
- (... and many more)

**How it works:**
- Select from dropdown list
- Transactions matching this rule → moved to this category
- If no match, transaction stays in "999 · OTHER EXPENSES"

---

### 3. **Amount Range** (Optional)

#### Minimum Amount
**When to use:** Only match transactions **larger than** a certain amount

**Examples:**
- Large checks ≥ $500 → SALARIES
- Small checks < $500 → SUPPLIES
- ATM withdrawals ≥ $200 → Cash Management

**Default:** 0 (no minimum)

#### Maximum Amount
**When to use:** Only match transactions **smaller than** a certain amount

**Examples:**
- Transactions ≤ $100 → Office Supplies
- Transactions ≤ $1000 → Marketing (not contracts)
- Small payments < $50 → Meals

**Default:** 0 (no maximum)

**Pro Tip:** Combine min + max to create ranges:
- Min: $500, Max: $1000 → Mid-size expenses
- Min: $1000, Max: $5000 → Large expenses

---

### 4. **Priority** (Optional, Advanced Only)
**What it is:** Processing order when multiple rules match

**Range:** 1 (highest) to 999 (lowest)

**How it works:**
1. System checks rules in priority order
2. **First matching rule wins** (first-match-wins)
3. Lower numbers = checked first

**Examples:**
```
Priority 1: "Stripe" + "refund" → REFUNDS (catch refunds first)
Priority 10: "Stripe" → SALES (then catch normal sales)
Priority 999: "misc payment" → OTHER (catch-all)
```

**Default:** 999 (process last)

**When to use:**
- Multiple rules with overlapping keywords
- Need specific rule to match before general one
- Avoiding mis-categorization

---

### 5. **Additional Keywords** (Optional)
**What it is:** Extra words that **ALL must be present** for rule to match

**Logic:** AND operation (all required)

**Examples:**

**Rule:** 
- Primary: "payment"
- Additional: "rent, landlord"
- Match: "Landlord payment for rent" ✅
- Match: "Landlord received payment" ✅
- Match: "Payment to supplier" ❌ (missing "landlord")

**Rule:**
- Primary: "Stripe"
- Additional: "charge, card"
- Match: "Stripe charge to card" ✅
- Match: "Stripe payout" ❌ (missing "charge" and "card")

**Syntax:**
- Comma-separated: `rent, landlord, payment`
- Case-insensitive
- Spaces trimmed automatically

---

### 6. **Exclusion Keywords** (Optional)
**What it is:** Words that **skip** this rule if **ANY** are found

**Logic:** OR operation (any excludes)

**Examples:**

**Rule:**
- Primary: "Check"
- Exclusion: "NSF, reversal, cancelled"
- Match: "Check 1234" ✅
- Skip: "Check reversal NSF" ❌ (found exclusion)
- Skip: "Check cancelled" ❌ (found exclusion)

**Rule:**
- Primary: "Stripe"
- Exclusion: "test, sandbox"
- Match: "Stripe transaction $50" ✅
- Skip: "Stripe test transaction" ❌ (test mode)
- Skip: "Stripe sandbox payment" ❌ (sandbox)

**Syntax:**
- Comma-separated: `refund, test, invalid`
- Case-insensitive
- Single keyword triggers skip

---

## 💡 Real-World Use Cases

### Use Case 1: Size-Based Check Categorization

**Challenge:** Different check amounts should go to different categories

**Solution:**

| Rule | Keyword | Amount | → Category |
|------|---------|--------|-----------|
| 1 | Check | ≥$1000 | SALARIES |
| 2 | Check | $500-$1000 | WAGES |
| 3 | Check | <$500 | SUPPLIES |

**Setup:**
```
Rule 1:
- Keyword: "Check"
- Min: $1000, Max: 0
- Account: SALARIES
- Priority: 1

Rule 2:
- Keyword: "Check"
- Min: $500, Max: $1000
- Account: WAGES
- Priority: 2

Rule 3:
- Keyword: "Check"
- Min: 0, Max: $500
- Account: SUPPLIES
- Priority: 3
```

---

### Use Case 2: Stripe Payment Processing

**Challenge:** Stripe has deposits AND refunds, need to separate

**Solution:**

```
Rule 1 (Priority 1):
- Keyword: "Stripe"
- Exclude: "refund, reversal, payout"
- Account: SALES
- Priority: 1

Rule 2 (Priority 2):
- Keyword: "Stripe"
- Additional: "refund"
- Account: REFUNDS
- Priority: 2

Rule 3 (Priority 3):
- Keyword: "Stripe"
- Additional: "payout"
- Account: BANK_TRANSFER
- Priority: 3
```

---

### Use Case 3: Rent Payment with Confirmation

**Challenge:** Avoid categorizing random "payment" transactions as rent

**Solution:**

```
Rule:
- Keyword: "payment"
- Additional: "landlord, rent, property"
- Min: $800 (your typical rent)
- Max: $1200 (rent range)
- Account: RENT
```

This ensures:
- Word "payment" present ✅
- AND ("landlord" OR "rent" OR "property") present ✅
- AND amount between $800-$1200 ✅

---

### Use Case 4: Utilities with Exclusions

**Challenge:** Utility bills sometimes refund, don't mix them

**Solution:**

```
Rule:
- Keyword: "utility, electric, water, gas"
- Exclude: "refund, credit, reversal"
- Min: $50 (actual bills, not adjustments)
- Account: UTILITIES
```

---

## 📊 How Rules Are Applied

### Processing Pipeline

```
For each transaction:
  ↓
Check ALL rules in PRIORITY order
  ↓
For each rule (priority 1 → 999):
  ├─ Is primary keyword found? NO → try next rule
  ├─ Is primary keyword found? YES
  │  ├─ Check amount range → NO → try next rule
  │  ├─ Check amount range → YES
  │  │  ├─ Check exclusion keywords → found ANY → try next rule
  │  │  ├─ Check exclusion keywords → found NONE
  │  │  │  ├─ Check additional keywords → missing ANY → try next rule
  │  │  │  ├─ Check additional keywords → has ALL
  │  │  │  │  └─ ✅ MATCH! Apply this rule
  │  │  │  │     └─ Move to target account code
  │  │  │  │     └─ Skip remaining rules
  └─ No rules matched
     └─ Keep in "999 · OTHER EXPENSES"
```

### Key Principles

1. **First Match Wins**: Once a rule matches, stop checking
2. **Priority Order**: Lower priority numbers checked first
3. **ALL filters must pass**: Every filter must succeed to match
4. **Case Insensitive**: "Check", "check", "CHECK" all match
5. **Partial matches OK**: "Check payment" matches keyword "Check"

---

## ✅ Best Practices

### DO ✅
- **Keep keywords specific** - "Check" not just "Ch"
- **Use round amounts** - Check amount ranges like $500, $1000
- **Start simple** - Add complexity only if needed
- **Test with a few rules** - Before creating many
- **Use priorities** - If rules might overlap
- **Document patterns** - Note why each rule exists

### DON'T ❌
- **Use too many keywords** - Filters become too restrictive
- **Mix unrelated conditions** - Keep each rule focused
- **Forget about exclusions** - These prevent errors
- **Ignore priorities** - Let specific rules run first
- **Set max amount = 0** - Always use actual values
- **Over-complicate** - Start with simple, add complexity later

---

## 🔧 Advanced Tips

### Tip 1: Debugging Unmatched Transactions

**Problem:** Rule not matching expected transactions

**Check:**
1. Is keyword spelled correctly?
2. Is it actually in the transaction description?
3. Check vendor/description field carefully
4. Does amount fall in range?
5. Are exclusion keywords preventing match?

**Solution:** Add debug rule with minimal filters first:
```
Keyword: "part of name"
Account: TEST_CATEGORY
```
See if it catches the transaction, then add filters.

---

### Tip 2: Overlapping Rules

**Problem:** Multiple rules could match same transaction

**Solution:** Use PRIORITY

```
Rule 1 (Priority 5): "Stripe" + exclude "refund"
Rule 2 (Priority 10): "Stripe" + additional "refund"
Rule 3 (Priority 20): "Stripe" (catch-all)
```

The system will:
1. Check Rule 1 first (if no refund keyword)
2. Check Rule 2 second (if has refund keyword)
3. Check Rule 3 last (any other Stripe)

---

### Tip 3: Testing Rules Safely

**Before committing:** 
1. Create rule with simple keyword only
2. Check if it matches expected transactions
3. Add filters gradually
4. Check results after each change

---

## 📈 Workflow for New Categories

### Goal: Automatically categorize a new expense type

**Step 1: Analyze Transactions**
- Review 5-10 recent transactions of this type
- Note common patterns in vendor/description
- Identify distinguishing characteristics

**Step 2: Identify Pattern**
- What keyword always appears?
- Do amounts cluster in a range?
- Are there related keywords?
- Any false positives to avoid?

**Step 3: Create Base Rule**
- Keyword: Most common identifier
- Account: Target category
- Add amount range if amounts vary

**Step 4: Refine with Filters**
- Add exclusions for false positives
- Add additional keywords for specificity
- Set priority if overlapping

**Step 5: Test**
- Check if matching correct transactions
- Verify none incorrectly categorized
- Review edge cases

**Step 6: Monitor**
- Check next upload cycle
- Look for missed transactions
- Update rule if needed

---

## 🚨 Common Mistakes & Fixes

### Mistake 1: Amount Range Too Narrow

**Wrong:** Min: $500, Max: $501
- Result: Only matches exact $500-$501

**Better:** Min: $500, Max: 0
- Result: Matches $500 and up

**Best:** Min: $500, Max: $1000
- Result: Matches $500-$1000 range

---

### Mistake 2: Additional Keywords Too Strict

**Wrong:** Additional: "payment, rent, monthly, landlord"
- Result: Requires ALL four words (probably impossible)

**Better:** Additional: "rent, landlord"
- Result: Requires "rent" OR "landlord"

**Note:** Use exclusions for "must NOT" conditions

---

### Mistake 3: No Priority on Similar Rules

**Problem:** Have "Check" rules for different amounts but same priority
- Result: First one matches always, others never used

**Fix:** Set different priorities:
```
Priority 1: Check, Min: $1000
Priority 2: Check, Min: $500, Max: $1000
Priority 3: Check, Min: 0, Max: $500
```

---

### Mistake 4: Keyword Too Generic

**Wrong:** Keyword: "A"
- Result: Matches almost everything ("A thousand dollars", "Bank", etc.)

**Better:** Keyword: "ATM"
- Result: Only matches ATM transactions

**Best:** Keyword: "ATM Withdrawal"
- Result: Precise matching

---

## 📞 Troubleshooting

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Rule not matching | Keyword not in description | Check actual transaction text, refine keyword |
| Too many false matches | Keyword too generic | Make keyword more specific |
| Amount not filtering | Min/Max set to 0 | Set actual amount values |
| Rule never used | Lower priority rule always matches first | Adjust priorities |
| Transactions still in "Other" | No rule matches | Review missing patterns, create new rule |
| Rule disappeared | Accidental delete | Re-create the rule (no undo) |

---

## 🎓 Learning Path

**Day 1: Basic Setup**
1. Create 1-2 simple rules with just keyword + account
2. Verify they match expected transactions
3. See how existing transactions update

**Day 2: Add Filters**
1. Create rule with amount range
2. Test size-based categorization
3. Adjust min/max values

**Day 3: Advanced**
1. Add exclusion keywords for false positives
2. Create overlapping rules with priorities
3. Use additional keywords for specificity

**Ongoing: Monitor & Refine**
1. Check each upload for missed transactions
2. Add rules for new patterns
3. Update existing rules as needed

---

## 💾 Data Persistence

Your rules are:
- ✅ **Saved automatically** after each creation
- ✅ **Persistent** across sessions
- ✅ **Applied immediately** to current transactions
- ✅ **Applied automatically** to future uploads
- ❌ **Cannot be undone** - delete is permanent

---

## 🎯 Key Takeaways

1. **Setup Once** → Works Forever
   - Create rules now, they apply to all future uploads

2. **Flexible Filtering** → Handles Complexity
   - Amount ranges, keyword matching, exclusions for any scenario

3. **Priority Control** → Predictable Results
   - Specific rules run first, catch-alls run last

4. **Auto-Apply** → Saves Time
   - No manual categorization needed

5. **Persistent** → Consistent
   - Same rules applied consistently every time

---

## 📚 Additional Resources

See documentation files for:
- **CATEGORIZATION_SCHEMA.md** - All available account codes
- **SCHEDULE_C_CATEGORIZER.md** - Tax categorization rules
- **README.md** - General system overview

---

**Questions?** Review the error message or check the troubleshooting section above.

Good luck with your automatic categorization setup! 🎉
