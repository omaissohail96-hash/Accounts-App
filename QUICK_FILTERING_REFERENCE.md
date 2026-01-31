# ⚡ Enhanced Filtering System - Quick Reference

## 🎯 5-Minute Setup Guide

### For Each New Expense Type:

```
1. KEYWORD (what to look for)
   Example: "Check", "Stripe", "Rent", "Utilities"

2. ACCOUNT CODE (where it goes)
   Pick from dropdown

3. FILTERS (optional, but powerful)
   Amount: Min $X, Max $Y
   Exclusions: skip if keyword found
   Additional: require keywords present
   Priority: process order (1=first, 999=last)

DONE ✅
```

---

## 📋 Setup Templates

### Template 1: Size-Based Categorization
```
Rule A:
- Keyword: "Check"
- Min: $1000, Max: 0
- Account: SALARIES
- Priority: 1

Rule B:
- Keyword: "Check"  
- Min: $500, Max: $1000
- Account: WAGES
- Priority: 2

Rule C:
- Keyword: "Check"
- Min: 0, Max: $500
- Account: SUPPLIES
- Priority: 3
```

### Template 2: Vendor + Amount
```
- Keyword: "Stripe"
- Min: $100, Max: 0
- Account: SALES
```

### Template 3: Pattern Matching (AND)
```
- Keyword: "payment"
- Additional: "rent, landlord"
- Min: $800, Max: $1200
- Account: RENT
```

### Template 4: Exclusion Handling
```
- Keyword: "Check"
- Exclude: "NSF, reversal, cancelled"
- Account: CHECKING_EXPENSE
```

### Template 5: Complex Logic
```
Priority 1 (run first):
- Keyword: "Stripe"
- Exclude: "refund, payout"
- Account: SALES

Priority 2 (if refund):
- Keyword: "Stripe"
- Additional: "refund"
- Account: REFUNDS

Priority 3 (catch-all):
- Keyword: "Stripe"
- Account: OTHER
```

---

## 🔑 Parameter Quick Guide

| Parameter | Required? | Examples | Notes |
|-----------|-----------|----------|-------|
| **Keyword** | ✅ YES | Check, Stripe, Rent | Primary search term |
| **Account** | ✅ YES | 801 SALARIES | Category destination |
| **Min Amount** | ❌ NO | 500, 1000 | 0 = no minimum |
| **Max Amount** | ❌ NO | 1000, 5000 | 0 = no maximum |
| **Priority** | ❌ NO | 1, 10, 100 | Lower = process first |
| **Additional** | ❌ NO | rent, landlord | ALL must be present |
| **Exclusion** | ❌ NO | refund, test | ANY triggers skip |

---

## ✅ Validation Checklist

Before clicking "Create Sorting Rule":

- [ ] Did you enter a keyword?
- [ ] Did you select an account code?
- [ ] Do the min/max amounts make sense?
- [ ] Are you excluding obvious false positives?
- [ ] Did you set priority if rules overlap?

---

## 🚀 Common Rules (Copy-Paste Ready)

### Rule: All Checks
```
Keyword: Check
Account: CHECKING_EXPENSE
```

### Rule: Large Checks (Salaries)
```
Keyword: Check
Min: $1000
Account: SALARIES
Priority: 1
```

### Rule: Small Checks (Supplies)
```
Keyword: Check
Max: $500
Account: SUPPLIES
Priority: 2
```

### Rule: Stripe Sales
```
Keyword: Stripe
Exclude: refund, payout
Account: SALES
Priority: 1
```

### Rule: Stripe Refunds
```
Keyword: Stripe
Additional: refund
Account: REFUNDS
Priority: 2
```

### Rule: PayPal
```
Keyword: PayPal
Account: PAYMENT_PROCESSOR
```

### Rule: Rent
```
Keyword: payment
Additional: rent
Min: $800
Max: $1200
Account: RENT
```

### Rule: Utilities
```
Keyword: utility
Exclude: refund
Min: $30
Account: UTILITIES
```

### Rule: Office Supplies
```
Keyword: office
Max: $500
Account: OFFICE_SUPPLIES
```

### Rule: Banking Fees
```
Keyword: fee
Max: $100
Account: BANK_FEES
```

---

## 🔄 Rule Application Logic

```
FOR each incoming transaction:
  FOR each rule (sorted by priority):
    IF keyword found
      AND amount in range (if set)
      AND exclusions NOT found
      AND all additional keywords found
    THEN
      Categorize to this rule's account
      STOP (don't check other rules)
```

---

## 📊 Example: Check Categorization

**Setup:**
```
Rule 1: Check, Min $1000 → SALARIES, Priority 1
Rule 2: Check, Min $500, Max $999 → WAGES, Priority 2  
Rule 3: Check, Max $500 → SUPPLIES, Priority 3
```

**Results:**
| Check Amount | Matches | Goes To |
|--------------|---------|---------|
| $100 | Rule 3 | SUPPLIES |
| $500 | Rule 2 or 3 | WAGES (lower priority) |
| $750 | Rule 2 | WAGES |
| $1000 | Rule 1 | SALARIES |
| $2000 | Rule 1 | SALARIES |

---

## 🎯 Setup Workflow

```
Step 1: Review Recent Transactions
  → What types do you see?
  → What amounts are typical?
  → What keywords are common?

Step 2: Create 1-2 Rules
  → Start simple
  → Use just keyword + account
  → See if they work

Step 3: Check Results
  → Do matching transactions update?
  → Any false positives?
  → Any missed transactions?

Step 4: Refine with Filters
  → Add amount ranges
  → Add exclusions for false positives
  → Set priorities if rules overlap

Step 5: Monitor Future Uploads
  → Check each new upload
  → Add rules for new patterns
  → Update if behavior changes
```

---

## ⚠️ Common Mistakes

| Mistake | ❌ Wrong | ✅ Correct |
|---------|----------|-----------|
| Too many keywords | `Additional: a, and, the, or` | `Additional: rent, landlord` |
| Both min & max zero | `Min: 0, Max: 0` | `Min: 500, Max: 0` (500+) |
| Wrong priority order | Catch-all at priority 1 | Catch-all at priority 999 |
| Too generic keyword | `Keyword: A` | `Keyword: Check` |
| No space for typos | Exact match only | Make keyword flexible |

---

## 🔍 Debug a Non-Matching Rule

**Issue: Rule not catching expected transactions**

```
1. Check the transaction description
   → Is your keyword actually there?
   → Is spelling correct?
   → Case matters? (No, it's case-insensitive)

2. Check the amount
   → Is it in the min/max range?
   → Did you set min/max correctly?
   → (0 means no limit)

3. Check exclusions
   → Is any exclusion keyword present?
   → Try removing exclusions, test again

4. Check additional keywords
   → Are ALL additional keywords present?
   → Try removing additional keywords, test again

5. Check priority
   → Is another rule matching first?
   → Lower priority numbers run first
   → Adjust priority to run earlier
```

---

## 💡 Pro Tips

- **Tip 1:** Create rules for the top 5-10 transaction types first
- **Tip 2:** Use priorities (1, 10, 100) to leave room for inserting new rules
- **Tip 3:** Test rules on one transaction type at a time
- **Tip 4:** Keep keywords short but specific
- **Tip 5:** Use amount ranges to handle variations
- **Tip 6:** Exclude common false positives immediately
- **Tip 7:** Monitor the first few uploads after creating rules

---

## 📱 Mobile Considerations

- Tab navigation works on mobile
- Use Step-by-Step tab on phones
- Advanced tab better for desktop

---

## 🔐 Important Notes

- ✅ Rules saved automatically
- ✅ Applied to all current transactions
- ✅ Applied to all future uploads
- ❌ Cannot undo deletion (delete is permanent)
- ❌ No rule versioning

---

## 📞 Quick Troubleshooting

| Q: My rule isn't working | A: Check if keyword is in transaction description |
| Q: Too many false matches | A: Make keyword more specific or add exclusions |
| Q: Amount filter ignored | A: Make sure min/max aren't both 0 |
| Q: Old transactions not updated | A: Rules only apply on next upload |
| Q: Where are my rules? | A: Check "Active Rules" section, they're there |

---

## 🎓 Level Up Path

**Level 1: Master Basic Rules**
- Create keyword → account mappings
- Understand exact matching

**Level 2: Add Amount Filters**
- Create size-based categorizations
- Use min/max combinations

**Level 3: Use Exclusions**
- Prevent false positives
- Skip special cases

**Level 4: Advanced Patterns**
- Combine additional keywords
- Use priorities effectively
- Build complex logic

**Level 5: Optimization**
- Monitor rule effectiveness
- Refine based on results
- Share templates with team

---

## 🎉 You're Ready!

Start with one rule and expand from there. The system is designed to grow with your needs.

**Good luck!**
