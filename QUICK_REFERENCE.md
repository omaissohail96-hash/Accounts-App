# Quick Reference: Custom Rules Filter

## 🎯 Problem
**$500 checks falling into "999 OTHER EXPENSES" instead of "801 SALARIES"**

---

## ✅ Solution
Add 2-3 custom rules with amount filters (takes 2 minutes)

---

## 📋 Rules to Add

### Rule 1: Medium Salary Checks
```
Keyword:        CHECK
Account:        801 · SALARIES-OFFICERS
Min Amount:     500
Max Amount:     2000
Priority:       1
```

### Rule 2: Large Salary Checks
```
Keyword:        CHECK
Account:        801 · SALARIES-OFFICERS
Min Amount:     2000
Max Amount:     10000
Priority:       2
```

### Rule 3 (Optional): Small Checks
```
Keyword:        CHECK
Account:        605 · SUPPLIES
Min Amount:     50
Max Amount:     500
Priority:       3
```

---

## 🔧 How Filters Work

### Filter Pipeline
```
1. Keyword Match? ────► Check if "CHECK" in text
2. Amount Range? ────► Check if $500 ≤ amount ≤ $2000
3. Exclusions? ────► Check if any exclude keywords present
4. Additions? ────► Check if all additional keywords present
5. Apply Rule! ────► Assign account code if all pass
```

### Priority System
- **Lower = Higher Priority** (checked first)
- **First match wins** (other rules ignored)
- Example: Priority 1 before Priority 2 before Priority 3

---

## 💡 Amount Filter Examples

| Amount | Rule 1 ($500-$2k) | Rule 2 ($2k-$10k) | Rule 3 ($50-$500) | Result |
|--------|-------------------|-------------------|-------------------|--------|
| $50    | ✗ Too small       | ✗ Too small       | ✓ Match           | 605 SUPPLIES |
| $250   | ✗ Too small       | ✗ Too small       | ✓ Match           | 605 SUPPLIES |
| $500   | ✓ Match           | ✗ Too small       | ✓ Match           | **801 SALARIES** (Rule 1 priority) |
| $750   | ✓ Match           | ✗ Too small       | ✗ Too large       | **801 SALARIES** ✅ FIXED |
| $2000  | ✓ Match           | ✓ Match           | ✗ Too large       | **801 SALARIES** (Rule 1 priority) |
| $3500  | ✗ Too large       | ✓ Match           | ✗ Too large       | **801 SALARIES** |
| $5000  | ✗ Too large       | ✓ Match           | ✗ Too large       | **801 SALARIES** |
| $10000 | ✗ Too large       | ✓ Match           | ✗ Too large       | **801 SALARIES** |
| $15000 | ✗ Too large       | ✗ Too large       | ✗ Too large       | 999 OTHER EXPENSES (default) |

---

## 🚀 Step-by-Step Setup

### Step 1: Open App & Go to Custom Rules Tab
```
⚙️ Custom Rules tab → "➕ Add New Rule"
```

### Step 2: Fill Rule 1
```
Keyword:        CHECK
Account:        801 · SALARIES-OFFICERS
Advanced Filters:
  Min Amount:   500
  Max Amount:   2000
  Priority:     1
Click: ➕ Add Rule
```

### Step 3: Fill Rule 2
```
Keyword:        CHECK
Account:        801 · SALARIES-OFFICERS
Advanced Filters:
  Min Amount:   2000
  Max Amount:   10000
  Priority:     2
Click: ➕ Add Rule
```

### Step 4: Verify in Active Rules
```
✓ Rule 1: 🔍 CHECK [≥ $500 & ≤ $2000] [Priority: 1]
         → 801 · SALARIES-OFFICERS

✓ Rule 2: 🔍 CHECK [≥ $2000 & ≤ $10000] [Priority: 2]
         → 801 · SALARIES-OFFICERS
```

### Step 5: Upload Test Bank Statement
```
Verify:
- Check $750 → 801 SALARIES ✅
- Check $3500 → 801 SALARIES ✅
```

---

## 📊 Filter Combinations

### Combination 1: Keyword Only
```json
{
  "keyword": "CHECK",
  "account_code": "801",
  "priority": 999
}
```
**Matches:** Any transaction with "check" (no amount filter)

### Combination 2: Keyword + Amount
```json
{
  "keyword": "CHECK",
  "min_amount": 500,
  "max_amount": 2000,
  "account_code": "801",
  "priority": 1
}
```
**Matches:** "check" AND $500-$2000

### Combination 3: Keyword + Exclusions
```json
{
  "keyword": "STRIPE",
  "exclude_keywords": ["REFUND"],
  "account_code": "601",
  "priority": 10
}
```
**Matches:** "stripe" AND NOT "refund"

### Combination 4: Keyword + Additions
```json
{
  "keyword": "PAYMENT",
  "additional_keywords": ["RENT", "LANDLORD"],
  "account_code": "808",
  "priority": 5
}
```
**Matches:** "payment" AND "rent" AND "landlord"

---

## 🎓 Key Concepts

| Term | Meaning |
|------|---------|
| **Keyword** | Main word to search for in transaction |
| **Min Amount** | Lowest transaction amount to match (≥) |
| **Max Amount** | Highest transaction amount to match (≤) |
| **Priority** | Rule order (1=first, 999=last) |
| **Exclude** | Skip rule if ANY of these keywords found |
| **Additional** | Skip rule if NOT ALL of these keywords found |
| **% Checks** | Different check amounts ($50, $500, $5000, etc.) |

---

## ❓ Common Questions

**Q: Will this affect existing transactions?**
A: ✅ Yes, all transactions (past & future) are re-categorized

**Q: How many rules can I create?**
A: ✅ Unlimited, but keep it organized

**Q: Can I edit a rule after adding?**
A: ❌ Delete and re-add with new values

**Q: What if amounts overlap?**
A: ✅ Priority number decides (lower = first)

**Q: Do rules apply automatically?**
A: ✅ Yes, immediately when you click "Add Rule"

**Q: Can rules be saved permanently?**
A: ✅ Yes, saved to database automatically

---

## 🔍 Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| $750 check still in OTHER | No rule created | Add Rule 1 with $500-$2000 range |
| Rule created but not working | Keyword doesn't match text | Check transaction contains "CHECK" keyword |
| Wrong account assigned | Rule not saved | Verify rule appears in "Active Rules" section |
| Amount filter ignored | Min/Max = 0 | Set specific amounts: Min=500, Max=2000 |
| Rule not persisted | Browser cache | Refresh page, rule should still be there |

---

## 📈 Expected Results

### Before Rules
```
P&L Report:
999 OTHER EXPENSES   $5,250
    Check #501       $250
    Check #502       $750  ← WRONG!
    Check #503       $3,500 ← WRONG!
```

### After Rules
```
P&L Report:
605 SUPPLIES         $250
801 SALARIES         $4,250  ← CORRECT!
    Check #502       $750   ← FIXED!
    Check #503       $3,500 ← FIXED!
```

---

## 🎯 Quick Checklist

- [ ] Open ⚙️ Custom Rules tab
- [ ] Click ➕ Add New Rule
- [ ] Fill Rule 1: CHECK, $500-$2000, 801, Priority 1
- [ ] Click ➕ Add Rule
- [ ] Fill Rule 2: CHECK, $2000-$10000, 801, Priority 2
- [ ] Click ➕ Add Rule
- [ ] Verify both rules in 📋 Active Rules
- [ ] Upload test bank statement
- [ ] Verify $750 check → 801 SALARIES ✅
- [ ] Verify $3500 check → 801 SALARIES ✅
- [ ] Done! Issue fixed 🎉

---

## 📚 Related Documents

- 📖 [CUSTOM_RULES_ANALYSIS.md](CUSTOM_RULES_ANALYSIS.md) - Full analysis
- 📊 [FILTER_FUNCTIONALITY_EXPLAINED.md](FILTER_FUNCTIONALITY_EXPLAINED.md) - Visual diagrams
- 🚀 [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) - Detailed setup
- 💻 [CODE_DEEP_DIVE.md](CODE_DEEP_DIVE.md) - Technical details
- 📋 [ANALYSIS_SUMMARY.md](ANALYSIS_SUMMARY.md) - Executive summary

---

## ⏱️ Time Estimate

| Task | Time |
|------|------|
| Read this guide | 2 min |
| Add 2 rules | 2 min |
| Test with sample data | 2 min |
| **Total** | **6 min** |

✅ **Issue completely resolved in under 10 minutes!**

---

**Status**: Ready to Implement | No Code Changes Needed | Configuration Only
