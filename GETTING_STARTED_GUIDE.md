# 🚀 Getting Started - Automatic Transaction Filtering

## ⏱️ 10-Minute Quick Start

### What You're Getting
An automatic system that learns your expense categorization and applies it to all future transactions.

**Set up rules once → They apply forever ✨**

---

## 🎯 Step 1: Open the Custom Rules Tab (2 minutes)

1. Open your Accounts App in Streamlit
2. Look for the tabs at the top
3. Click on **"⚙️ Enhanced Custom Account Code Rules"** or similar

You should see:
```
⚙️ Automatic Transaction Sorting & Categorization
Set up automatic filtering rules ONCE - they'll automatically sort 
transactions into the right categories for all future uploads.
```

---

## 📝 Step 2: Create Your First Rule (5 minutes)

### Easiest Way: Use Step-by-Step Tab

#### STEP 1: What transactions do you want to categorize?
Think of something common in your expenses:
- Checks
- PayPal payments
- Stripe charges
- Rent payments
- Utility bills

**Example:** Let's say you want to categorize **Checks**

#### STEP 2: Enter the keyword

```
Keyword/Vendor Name: Check
```

Click on the input field and type: `Check`

#### STEP 3: Where should these go?

```
Account Code: 801 · SALARIES
```

Click the dropdown and select the category.

#### STEP 4: Add optional filters (if needed)

For checks, you might want different amounts to go different places:
- Large checks ($1000+) → SALARIES
- Small checks (<$500) → SUPPLIES

To do this:
1. Check the box: "☐ Filter by Transaction Amount?"
2. Enter: Min: 1000, Max: 0 (means $1000 and up)
3. Account: 801 · SALARIES

#### STEP 5: Create the rule

Click the blue button: **✅ Create Sorting Rule**

You should see:
```
✅ Sorting rule created: 'Check' → 801 · SALARIES ($1000+)
```

---

## 3️⃣ Step 3: Verify It Works (2 minutes)

### Check Active Rules Section

Scroll down to **"📋 Active Rules"**

You should see your new rule listed:
```
▼ 🔍 Check [≥ $1000]
  → 801 · SALARIES
```

### Check Your Transactions

Look at your transaction list - check transactions over $1000 should now show as categorized under "801 · SALARIES" instead of "999 · OTHER EXPENSES"

---

## 1️⃣ Step 4: Add More Rules (ongoing)

Repeat the same process for other transaction types:

```
Rule 2: Stripe → SALES
Rule 3: PayPal → SALES  
Rule 4: Rent → RENT
Rule 5: Utilities → UTILITIES
```

**Each rule:**
1. Takes 2-3 minutes to create
2. Works forever (applies to all future uploads)
3. Updates existing transactions immediately

---

## 💡 Common First Rules

### Rule 1: Check Payments
```
Keyword: Check
Account: SALARIES (or appropriate category)
```

### Rule 2: PayPal
```
Keyword: PayPal
Account: SALES (or PAYMENT_PROCESSOR)
```

### Rule 3: Stripe
```
Keyword: Stripe
Account: SALES (or appropriate category)
```

### Rule 4: Rent
```
Keyword: rent
Account: RENT (or UTILITIES)
```

### Rule 5: Utilities
```
Keyword: electric, water, gas, utility
Account: UTILITIES
```

---

## ❓ Frequently Asked Questions

### Q: How long do the rules last?
**A:** Forever! Once created, they persist across sessions and apply to all future uploads.

### Q: Do existing transactions get updated?
**A:** Yes! When you create a new rule, it immediately updates matching transactions in your current file.

### Q: Can I delete a rule?
**A:** Yes. Find it in "Active Rules" and click the [🗑️ Delete] button. Deletion is permanent.

### Q: What if the rule doesn't work?
**A:** See the Troubleshooting section below.

### Q: Can I create multiple rules for the same keyword?
**A:** Yes! Use priorities. The rule with priority "1" is checked first. Example:
- Priority 1: "Check" + Min $1000 → SALARIES
- Priority 2: "Check" + Min $500, Max $999 → WAGES
- Priority 3: "Check" → SUPPLIES

### Q: What does "Additional Keywords" do?
**A:** Requires multiple keywords to be present. Example:
- Keyword: "payment"
- Additional: "rent, landlord"
- Matches: "landlord payment for rent" ✅
- Skips: "payment to someone" ❌ (missing "landlord")

### Q: What does "Exclusion Keywords" do?
**A:** Skips the rule if any exclusion keyword is found. Example:
- Keyword: "Check"
- Exclusion: "NSF, cancelled"
- Matches: "Check 1234" ✅
- Skips: "Check NSF" ❌
- Skips: "Check cancelled" ❌

---

## 🔧 Troubleshooting

### Problem: Rule not catching expected transactions

**Possible causes:**
1. **Keyword doesn't match**: Check if the exact keyword appears in your transaction description
2. **Amount filter too restrictive**: If you set Min/Max amounts, verify transactions fall in that range
3. **Exclusion keyword blocking**: Check if any exclusion keywords appear in the transaction
4. **Priority issue**: If multiple rules could match, lower priority rules run first

**Solution:**
1. First, create a simple rule with just keyword + account
2. Verify it catches something
3. Then gradually add filters
4. Test after each change

### Problem: Too many false matches

**Solution:**
1. Make the keyword more specific
2. Add exclusion keywords for the false positive cases
3. Add additional keywords to require multiple terms

### Problem: Rule disappeared

**Solution:**
The rule was likely accidentally deleted. Simply recreate it using the same parameters.

### Problem: Amount filter not working

**Possible causes:**
1. Min and Max both set to 0 (0 means "no limit")
2. Transactions fall outside the specified range

**Solution:**
- For "$500 and up": Min: 500, Max: 0
- For "up to $500": Min: 0, Max: 500
- For "$500-$1000": Min: 500, Max: 1000

---

## 📚 Learn More

After your first rule works, check out:

1. **QUICK_FILTERING_REFERENCE.md** - Fast reference guide
2. **ENHANCED_FILTERING_SYSTEM.md** - Detailed user guide
3. **RULE_CONFIGURATION_EXAMPLES.md** - 50+ ready-to-use examples

---

## ✅ Success Checklist

After your first rule:
- [ ] Rule appears in "Active Rules" section
- [ ] Matching transactions updated in your list
- [ ] Rule persists after closing app
- [ ] Rule applies when you upload new data

---

## 🎉 You're Done!

You now have an automatic transaction categorization system. As you add more rules, you'll spend less time manually categorizing and more time on analysis.

### Next Steps:
1. **Today**: Create 1-2 simple rules
2. **This Week**: Add rules for your top 5 expense types
3. **Going Forward**: Add new rules as new transaction patterns appear

---

## 💾 Your Setup is Saved

Every rule you create is:
- ✅ Saved automatically
- ✅ Applied immediately
- ✅ Persistent across sessions
- ✅ Ready for future uploads

---

## 🚀 Ready to Set Up?

Go back to your Accounts App and:
1. Click the **Custom Rules tab**
2. Click **"📋 Step-by-Step"**
3. Enter your first rule
4. Click **"✅ Create Sorting Rule"**

That's it! You're now using automatic transaction filtering! 🎊

---

## 📞 Need Help?

**For specific rule setup:**
→ Check RULE_CONFIGURATION_EXAMPLES.md

**For understanding how rules work:**
→ Check ENHANCED_FILTERING_SYSTEM.md

**For quick lookup:**
→ Check QUICK_FILTERING_REFERENCE.md

**For UI walkthrough:**
→ Check UI_VISUAL_GUIDE.md

---

**Happy sorting! Your transactions will be organized automatically from now on.** ✨
