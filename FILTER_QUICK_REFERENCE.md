# 🔍 Filter Quick Reference

## One-Page Filter Cheat Sheet

### Filter Controls

| Control | Purpose | Example |
|---------|---------|---------|
| **📅 Date Range** | Limit to specific period | Jan 1 - Mar 31, 2025 |
| **💵 Transaction Type** | Deposits/Withdrawals/Both | Show only income |
| **💰 Amount Range** | Min/Max dollar amounts | $100 - $5,000 |
| **🔎 Search** | Find by vendor/keyword | "Amazon", "Stripe" |
| **📊 Account Code** | Filter by category | 601 · SALES |

### Action Buttons

| Button | Action |
|--------|--------|
| **✅ Apply Filter** | Execute all filter criteria at once |
| **🔄 Reset Filter** | Clear all filters, show all transactions |
| **🔒 Lock Filter** | Preserve filter for exports/printing |
| **🔓 Unlock Filter** | Allow filter changes again |

### Visual Indicators

| Indicator | Meaning |
|-----------|---------|
| 🟢 **Green Banner** | Filter is active |
| 🔵 **Blue Banner** | Showing all transactions |
| 🔒 **Lock Icon** | Filter settings preserved |
| **Filter Breakdown** | Shows what was filtered out |

### Common Use Cases

```
📊 Quarterly Report:
  1. Set dates: Q1 (Jan 1 - Mar 31)
  2. Click "Apply Filter"
  3. Click "Lock Filter"
  4. Download all reports

💰 Large Transactions:
  1. Set min amount: $1000
  2. Click "Apply Filter"
  3. Review results

🏢 Vendor Analysis:
  1. Search: "Vendor Name"
  2. Click "Apply Filter"
  3. Check transaction details

💵 Income Only:
  1. Type: "Deposits Only"
  2. Click "Apply Filter"
  3. Review deposit summary
```

### Filter Logic

**AND Logic:** Transaction must pass ALL filters
```
✓ Date in range
  AND
✓ Type matches
  AND
✓ Amount in range
  AND
✓ Search text found (if provided)
  AND
✓ Account code matches (if selected)
```

### Pro Tips

✅ **DO:**
- Lock filter before exporting multiple reports
- Use search for quick vendor lookups
- Check filter breakdown to understand results
- Reset filter between different analyses

❌ **DON'T:**
- Forget to unlock when switching tasks
- Apply filter without reviewing criteria
- Export without checking filter status
- Ignore the filter status banner

### Exports with Filters

When filter is active and locked:
- ✅ All CSV downloads use filtered data
- ✅ Print view shows filtered results
- ✅ Filenames include "_filtered" suffix
- ✅ P&L reports reflect filtered period
- ✅ All tabs show filtered transactions

---

**Quick Access:** Scroll to "Advanced Transaction Filter" section after uploading statements

**Remember:** The green/blue banner always shows your current filter status!
