# 🔍 Professional Filter System Guide

## Overview
The Accounts App now features a **professional-grade, multi-criteria filter system** that allows precise transaction filtering with all the accuracy and efficiency of enterprise accounting software.

## ✨ Key Features

### 1. **Single Apply Filter Button** ✅
- All filter criteria are applied together with one click
- No more multiple button presses or confusion
- Instant feedback showing exactly how many transactions match
- Filter breakdown shows which criteria filtered out which transactions

### 2. **Lock Filter for Printing** 🔒
- **Lock Filter button** preserves your filter settings
- When locked:
  - Filter controls are disabled
  - All exports (CSV, PDF) use filtered data
  - Print view shows filtered results only
  - Clear visual indicator shows filter is locked
- **Unlock Filter button** to make changes again

### 3. **Accurate Date Matching** 📅
- Automatically detects statement coverage dates
- Works with single or multiple uploaded statements
- Shows clear date range: "Statement Coverage: Jan 01, 2025 → Dec 31, 2025"
- Defaults to first day of starting month and last day of ending month
- Handles year-end transitions correctly

### 4. **Advanced Filter Criteria**

#### **Date Range Filter** 📆
- Select any date range within statement coverage
- Automatically validates dates
- Shows how many days covered

#### **Transaction Type Filter** 💵
- All Transactions (default)
- Deposits Only
- Withdrawals Only
- Perfect for analyzing income or expenses separately

#### **Amount Range Filter** 💰
- Set minimum and maximum amounts
- Automatically detects possible amount range from your data
- Great for finding large transactions or filtering out small ones
- Example: Find all transactions between $500 and $5,000

#### **Vendor/Keyword Search** 🔎
- Search across vendor name, description, and raw transaction data
- Case-insensitive matching
- Instant text-based filtering
- Example: Search "Amazon" to find all Amazon transactions

#### **Account Code Filter** 📊
- Filter by specific account codes (Sales, Rent, Utilities, etc.)
- Dropdown with all available account codes
- "All Account Codes" shows everything
- Perfect for analyzing specific expense categories

## 🎯 How to Use

### Basic Workflow

1. **Upload your bank statements** as usual
2. **Scroll to the Filter section** (appears after processing)
3. **Set your filter criteria:**
   - Choose date range
   - Select transaction type
   - Set amount range
   - Enter search keywords (optional)
   - Select account code (optional)
4. **Click "Apply Filter"** button
5. **Review the results:**
   - See how many transactions matched
   - View filter breakdown
   - Check the active filter summary

### Advanced Usage

#### Lock Filter for Reports
1. Set up your filters
2. Click **"🔒 Lock Filter"**
3. Generate reports, exports, or print
4. All outputs will use filtered data
5. Click **"🔓 Unlock Filter"** to make changes

#### Reset Filter
- Click **"🔄 Reset Filter"** to clear all criteria
- Returns to showing all transactions
- Resets date range to full statement coverage

## 📊 Visual Indicators

### Filter Status Banner
- **Green banner** = Filter Active
  - Shows filtered transaction count
  - Shows date range
  - Shows lock status
- **Blue banner** = All Transactions
  - Shows total transaction count

### Active Filter Summary
After applying a filter, you'll see a detailed summary:
```
🔍 Active Filters:
📅 Dates: Jan 01, 2025 - Mar 31, 2025
💵 Type: All Transactions
💰 Amount: $0.00 - $10,000.00
🔎 Search: 'Amazon'
📊 Account: 601 · SALES
```

### Filter Breakdown
See exactly what was filtered:
```
• Date filter removed: 45 transactions
• Type filter removed: 12 transactions
• Amount filter removed: 8 transactions
• Search filter removed: 23 transactions
• Account code filter removed: 5 transactions
```

## 💡 Use Cases

### 1. **Quarterly Analysis**
- Set date range to Q1, Q2, Q3, or Q4
- Lock filter
- Export all reports for that quarter

### 2. **Expense Category Review**
- Select specific account code (e.g., "928 · RENT")
- See all rent payments in one view
- Check for duplicates or errors

### 3. **Large Transaction Audit**
- Set minimum amount to $1,000
- Review all large transactions
- Verify proper categorization

### 4. **Vendor Analysis**
- Search for specific vendor name
- See all transactions from that vendor
- Calculate total spent

### 5. **Income vs Expense Comparison**
- Filter for "Deposits Only"
- Note the total
- Switch to "Withdrawals Only"
- Compare totals

## 🔧 Technical Details

### Filter Logic
All filters use **AND logic** - a transaction must pass ALL filter criteria to be included:
1. Date must be within range AND
2. Type must match selection AND
3. Amount must be within range AND
4. Search text must be found (if provided) AND
5. Account code must match (if selected)

### Performance
- Filters are applied in memory (instant)
- Works efficiently with 10,000+ transactions
- No database queries needed
- Results cached until filter changes

### Data Integrity
- Original transactions never modified
- Filter creates a separate filtered list
- Can always reset to see all data
- Export functions respect filter state

## 🎨 Best Practices

1. **Start Broad, Then Narrow**
   - Begin with date range
   - Add more criteria as needed
   - Use "Apply Filter" to check results

2. **Lock Before Exporting**
   - Set up your perfect filter
   - Lock it
   - Export all needed reports
   - This ensures consistency across all exports

3. **Use Search for Quick Checks**
   - Type vendor name to see their transactions
   - Great for spot-checking specific items

4. **Review Filter Breakdown**
   - Shows which filters did most work
   - Helps optimize your filter criteria

5. **Reset When Switching Tasks**
   - Clear filter between different analyses
   - Prevents confusion about what you're viewing

## 🚀 Keyboard Shortcuts & Tips

- The filter state persists across tab switches
- Locked filters remain locked until you unlock them
- Filter is automatically applied to all tabs (Deposits, Withdrawals, P&L, etc.)
- Download filenames include "_filtered" suffix when filter is active

## 📝 Notes

- Filter applies to all views (Deposits, Withdrawals, P&L, All Transactions)
- Custom rules still work with filtered data
- Manual exclusions work independently of filters
- Opening balance is handled correctly regardless of filter

---

## Need Help?

If you encounter any issues or have suggestions for the filter system, please refer to the main README.md or contact support.

**Version:** 2.0 (Professional Filter System)
**Last Updated:** February 2026
