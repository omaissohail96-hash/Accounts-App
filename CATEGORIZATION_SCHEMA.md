# Transaction Categorization - Keyword Schema Documentation

## Overview

The transaction categorization system uses a keyword-based approach with rank-based priority and include/exclude logic to automatically categorize transactions into accounting codes.

## JSON Schema Structure

Each account code in `account_keywords.json` follows this structure:

```json
{
  "602": {
    "name": "RETURNS & ALLOWANCES",
    "rank": 1,
    "include_keywords": [
      "refund",
      "return",
      "chargeback",
      "reversal",
      "dispute"
    ],
    "exclude_keywords": []
  }
}
```

### Fields

- **name** (string): The human-readable account name
- **rank** (integer): Priority ranking where **1 = highest priority**. Lower numbers are checked first.
- **include_keywords** (array): List of keywords that trigger this category
- **exclude_keywords** (array): List of keywords that prevent this category from matching

## Categorization Algorithm

The system processes transactions using this logic:

1. **Text Normalization**
   - Combine vendor name and transaction description
   - Convert to lowercase
   - Trim whitespace

2. **Sort by Rank**
   - Categories are sorted by rank in ascending order (1, 2, 3, ...)
   - Lower rank = higher priority

3. **Process Each Category in Order**
   - **Step 1: Check Exclusions**
     - If ANY `exclude_keywords` match the normalized text → skip this category
   
   - **Step 2: Check Inclusions**
     - If ANY `include_keywords` match the normalized text → assign this category and **STOP**

4. **Default Behavior**
   - If no category matches, the transaction remains uncategorized
   - The system defaults to account code "999 · OTHER EXPENSES" for expenses
   - Or "601 · SALES" for income transactions

## Example Scenarios

### Example 1: Refund Processing (Exclusion Logic)

**Transaction:** "Stripe payment refund"

**Processing:**
1. Rank 1 (RETURNS & ALLOWANCES): 
   - Exclude: None match
   - Include: "refund" matches ✅
   - **Result:** 602 · RETURNS & ALLOWANCES

**Transaction:** "Stripe payment received"

**Processing:**
1. Rank 1 (RETURNS & ALLOWANCES): 
   - Include: No match
2. Rank 2 (SALES):
   - Exclude: None match
   - Include: "payment received" matches ✅
   - **Result:** 601 · SALES

### Example 2: Payroll vs Payroll Tax (Exclusion Logic)

**Transaction:** "Gusto payroll processing"

**Processing:**
1. ...
2. Rank 8 (SALARIES-OFFICERS):
   - Exclude: "tax" not in text ✅
   - Include: "payroll" matches ✅
   - **Result:** 801 · SALARIES-OFFICERS

**Transaction:** "Gusto payroll tax withholding"

**Processing:**
1. ...
2. Rank 8 (SALARIES-OFFICERS):
   - Exclude: "tax" matches ❌ → Skip this category
3. Rank 11 (PAYROLL TAXES):
   - Exclude: None match
   - Include: "payroll tax" matches ✅
   - **Result:** 821 · PAYROLL TAXES

## Best Practices

### 1. Use Rank Strategically
- Place more specific categories at lower ranks (higher priority)
- Generic catch-all categories should have higher rank numbers (lower priority)
- Example: "RETURNS & ALLOWANCES" (rank 1) before "SALES" (rank 2)

### 2. Include Keywords
- Use specific, unique keywords when possible
- Include common variations and synonyms
- Include vendor names for vendor-specific categories
- Use lowercase (system normalizes automatically)

### 3. Exclude Keywords
- Use to prevent ambiguous matches
- Example: Exclude "tax" from SALARIES to avoid matching payroll tax transactions
- Example: Exclude "refund" from SALES to ensure refunds go to RETURNS

### 4. Extensibility
- To add a new category:
  1. Choose an appropriate account code
  2. Assign a rank based on priority needs
  3. List relevant include_keywords
  4. Add exclude_keywords if needed to prevent false matches
  
- To modify existing category:
  1. Adjust rank to change priority
  2. Add/remove keywords as needed
  3. Test with sample transactions

## Code Implementation

The categorization logic is implemented in `account_code_mapper.py`:

```python
def _match_by_keywords(self, vendor: Optional[str], description: Optional[str]) -> Optional[tuple]:
    # 1. Normalize text
    # 2. Sort categories by rank
    # 3. Check exclude_keywords (if match, skip)
    # 4. Check include_keywords (if match, return and stop)
    # 5. Return None if no match
```

## Testing

Run the test suite to verify categorization logic:

```bash
python3 test_new_categorization.py
```

The test suite includes:
- Priority/rank ordering
- Exclude keyword logic
- Include keyword matching
- Edge cases and no-match scenarios

## Migration from Old Schema

**Old Schema:**
```json
{
  "602": {
    "name": "RETURNS & ALLOWANCES",
    "priority": 1,
    "keywords": ["refund", "return"]
  }
}
```

**New Schema:**
```json
{
  "602": {
    "name": "RETURNS & ALLOWANCES",
    "rank": 1,
    "include_keywords": ["refund", "return"],
    "exclude_keywords": []
  }
}
```

**Changes:**
- `priority` → `rank` (semantic clarity)
- `keywords` → `include_keywords` (explicit purpose)
- Added `exclude_keywords` (new feature)

## Summary

The new keyword schema provides:
✅ Clear priority-based matching (rank)
✅ Explicit inclusion logic (include_keywords)
✅ Powerful exclusion logic (exclude_keywords)
✅ Easy to extend and maintain
✅ Normalized text matching
✅ Clean, predictable behavior
