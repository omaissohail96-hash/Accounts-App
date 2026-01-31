# 🎨 Enhanced Filtering System - Visual UI Guide

## 📍 UI Layout Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  ⚙️ Automatic Transaction Sorting & Categorization             │
│  Set up automatic filtering rules ONCE - they'll automatically │
│  sort transactions into the right categories for all future     │
│  uploads. Configure your sorting parameters below and the      │
│  system will remember them.                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  ⚡ Quick Setup: Common Scenarios                               │
│                                                                  │
│  [📝 Check Payments] [💳 Payment Processors] [🏢 Regular Expenses]
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  ➕ Setup New Sorting Rule                                      │
│                                                                  │
│  [📋 Step-by-Step] [⚙️ Advanced]                               │
│                                                                  │
│  TAB 1: Step-by-Step Method                                    │
│  ├─ STEP 1: What to Look For?                                  │
│  │  └─ [Keyword/Vendor Name input field]                       │
│  │     Placeholder: "e.g., Check, Stripe, PayPal, Rent"        │
│  │                                                              │
│  ├─ STEP 2: Where Should It Go?                                │
│  │  └─ [Account Code selector]                                 │
│  │     Options: 801 · SALARIES, 802 · WAGES, etc.              │
│  │                                                              │
│  └─ STEP 3: Any Additional Conditions?                         │
│     ├─ ☐ Filter by Transaction Amount?                         │
│     │  └─ If checked: [Min Amount field] [Max Amount field]     │
│     └─ ☐ Use Additional Keywords?                              │
│        └─ If checked: [Additional keywords] [Exclusion keywords]│
│                                                                  │
│  TAB 2: Advanced Configuration                                  │
│  ├─ [Primary Keyword field]                                    │
│  ├─ [Target Account Code selector]                             │
│  ├─ [Min Amount field] [Max Amount field]                       │
│  ├─ [Priority field] (1-999)                                   │
│  ├─ [Additional Keywords field]                                │
│  └─ [Exclusion Keywords field]                                 │
│                                                                  │
│  [✅ Create Sorting Rule] button                                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  📋 Active Rules                                                │
│  Total rules: 5 (sorted by priority)                            │
│                                                                  │
│  ▼ 🔍 Check [≥ $500 & ≤ $999] [Priority: 2]                   │
│    → 802 · WAGES                                               │
│    Filters:                                                      │
│    - Min Amount: $500                                           │
│    - Max Amount: $999                                           │
│    - Priority: 2                                                │
│    [🗑️ Delete]                                                 │
│                                                                  │
│  ▼ 🔍 Stripe [Priority: 1]                                     │
│    → 401 · SALES                                               │
│    Filters:                                                      │
│    - Exclusion Keywords: refund, payout                         │
│    [🗑️ Delete]                                                 │
│                                                                  │
│  (more rules...)                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Step-by-Step Tab - Detailed View

### STEP 1: What to Look For?

```
┌──────────────────────────────────────────────────────────────┐
│ **STEP 1: What to Look For?**                               │
│ _Enter the keyword or vendor name that identifies these_   │
│ _transactions_                                              │
│                                                              │
│ Keyword/Vendor Name *                                       │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ e.g., Check, Stripe, PayPal, Rent, Utilities, etc.    │  │
│ └────────────────────────────────────────────────────────┘  │
│ Primary keyword to match. This is REQUIRED.                 │
└──────────────────────────────────────────────────────────────┘
```

### STEP 2: Where Should It Go?

```
┌──────────────────────────────────────────────────────────────┐
│ **STEP 2: Where Should It Go?**                             │
│ _Select the account category for these transactions_        │
│                                                              │
│ Account Code *                                              │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ 801 · SALARIES                              ▼           │  │
│ │ ├─ 802 · WAGES                                         │  │
│ │ ├─ 803 · EMPLOYEE BENEFITS                             │  │
│ │ ├─ 804 · CONTRACTS                                     │  │
│ │ ├─ 805 · PROFESSIONAL SERVICES                         │  │
│ │ └─ (... more options)                                  │  │
│ └────────────────────────────────────────────────────────┘  │
│ The category these transactions should be sorted into.       │
└──────────────────────────────────────────────────────────────┘
```

### STEP 3: Any Additional Conditions?

```
┌──────────────────────────────────────────────────────────────┐
│ **STEP 3: Any Additional Conditions? (Optional)**           │
│ _Fine-tune the rule to catch exactly what you want_         │
│                                                              │
│ ☐ Filter by Transaction Amount?  ☐ Use Additional Keywords?│
│ Check this to only match certain  Check this to require      │
│ transaction sizes                  multiple keywords         │
│                                                              │
│ [If Amount Filter Checked:]                                 │
│ _Set the transaction amount range:_                         │
│ Minimum Amount ($)      Maximum Amount ($)                  │
│ ┌──────────────┐        ┌──────────────┐                   │
│ │      0       │        │      0       │                   │
│ └──────────────┘        └──────────────┘                   │
│ Minimum transaction size.    Maximum transaction size.      │
│ Enter 0 for no minimum.      Enter 0 for no maximum.        │
│                                                              │
│ [If Keyword Filter Checked:]                                │
│ _Enter additional matching conditions:_                     │
│ Keywords ALL Must Be Present                                │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ e.g., payment, landlord (comma-separated)           │   │
│ └──────────────────────────────────────────────────────┘   │
│ All these keywords must appear (AND logic)                  │
│                                                              │
│ Skip If ANY Of These Present                                │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ e.g., refund, reversal (comma-separated)            │   │
│ └──────────────────────────────────────────────────────┘   │
│ Skip this rule if any keyword found                         │
└──────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Advanced Tab - Detailed View

```
┌──────────────────────────────────────────────────────────────┐
│ **Advanced Rule Configuration**                             │
│ _For more control, configure all parameters here_           │
│                                                              │
│ Primary Keyword *          Target Account Code *            │
│ ┌──────────────────┐      ┌─────────────────────────────┐   │
│ │ e.g., Check      │      │ 801 · SALARIES       ▼      │   │
│ └──────────────────┘      └─────────────────────────────┘   │
│                                                              │
│ **Filtering Parameters:**                                   │
│ Min Amount                 Max Amount                       │
│ ┌──────────────┐           ┌──────────────┐               │
│ │     0.0      │           │     0.0      │               │
│ └──────────────┘           └──────────────┘               │
│                                                              │
│ **Pattern Matching:**                                       │
│ Priority (1=highest, 999=lowest)                            │
│ ┌──────────────┐                                           │
│ │     999      │                                           │
│ └──────────────┘                                           │
│                                                              │
│ Additional Keywords (ALL must match, comma-separated)       │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ e.g., rent, payment                                  │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
│ Exclusion Keywords (skip if ANY match, comma-separated)    │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ e.g., refund, dispute                               │   │
│ └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## 📋 Active Rules Display - Examples

### Example 1: Simple Rule

```
▼ 🔍 Stripe
  → 401 · SALES
  
[🗑️ Delete]
```

### Example 2: Rule with Amount Filter

```
▼ 🔍 Check [≥ $500 & ≤ $999]
  → 802 · WAGES
  
  Filters:
  - Min Amount: $500.00
  - Max Amount: $999.00
  
[🗑️ Delete]
```

### Example 3: Rule with Priority

```
▼ 🔍 Check [≥ $1000] [Priority: 1]
  → 801 · SALARIES
  
  Filters:
  - Min Amount: $1000.00
  - Priority: 1
  
[🗑️ Delete]
```

### Example 4: Complex Rule

```
▼ 🔍 Stripe [Priority: 1]
  → 401 · SALES
  
  Filters:
  - Exclusion Keywords: refund, payout
  - Priority: 1
  
[🗑️ Delete]
```

### Example 5: Pattern Matching Rule

```
▼ 🔍 payment [≥ $800 & ≤ $1200]
  → 822 · RENT
  
  Filters:
  - Min Amount: $800.00
  - Max Amount: $1200.00
  - Additional Keywords: rent, landlord
  
[🗑️ Delete]
```

---

## 🎯 Interactive Flow Diagram

```
USER ACTION: Click on "Custom Rules" Tab
                    ↓
        Display Main Section:
   "Automatic Transaction Sorting & Categorization"
        ↓ ↓ ↓
        │ │ └─→ Quick Templates [Check] [PayPal] [Expenses]
        │ │
        │ └─→ Setup New Sorting Rule
        │     ├─ Choice: Step-by-Step OR Advanced
        │     │
        │     ├─ Step-by-Step Path:
        │     │  STEP 1 → STEP 2 → STEP 3 → [Create Rule]
        │     │
        │     └─ Advanced Path:
        │        [All fields visible] → [Create Rule]
        │
        └─→ Active Rules Display
            Shows all saved rules
            ├─ Sorted by priority
            ├─ Expandable view
            └─ Delete buttons

USER CLICKS: [✅ Create Sorting Rule]
                    ↓
    ✅ Success Message (if valid)
    └─ Rule saved to database
    └─ Transactions updated
    └─ Page reloads to show rule in Active Rules
    
OR
    
    ❌ Error Message (if invalid)
    └─ "Please enter a keyword to search for"
    └─ Focus on keyword field
```

---

## 🔄 User Interaction Scenarios

### Scenario 1: First-Time User (Step-by-Step)

```
1. User lands on Custom Rules tab
   ↓
2. User clicks "📋 Step-by-Step" tab
   ↓
3. STEP 1: User enters "Check" in keyword field
   ↓
4. STEP 2: User selects "801 · SALARIES" from dropdown
   ↓
5. STEP 3: User checks "Filter by Transaction Amount?"
   ↓
6. STEP 3: User enters Min: 1000, Max: 0
   ↓
7. User clicks "✅ Create Sorting Rule"
   ↓
8. ✅ Success: "Sorting rule created: 'Check' → 801 · SALARIES ($1000+)"
   ↓
9. Rule appears in "Active Rules" section
   ↓
10. Existing transactions update immediately
   ↓
11. All future uploads use this rule automatically
```

### Scenario 2: Power User (Advanced)

```
1. User lands on Custom Rules tab
   ↓
2. User clicks "⚙️ Advanced" tab
   ↓
3. User fills all fields:
   - Primary Keyword: Stripe
   - Target Account: 401 · SALES
   - Min: 100, Max: 0
   - Priority: 1
   - Additional: sales (optional)
   - Exclusion: refund, payout
   ↓
4. User clicks "✅ Create Sorting Rule"
   ↓
5. Rule created with all specifications
   ↓
6. Rule appears in Active Rules (sorted by priority)
```

### Scenario 3: Template User

```
1. User sees "⚡ Quick Setup: Common Scenarios"
   ↓
2. User clicks [💳 Payment Processors]
   ↓
3. "processor" template fills into memory
   ↓
4. Step-by-Step tab shows templated values
   ↓
5. User can adjust or accept defaults
   ↓
6. Click Create to apply template rule
```

### Scenario 4: Deleting a Rule

```
1. User finds unwanted rule in "Active Rules"
   ↓
2. User expands the rule (click on it)
   ↓
3. User clicks [🗑️ Delete] button
   ↓
4. ✅ Success: "Rule deleted and transactions updated!"
   ↓
5. Rule disappears from Active Rules
   ↓
6. Transactions revert to previous categorization
```

---

## 🎨 Color & Visual Indicators

### Icons Used
- 🔍 Keyword match indicator
- → Arrow showing target category
- ✅ Success indicator
- ❌ Error indicator
- ⚙️ Configuration/Advanced
- 📋 Step-by-step
- 🗑️ Delete action
- 💡 Tips and examples
- ⚡ Quick action
- 🔧 Settings
- 📝 Template options

### Visual Hierarchy
```
Main Title (Largest)
    ↓
Section Headers (Medium, with icons)
    ↓
Subsection Headers (Smaller, descriptive text)
    ↓
Input Fields (Standard size)
    ↓
Help Text (Small, italic, hint color)
```

---

## 📱 Responsive Design Notes

### Desktop (Full Width)
- Two columns for side-by-side field placement
- Expanded views
- All features visible
- Full rule details displayed

### Tablet (Medium Width)
- Adapts to available width
- Stacks when needed
- Expandable sections
- Touch-friendly buttons

### Mobile (Small Width)
- Single column layout
- Step-by-Step tab prioritized
- Minimal width fields
- Adequate button sizes
- Scrollable sections

---

## 🎯 Best Practices for User Experience

### Visual Organization
- ✅ Group related inputs together
- ✅ Use clear section headers
- ✅ Provide examples in placeholders
- ✅ Show help text on hover/focus
- ✅ Use appropriate field types

### Clarity
- ✅ Label every required field with *
- ✅ Provide examples for each input
- ✅ Show success/error messages
- ✅ Explain what each filter does
- ✅ Display rule logic visually

### Usability
- ✅ Step-by-step for beginners
- ✅ Advanced for power users
- ✅ Templates for quick start
- ✅ Delete with confirmation feeling
- ✅ Immediate visual feedback

---

## 🔔 Notifications & Feedback

### Success Messages
```
✅ Sorting rule created: 'Check' → 801 · SALARIES ($500-$999)
✅ Rule deleted and transactions updated!
```

### Error Messages
```
❌ Please enter a keyword to search for
```

### Info Messages
```
ℹ️ Total rules: 5 (sorted by priority)
ℹ️ No custom rules defined yet. Add rules above...
```

### Help Text Examples
```
"Required: Primary keyword to match in transactions"
"Only match transactions >= this amount (use 0 for no minimum)"
"Rules with lower priority numbers are checked first"
```

---

## 📊 Visual Statistics

```
┌─────────────────────────────┐
│ Total rules: 5              │
│ Processing order: by priority│
│ Last updated: just now      │
└─────────────────────────────┘
```

---

**The UI is designed to be:**
- 🎯 Clear and intuitive
- 📚 Self-documenting with help text
- 🎓 Progressive (beginner to advanced)
- ⚡ Fast to use
- 🎨 Visually organized
