# 📚 Rule Configuration Examples

## 🎯 Ready-to-Use Rule Templates

All examples below can be directly entered into the Enhanced Filtering System. Copy the parameters and paste into the form.

---

## 1️⃣ CHECK PAYMENTS - By Size

**Scenario:** Different check amounts should go to different categories

### Rule 1.1: Large Checks (Salaries)
```
Keyword: Check
Min Amount: $1000
Max Amount: 0
Account: 801 · SALARIES
Priority: 1
```

### Rule 1.2: Medium Checks (Wages)
```
Keyword: Check
Min Amount: $500
Max Amount: $999
Account: 802 · WAGES  
Priority: 2
```

### Rule 1.3: Small Checks (Supplies)
```
Keyword: Check
Min Amount: 0
Max Amount: $499
Account: 806 · SUPPLIES
Priority: 3
```

### Rule 1.4: NSF/Bad Checks
```
Keyword: Check
Exclusion Keywords: NSF, reversal, cancelled
Account: 902 · BANK_FEES
Priority: 0 (run before other check rules)
```

---

## 2️⃣ PAYMENT PROCESSORS

**Scenario:** Handle multiple payment platforms with different rules

### Rule 2.1: Stripe Sales (Main)
```
Keyword: Stripe
Exclusion Keywords: refund, payout, fee, reversal
Account: 401 · SALES
Priority: 1
```

### Rule 2.2: Stripe Refunds
```
Keyword: Stripe
Additional Keywords: refund
Account: 401 · SALES (with negative amount)
Priority: 2
```

### Rule 2.3: Stripe Payouts
```
Keyword: Stripe
Additional Keywords: payout
Account: 111 · BANK_TRANSFER
Priority: 3
```

### Rule 2.4: PayPal Transactions
```
Keyword: PayPal
Exclusion Keywords: refund, reversal
Account: 401 · SALES
Priority: 1
```

### Rule 2.5: PayPal Refunds
```
Keyword: PayPal
Additional Keywords: refund
Account: 401 · SALES
Priority: 2
```

### Rule 2.6: Square Payments
```
Keyword: Square
Exclusion Keywords: test, refund
Account: 401 · SALES
Priority: 1
```

---

## 3️⃣ RENT & HOUSING

**Scenario:** Separate rent payments, utilities, and repairs

### Rule 3.1: Rent Payment
```
Keyword: rent
Additional Keywords: landlord, payment, monthly
Min Amount: $500
Max Amount: $5000
Account: 822 · RENT
Priority: 1
```

### Rule 3.2: Rent - Direct Name
```
Keyword: landlord
Account: 822 · RENT
Priority: 1
```

### Rule 3.3: Utilities - Electric
```
Keyword: electric
Exclusion Keywords: refund, credit
Min Amount: $20
Account: 821 · UTILITIES
Priority: 1
```

### Rule 3.4: Utilities - Water/Gas
```
Keyword: water, gas, utility
Exclusion Keywords: refund, credit
Account: 821 · UTILITIES
Priority: 1
```

### Rule 3.5: Building Repairs
```
Keyword: repair
Additional Keywords: building, maintenance
Min Amount: $100
Account: 823 · REPAIRS & MAINTENANCE
Priority: 1
```

---

## 4️⃣ PAYROLL & WAGES

**Scenario:** Multiple employees, contractors, wage types

### Rule 4.1: Contractor Payments
```
Keyword: contractor
Min Amount: $500
Account: 804 · CONTRACTS
Priority: 1
```

### Rule 4.2: Payroll - Large (Salary)
```
Keyword: payroll
Min Amount: $1000
Account: 801 · SALARIES
Priority: 1
```

### Rule 4.3: Payroll - Small (Part-time)
```
Keyword: payroll
Max Amount: $999
Account: 802 · WAGES
Priority: 2
```

### Rule 4.4: Benefits Payments
```
Keyword: benefit
Account: 803 · EMPLOYEE BENEFITS
Priority: 1
```

---

## 5️⃣ OFFICE & SUPPLIES

**Scenario:** Different office expenses by type and amount

### Rule 5.1: Office Supplies - Small
```
Keyword: office
Max Amount: $500
Account: 806 · SUPPLIES
Priority: 1
```

### Rule 5.2: Office Supplies - Bulk
```
Keyword: office
Min Amount: $500
Account: 807 · EQUIPMENT
Priority: 2
```

### Rule 5.3: Printer/Copier
```
Keyword: printer, copier, toner
Account: 806 · SUPPLIES
Priority: 1
```

### Rule 5.4: Stationery
```
Keyword: stationary
Account: 806 · SUPPLIES
Priority: 1
```

### Rule 5.5: Furniture
```
Keyword: furniture
Min Amount: $100
Account: 807 · EQUIPMENT
Priority: 1
```

---

## 6️⃣ UTILITIES & SERVICES

**Scenario:** Recurring utilities and service subscriptions

### Rule 6.1: Internet Service
```
Keyword: internet
Exclusion Keywords: refund, credit
Account: 821 · UTILITIES
Priority: 1
```

### Rule 6.2: Phone/Mobile
```
Keyword: verizon, at&t, mobile, phone
Exclusion Keywords: refund
Account: 821 · UTILITIES
Priority: 1
```

### Rule 6.3: Electric Company
```
Keyword: electric, electricity, power company
Account: 821 · UTILITIES
Priority: 1
```

### Rule 6.4: Water Company
```
Keyword: water, sewage, aqua
Account: 821 · UTILITIES
Priority: 1
```

### Rule 6.5: Gas Company
```
Keyword: gas, natural gas
Exclusion Keywords: gasoline, fuel
Account: 821 · UTILITIES
Priority: 1
```

---

## 7️⃣ TRANSPORTATION & FUEL

**Scenario:** Vehicle expenses, gas, maintenance

### Rule 7.1: Fuel/Gasoline
```
Keyword: gasoline, fuel, shell, exxon, bp
Min Amount: $20
Account: 814 · FUEL
Priority: 1
```

### Rule 7.2: Car Maintenance
```
Keyword: maintenance, service, oil change
Account: 815 · VEHICLE MAINTENANCE
Priority: 1
```

### Rule 7.3: Auto Insurance
```
Keyword: insurance
Additional Keywords: auto, vehicle, car
Account: 816 · AUTO INSURANCE
Priority: 1
```

### Rule 7.4: Parking
```
Keyword: parking, garage
Account: 814 · FUEL (or separate category)
Priority: 1
```

---

## 8️⃣ MEALS & ENTERTAINMENT

**Scenario:** Meals, entertainment, client entertainment

### Rule 8.1: Meals & Entertainment
```
Keyword: restaurant, cafe, coffee
Max Amount: $100
Account: 812 · MEALS & ENTERTAINMENT
Priority: 1
```

### Rule 8.2: Business Meals
```
Keyword: restaurant
Additional Keywords: business, client, meeting
Min Amount: $50
Max Amount: $500
Account: 812 · MEALS & ENTERTAINMENT
Priority: 1
```

### Rule 8.3: Entertainment - Large
```
Keyword: entertainment
Min Amount: $500
Account: 812 · MEALS & ENTERTAINMENT
Priority: 1
```

---

## 9️⃣ PROFESSIONAL SERVICES

**Scenario:** Accountants, lawyers, consultants

### Rule 9.1: Accounting/Tax Services
```
Keyword: accounting, cpa, tax
Account: 805 · PROFESSIONAL SERVICES
Priority: 1
```

### Rule 9.2: Legal Services
```
Keyword: legal, attorney, law firm
Account: 805 · PROFESSIONAL SERVICES
Priority: 1
```

### Rule 9.3: Consulting
```
Keyword: consulting, consultant
Min Amount: $200
Account: 805 · PROFESSIONAL SERVICES
Priority: 1
```

### Rule 9.4: Software/Subscriptions
```
Keyword: software, subscription, license
Account: 808 · SOFTWARE
Priority: 1
```

---

## 🔟 ADVERTISING & MARKETING

**Scenario:** Ad spend, marketing campaigns

### Rule 10.1: Google Ads
```
Keyword: google ads, adwords
Account: 813 · ADVERTISING & MARKETING
Priority: 1
```

### Rule 10.2: Facebook Ads
```
Keyword: facebook ads, meta ads
Account: 813 · ADVERTISING & MARKETING
Priority: 1
```

### Rule 10.3: Email Marketing
```
Keyword: mailchimp, constant contact, email
Account: 813 · ADVERTISING & MARKETING
Priority: 1
```

### Rule 10.4: Marketing - General
```
Keyword: marketing
Min Amount: $100
Account: 813 · ADVERTISING & MARKETING
Priority: 2
```

---

## 1️⃣1️⃣ BANKING & FINANCIAL

**Scenario:** Bank fees, transfers, interest

### Rule 11.1: Bank Fees
```
Keyword: fee
Max Amount: $100
Account: 902 · BANK_FEES
Priority: 1
```

### Rule 11.2: ATM Withdrawal
```
Keyword: atm
Min Amount: $100
Max Amount: $500
Account: 111 · CASH ACCOUNT
Priority: 1
```

### Rule 11.3: Wire Transfer
```
Keyword: wire transfer
Min Amount: $100
Account: 111 · BANK_TRANSFER
Priority: 1
```

### Rule 11.4: Interest Income
```
Keyword: interest
Account: 401 · INCOME (deposit category)
Priority: 1
```

---

## 1️⃣2️⃣ TAXES & LICENSES

**Scenario:** Tax payments, business licenses

### Rule 12.1: Income Tax Payment
```
Keyword: tax, irs
Min Amount: $500
Account: 903 · TAXES_PAID
Priority: 1
```

### Rule 12.2: Sales Tax
```
Keyword: sales tax
Account: 904 · SALES_TAX_LIABILITY
Priority: 1
```

### Rule 12.3: Business License
```
Keyword: license
Additional Keywords: business, permit
Min Amount: $50
Max Amount: $1000
Account: 809 · LICENSES & PERMITS
Priority: 1
```

---

## 🔗 INTEGRATION EXAMPLES

### Multi-Vendor Scenario
```
Setup 3 vendors with same product category

Rule 1:
- Keyword: Amazon
- Account: 806 · SUPPLIES
- Priority: 1

Rule 2:
- Keyword: eBay
- Account: 806 · SUPPLIES
- Priority: 1

Rule 3:
- Keyword: Staples
- Account: 806 · SUPPLIES
- Priority: 1
```

### Amount-Based Routing
```
Small expenses automatic, large expenses manual

Rule 1:
- Keyword: purchase
- Max Amount: $500
- Account: 806 · SUPPLIES (auto)
- Priority: 1

Rule 2:
- Keyword: purchase
- Min Amount: $500
- Account: 999 · OTHER (manual review)
- Priority: 2
```

### Complex Logic Example
```
Priority-based Stripe categorization

Rule 1 (Run First):
- Keyword: Stripe
- Additional: test
- Exclude rule (manual category)
- Priority: 1

Rule 2:
- Keyword: Stripe
- Additional: refund
- Account: REFUNDS
- Priority: 2

Rule 3:
- Keyword: Stripe
- Exclude: refund, payout
- Account: SALES
- Priority: 3

Rule 4:
- Keyword: Stripe
- Account: OTHER (catch-all)
- Priority: 999
```

---

## ⚡ Quick Copy-Paste Snippets

### Expense Type: Checks
```
Keyword: Check
(adjust min/max as needed)
```

### Expense Type: Credit Card Payments
```
Keyword: credit card, visa, mastercard
```

### Expense Type: Loans
```
Keyword: loan, payment
```

### Expense Type: Insurance
```
Keyword: insurance
```

### Expense Type: Subscriptions
```
Keyword: subscription, monthly, annual
```

### Expense Type: Software
```
Keyword: software, app, license
```

---

## 🎯 Priority Recommendation Matrix

| Scenario | Priority 1 | Priority 2 | Priority 3 | Priority 999 |
|----------|-----------|-----------|-----------|-----------|
| Checks | Specific (NSF) | Large (1k+) | Medium (500-1k) | Small (<500) |
| Stripe | Tests/Bad | Refunds | Payouts | Sales |
| Rent | Exclusions | Main rule | - | - |
| Billing | Bad charges | Refunds | - | Main charges |

---

## 🔄 Testing Workflow

For each new rule:

1. **Create rule with just keyword**
   - Does it catch something?
   
2. **Add amount filter**
   - Check if range is correct
   
3. **Add exclusions**
   - Do false positives disappear?
   
4. **Add additional keywords**
   - More accurate matching?
   
5. **Set priority**
   - Runs in correct order?
   
6. **Monitor next upload**
   - Still working correctly?

---

## 📊 Account Code Reference

```
400-499: Income
801: Salaries
802: Wages  
803: Employee Benefits
804: Contracts
805: Professional Services
806: Supplies
807: Equipment
808: Software
809: Licenses & Permits
811: Rent/Lease
812: Meals & Entertainment
813: Advertising & Marketing
814: Fuel
815: Vehicle Maintenance
816: Auto Insurance
821: Utilities
822: Rent
823: Repairs & Maintenance
901: Depreciation
902: Bank Fees
903: Taxes Paid
904: Sales Tax Liability
999: Other Expenses
```

---

**Ready to configure?** Start with one template that matches your most common transaction type, then expand!
