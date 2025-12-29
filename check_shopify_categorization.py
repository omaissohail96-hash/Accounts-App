#!/usr/bin/env python3
"""
Quick script to show ALL Shopify transactions with their categorization
"""

from bank_statement_parser import BankStatementParser
from schedule_c_categorizer import ScheduleCCategorizer
from account_code_mapper import AccountCodeMapper

def main():
    # Parse bank statements
    parser = BankStatementParser()
    transactions = parser.parse_all_pdfs('.')
    
    # Initialize categorizers
    schedule_c = ScheduleCCategorizer()
    mapper = AccountCodeMapper()
    
    # Find all Shopify transactions
    shopify_transactions = [
        t for t in transactions 
        if 'shopify' in t.vendor.lower() or 'shopify' in t.description.lower()
    ]
    
    print(f"\n{'='*80}")
    print(f"FOUND {len(shopify_transactions)} SHOPIFY TRANSACTIONS")
    print(f"{'='*80}\n")
    
    if not shopify_transactions:
        print("❌ NO SHOPIFY TRANSACTIONS FOUND!")
        print("\nSearching for transactions with 'shop' in them:")
        shop_transactions = [
            t for t in transactions 
            if 'shop' in t.vendor.lower() or 'shop' in t.description.lower()
        ]
        print(f"Found {len(shop_transactions)} transactions with 'shop'")
        for i, txn in enumerate(shop_transactions[:5], 1):
            print(f"\n{i}. Vendor: {txn.vendor}")
            print(f"   Description: {txn.description[:100]}")
        return
    
    for i, txn in enumerate(shopify_transactions, 1):
        # Categorize the transaction
        category = schedule_c.categorize_transaction(txn)
        
        # Get account code
        account_code = mapper.get_account_code(
            vendor_name=txn.vendor,
            description=txn.description,
            amount=txn.amount,
            category=category['category']
        )
        
        # Determine type based on account code
        if account_code.startswith('6'):
            txn_type = "Income"
        else:
            txn_type = "Expense"
        
        print(f"\n{'-'*80}")
        print(f"Transaction #{i}")
        print(f"{'-'*80}")
        print(f"Date: {txn.transaction_date}")
        print(f"Vendor: {txn.vendor}")
        print(f"Description: {txn.description[:100]}...")
        print(f"Amount: ${txn.amount:,.2f}")
        print(f"Direction: {txn.transaction_type}")
        print(f"\nSchedule C Category: {category['category']}")
        print(f"Tax Code: {category['tax_code']}")
        print(f"\nAccount Code: {account_code}")
        print(f"Type: {txn_type}")
        print(f"Is Excluded: {category['is_excluded']}")
        print(f"Is Owner Draw: {category['is_owner_draw']}")
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total Shopify Transactions: {len(shopify_transactions)}")
    
    income_count = sum(1 for t in shopify_transactions if t.amount > 0)
    expense_count = sum(1 for t in shopify_transactions if t.amount < 0)
    
    print(f"Positive amounts (deposits): {income_count}")
    print(f"Negative amounts (withdrawals): {expense_count}")
    print(f"\n✅ ALL of these should appear in your P&L report!")
    print(f"   Deposits → 601 SALES (Income)")
    print(f"   Withdrawals with 'fee' → 860 BANK & EBAY CHARGES (Expense)")
    print(f"   Withdrawals without 'fee' → 601 SALES (Income)")

if __name__ == "__main__":
    main()
