"""
Test the income statement-style report generation
"""
from bank_statement_parser import Transaction
from schedule_c_categorizer import ScheduleCCategorizer

def test_income_statement_report():
    """Test income statement report generation"""
    
    categorizer = ScheduleCCategorizer()
    
    # Create realistic sample transactions
    transactions = [
        # Income
        Transaction(date="2024-10-01", transaction_type="deposit", vendor="Shopify", 
                   amount=15000.00, description="Shopify payout for sales", raw_line=""),
        Transaction(date="2024-10-05", transaction_type="deposit", vendor="eBay", 
                   amount=8968.45, description="eBay payout", raw_line=""),
        Transaction(date="2024-10-10", transaction_type="deposit", vendor="Stripe", 
                   amount=15716.16, description="Stripe payment", raw_line=""),
        
        # COGS
        Transaction(date="2024-10-03", transaction_type="withdrawal", vendor="Wholesale Supplier", 
                   amount=-1369.02, description="Inventory purchase", raw_line=""),
        Transaction(date="2024-10-07", transaction_type="withdrawal", vendor="UPS", 
                   amount=-8952.01, description="Shipping supplies and freight", raw_line=""),
        
        # Expenses
        Transaction(date="2024-10-02", transaction_type="withdrawal", vendor="Payroll", 
                   amount=-3000.00, description="Salaries - officers", raw_line=""),
        Transaction(date="2024-10-08", transaction_type="withdrawal", vendor="Upwork", 
                   amount=-1450.00, description="Temporary help", raw_line=""),
        Transaction(date="2024-10-09", transaction_type="withdrawal", vendor="Fiverr", 
                   amount=-1733.84, description="Offshore labor", raw_line=""),
        Transaction(date="2024-10-04", transaction_type="withdrawal", vendor="Payroll Tax", 
                   amount=-229.50, description="Payroll taxes", raw_line=""),
        Transaction(date="2024-10-11", transaction_type="withdrawal", vendor="CPA Firm", 
                   amount=-375.00, description="Accounting services", raw_line=""),
        Transaction(date="2024-10-12", transaction_type="withdrawal", vendor="Google Ads", 
                   amount=-1180.22, description="Advertisement", raw_line=""),
        Transaction(date="2024-10-13", transaction_type="withdrawal", vendor="eBay", 
                   amount=-32.50, description="Bank & eBay charges", raw_line=""),
        Transaction(date="2024-10-14", transaction_type="withdrawal", vendor="Adobe", 
                   amount=-387.78, description="Dues and subscription", raw_line=""),
        Transaction(date="2024-10-15", transaction_type="withdrawal", vendor="Landlord", 
                   amount=-2664.00, description="Rent", raw_line=""),
        Transaction(date="2024-10-16", transaction_type="withdrawal", vendor="Office Depot", 
                   amount=-327.11, description="Store supplies", raw_line=""),
        Transaction(date="2024-10-17", transaction_type="withdrawal", vendor="Verizon", 
                   amount=-286.05, description="Telephone", raw_line=""),
        Transaction(date="2024-10-18", transaction_type="withdrawal", vendor="United Airlines", 
                   amount=-464.98, description="Travel", raw_line=""),
        Transaction(date="2024-10-19", transaction_type="withdrawal", vendor="Electric Co", 
                   amount=-141.80, description="Utilities - electricity", raw_line=""),
    ]
    
    # Categorize transactions
    categorized = categorizer.categorize_transactions(transactions)
    
    # Generate income statement report
    print("=" * 80)
    print("INCOME STATEMENT REPORT (Income Statement Style)")
    print("=" * 80)
    print()
    
    report = categorizer.generate_income_statement_report(categorized, period="Oct 25")
    print(report)
    
    # Also generate the dictionary format for potential CSV/Excel export
    print("\n" + "=" * 80)
    print("INCOME STATEMENT DICTIONARY (for CSV/Excel export)")
    print("=" * 80)
    print()
    
    income_dict = categorizer.generate_income_statement_dict(categorized)
    
    print(f"Total Income: ${income_dict['totals']['income_total']:,.2f}")
    print(f"Total COGS: ${income_dict['totals']['cogs_total']:,.2f}")
    print(f"Gross Profit: ${income_dict['totals']['gross_profit']:,.2f}")
    print(f"Total Expenses: ${income_dict['totals']['total_expenses']:,.2f}")
    print(f"Net Income: ${income_dict['totals']['net_income']:,.2f}")
    
    print("\nIncome Items:")
    for item in income_dict['income']:
        print(f"  {item['line_number']} · {item['category']}: ${item['amount']:,.2f} ({item['percent_of_income']}%)")
    
    print("\nCOGS Items:")
    for item in income_dict['cogs']:
        print(f"  {item['line_number']} · {item['category']}: ${item['amount']:,.2f} ({item['percent_of_income']}%)")
    
    print("\nExpense Items:")
    for item in income_dict['expenses']:
        print(f"  {item['line_number']} · {item['category']}: ${item['amount']:,.2f} ({item['percent_of_income']}%)")

if __name__ == "__main__":
    test_income_statement_report()

