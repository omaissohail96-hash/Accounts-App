"""
Export income statement to CSV format for Excel/spreadsheet import
"""
import csv
import io
from typing import List, Tuple
from schedule_c_categorizer import ScheduleCCategorizer
from bank_statement_parser import Transaction


def export_income_statement_to_csv(categorizer: ScheduleCCategorizer, 
                                   categorized_transactions: List[Tuple[Transaction, any]], 
                                   period: str = "") -> str:
    """
    Export income statement to CSV format.
    
    Args:
        categorizer: ScheduleCCategorizer instance
        categorized_transactions: List of (Transaction, ScheduleCCategory) tuples
        period: Period label (e.g., "Oct 25")
    
    Returns:
        CSV string
    """
    income_dict = categorizer.generate_income_statement_dict(categorized_transactions)
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(["ORDINARY INCOME/EXPENSE", period if period else ""])
    writer.writerow([])
    writer.writerow(["Category", "Amount", "% of Income"])
    writer.writerow([])
    
    # Income section
    writer.writerow(["INCOME:"])
    for item in income_dict['income']:
        category_name = f"{item['line_number']} · {item['category'].upper()}"
        writer.writerow([category_name, item['amount'], f"{item['percent_of_income']}%"])
    
    writer.writerow(["Total Income", income_dict['totals']['income_total'], "100.0%"])
    writer.writerow([])
    
    # COGS section
    if income_dict['totals']['cogs_total'] > 0:
        writer.writerow(["COST OF GOODS SOLD:"])
        for item in sorted(income_dict['cogs'], key=lambda x: x['amount'], reverse=True):
            category_name = f"{item['line_number']} · {item['category'].upper()}"
            writer.writerow([category_name, item['amount'], f"{item['percent_of_income']}%"])
        
        cogs_pct = (income_dict['totals']['cogs_total'] / income_dict['totals']['income_total'] * 100) if income_dict['totals']['income_total'] > 0 else 0
        writer.writerow(["Total COGS", income_dict['totals']['cogs_total'], f"{cogs_pct:.1f}%"])
        writer.writerow([])
        
        # Gross Profit
        gross_profit_pct = (income_dict['totals']['gross_profit'] / income_dict['totals']['income_total'] * 100) if income_dict['totals']['income_total'] > 0 else 0
        writer.writerow(["Gross Profit", income_dict['totals']['gross_profit'], f"{gross_profit_pct:.1f}%"])
        writer.writerow([])
    
    # Expenses section
    writer.writerow(["EXPENSE:"])
    for item in income_dict['expenses']:
        category_name = f"{item['line_number']} · {item['category'].upper()}"
        writer.writerow([category_name, item['amount'], f"{item['percent_of_income']}%"])
    
    total_exp_pct = (income_dict['totals']['total_expenses'] / income_dict['totals']['income_total'] * 100) if income_dict['totals']['income_total'] > 0 else 0
    writer.writerow(["Total Expense", income_dict['totals']['total_expenses'], f"{total_exp_pct:.1f}%"])
    writer.writerow([])
    
    # Net Income
    net_income_pct = (income_dict['totals']['net_income'] / income_dict['totals']['income_total'] * 100) if income_dict['totals']['income_total'] > 0 else 0
    writer.writerow(["Net Ordinary Income", income_dict['totals']['net_income'], f"{net_income_pct:.1f}%"])
    writer.writerow([])
    writer.writerow(["Net Income", income_dict['totals']['net_income'], f"{net_income_pct:.1f}%"])
    
    return output.getvalue()


def export_income_statement_to_file(categorizer: ScheduleCCategorizer,
                                    categorized_transactions: List[Tuple[Transaction, any]],
                                    output_path: str,
                                    period: str = ""):
    """
    Export income statement to CSV file.
    
    Args:
        categorizer: ScheduleCCategorizer instance
        categorized_transactions: List of (Transaction, ScheduleCCategory) tuples
        output_path: Path to output CSV file
        period: Period label
    """
    csv_content = export_income_statement_to_csv(categorizer, categorized_transactions, period)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        f.write(csv_content)


if __name__ == "__main__":
    # Example usage
    from schedule_c_categorizer import ScheduleCCategorizer
    from bank_statement_parser import Transaction
    
    # Sample transactions
    transactions = [
        Transaction(date="2024-10-01", transaction_type="deposit", vendor="Shopify", 
                   amount=15000.00, description="Shopify payout", raw_line=""),
        Transaction(date="2024-10-12", transaction_type="withdrawal", vendor="Google Ads", 
                   amount=-200.00, description="Google Ads spend", raw_line=""),
    ]
    
    categorizer = ScheduleCCategorizer()
    categorized = categorizer.categorize_transactions(transactions)
    
    # Export to CSV string
    csv_output = export_income_statement_to_csv(categorizer, categorized, period="Oct 25")
    print(csv_output)
    
    # Or export to file
    # export_income_statement_to_file(categorizer, categorized, "income_statement.csv", period="Oct 25")



