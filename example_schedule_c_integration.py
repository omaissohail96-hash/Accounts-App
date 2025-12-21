"""
Example: Integration of Schedule C Categorizer with Bank Statement Parser

This example shows how to use the Schedule C categorizer with the existing
bank statement parser to categorize transactions for tax preparation.
"""
from bank_statement_parser import BankStatementParser, Transaction
from schedule_c_categorizer import ScheduleCCategorizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_bank_statement_for_schedule_c(file_path: str):
    """
    Parse a bank statement and categorize transactions for Schedule C.
    
    Args:
        file_path: Path to bank statement file (PDF, CSV, etc.)
    
    Returns:
        Dictionary with categorized transactions and Schedule C summary
    """
    # Step 1: Parse bank statement
    parser = BankStatementParser()
    
    try:
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
        
        # Parse the statement (you may need to adapt this based on your parser implementation)
        # For now, this is a conceptual example
        logger.info(f"Parsing bank statement: {file_path}")
        # transactions = parser.parse(file_bytes)  # Adjust based on your parser API
        
        # For demonstration, create sample transactions
        # In real usage, replace this with actual parsed transactions
        transactions = [
            Transaction(
                date="2024-01-15",
                transaction_type="deposit",
                vendor="Shopify",
                amount=1500.00,
                description="Shopify payout for sales",
                raw_line="",
            ),
            Transaction(
                date="2024-01-12",
                transaction_type="withdrawal",
                vendor="Google Ads",
                amount=-200.00,
                description="Google Ads spend",
                raw_line="",
            ),
        ]
        
    except Exception as e:
        logger.error(f"Error parsing bank statement: {e}")
        return None
    
    # Step 2: Categorize transactions using Schedule C categorizer
    categorizer = ScheduleCCategorizer()
    logger.info(f"Categorizing {len(transactions)} transactions...")
    
    categorized_transactions = categorizer.categorize_transactions(transactions)
    
    # Step 3: Generate Schedule C summary
    summary = categorizer.generate_schedule_c_summary(categorized_transactions)
    
    # Step 4: Generate human-readable report
    report = categorizer.generate_schedule_c_report(categorized_transactions)
    
    return {
        "categorized_transactions": categorized_transactions,
        "summary": summary,
        "report": report,
        "transaction_count": len(transactions)
    }


def export_for_bookkeeping(categorized_transactions, output_format="csv"):
    """
    Export categorized transactions in a format suitable for bookkeeping software.
    
    Args:
        categorized_transactions: List of (Transaction, ScheduleCCategory) tuples
        output_format: "csv", "json", or "excel"
    
    Returns:
        Exported data (as string or file path)
    """
    import csv
    import json
    from datetime import datetime
    
    if output_format == "csv":
        # Generate CSV for import into QuickBooks, Xero, etc.
        output_lines = []
        output_lines.append([
            "Date", "Vendor", "Description", "Amount", 
            "Schedule C Part", "Line Number", "Category", "Tax Code"
        ])
        
        for transaction, category in categorized_transactions:
            # Skip excluded transactions and owner draws for bookkeeping export
            if category.is_excluded or category.is_owner_draw:
                continue
            
            output_lines.append([
                transaction.date,
                transaction.vendor or "",
                transaction.description or "",
                transaction.amount,
                category.part,
                category.line_number or "",
                category.category_name,
                category.tax_code
            ])
        
        # Return as CSV string
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(output_lines)
        return output.getvalue()
    
    elif output_format == "json":
        # Generate JSON for API integration
        data = []
        for transaction, category in categorized_transactions:
            if category.is_excluded or category.is_owner_draw:
                continue
            
            data.append({
                "date": transaction.date,
                "vendor": transaction.vendor,
                "description": transaction.description,
                "amount": transaction.amount,
                "schedule_c": {
                    "part": category.part,
                    "line_number": category.line_number,
                    "category": category.category_name,
                    "tax_code": category.tax_code
                }
            })
        
        return json.dumps(data, indent=2)
    
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


if __name__ == "__main__":
    # Example usage
    print("Schedule C Categorizer Integration Example")
    print("=" * 80)
    print()
    
    # Example: Process a bank statement
    # result = process_bank_statement_for_schedule_c("path/to/bank_statement.pdf")
    
    # Example: Export for bookkeeping
    from schedule_c_categorizer import ScheduleCCategorizer
    from bank_statement_parser import Transaction
    
    # Create sample transactions
    transactions = [
        Transaction(
            date="2024-01-15",
            transaction_type="deposit",
            vendor="Shopify",
            amount=1500.00,
            description="Shopify payout",
            raw_line="",
        ),
        Transaction(
            date="2024-01-12",
            transaction_type="withdrawal",
            vendor="Google Ads",
            amount=-200.00,
            description="Google Ads spend",
            raw_line="",
        ),
    ]
    
    categorizer = ScheduleCCategorizer()
    categorized = categorizer.categorize_transactions(transactions)
    
    # Export as CSV
    csv_output = export_for_bookkeeping(categorized, "csv")
    print("CSV Export (for bookkeeping software):")
    print(csv_output)
    print()
    
    # Export as JSON
    json_output = export_for_bookkeeping(categorized, "json")
    print("JSON Export (for API integration):")
    print(json_output)
    print()
    
    # Generate income statement-style report
    print("=" * 80)
    print("INCOME STATEMENT STYLE REPORT")
    print("=" * 80)
    income_report = categorizer.generate_income_statement_report(categorized, period="Sample Period")
    print(income_report)

