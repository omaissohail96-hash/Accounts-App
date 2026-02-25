"""
Example: Integrating Multi-Bank Parser with PDF Processing
Demonstrates how to use the refactored parser with PDF statements
"""

import pdfplumber
from pathlib import Path
from bank_statement_parser import (
    parse_bank_statement,
    detect_bank,
    BankName,
    Transaction,
    ParsedStatement
)


def parse_pdf_statement(pdf_path: str) -> ParsedStatement:
    """
    Parse a bank statement PDF file.
    
    Args:
        pdf_path: Path to PDF statement file
    
    Returns:
        ParsedStatement with transactions and metadata
    """
    print(f"Processing: {pdf_path}")
    
    # Extract text from PDF
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                full_text += text + "\n"
            print(f"  Extracted page {page_num}/{len(pdf.pages)}")
    
    # Detect bank
    bank = detect_bank(full_text)
    print(f"  Detected bank: {bank.value}")
    
    # Parse statement
    statement = parse_bank_statement(full_text, bank_name=bank)
    
    # Print summary
    print(f"  ✓ Parsed {len(statement.transactions)} transactions")
    if statement.beginning_balance:
        print(f"  Beginning Balance: ${statement.beginning_balance:,.2f}")
    if statement.ending_balance:
        print(f"  Ending Balance: ${statement.ending_balance:,.2f}")
    
    # Check for errors
    if statement.errors:
        print(f"  ⚠ Errors: {', '.join(statement.errors)}")
    
    return statement


def process_directory(directory_path: str):
    """
    Process all PDF statements in a directory.
    
    Args:
        directory_path: Path to directory containing PDF statements
    """
    directory = Path(directory_path)
    pdf_files = list(directory.glob("*.pdf"))
    
    print(f"\nFound {len(pdf_files)} PDF files in {directory_path}\n")
    
    all_statements = []
    
    for pdf_file in pdf_files:
        try:
            statement = parse_pdf_statement(str(pdf_file))
            all_statements.append(statement)
            print()
        except Exception as e:
            print(f"  ✗ Error processing {pdf_file.name}: {e}\n")
    
    # Generate summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    
    for statement in all_statements:
        total_credits = sum(t.amount for t in statement.transactions if t.amount > 0)
        total_debits = sum(abs(t.amount) for t in statement.transactions if t.amount < 0)
        
        print(f"\n{statement.bank_name.upper()}:")
        print(f"  Transactions: {len(statement.transactions)}")
        print(f"  Total Credits: ${total_credits:,.2f}")
        print(f"  Total Debits: ${total_debits:,.2f}")
        print(f"  Net: ${total_credits - total_debits:,.2f}")


def export_to_csv(statement: ParsedStatement, output_path: str):
    """
    Export transactions to CSV file.
    
    Args:
        statement: ParsedStatement to export
        output_path: Path to output CSV file
    """
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['date', 'description', 'amount', 'type', 'category', 'balance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for trans in statement.transactions:
            writer.writerow({
                'date': trans.date.strftime('%Y-%m-%d'),
                'description': trans.description,
                'amount': f"{trans.amount:.2f}",
                'type': trans.type.value,
                'category': trans.category.value if trans.category else '',
                'balance': f"{trans.running_balance:.2f}" if trans.running_balance else ''
            })
    
    print(f"Exported {len(statement.transactions)} transactions to {output_path}")


def convert_to_dataframe(statement: ParsedStatement):
    """
    Convert parsed statement to pandas DataFrame.
    
    Args:
        statement: ParsedStatement to convert
    
    Returns:
        pandas DataFrame with transaction data
    """
    import pandas as pd
    
    data = []
    for trans in statement.transactions:
        data.append({
            'date': trans.date,
            'description': trans.description,
            'amount': trans.amount,
            'type': trans.type.value,
            'category': trans.category.value if trans.category else None,
            'balance': trans.running_balance,
            'bank': trans.bank_name,
            'needs_review': trans.needs_review
        })
    
    df = pd.DataFrame(data)
    
    # Sort by date
    df = df.sort_values('date')
    
    return df


def analyze_transactions(statement: ParsedStatement):
    """
    Perform basic analysis on parsed transactions.
    
    Args:
        statement: ParsedStatement to analyze
    """
    print(f"\nAnalysis for {statement.bank_name.upper()}")
    print("="*60)
    
    # Categorize transactions
    by_category = {}
    for trans in statement.transactions:
        cat = trans.category.value if trans.category else "uncategorized"
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(trans)
    
    print("\nTransactions by Category:")
    for category, transactions in sorted(by_category.items()):
        total = sum(t.amount for t in transactions)
        print(f"  {category:20s}: {len(transactions):3d} transactions | ${total:>12,.2f}")
    
    # Find largest transactions
    sorted_trans = sorted(statement.transactions, key=lambda t: abs(t.amount), reverse=True)
    
    print("\nTop 5 Largest Transactions:")
    for i, trans in enumerate(sorted_trans[:5], 1):
        print(f"  {i}. {trans.date.strftime('%Y-%m-%d')} | {trans.description[:40]:40s} | ${trans.amount:>10,.2f}")
    
    # Transactions needing review
    needs_review = [t for t in statement.transactions if t.needs_review]
    if needs_review:
        print(f"\n⚠ {len(needs_review)} transactions need manual review")


def reconcile_balance(statement: ParsedStatement):
    """
    Verify that running balances are correct.
    
    Args:
        statement: ParsedStatement to reconcile
    
    Returns:
        bool: True if balances match, False otherwise
    """
    if not statement.beginning_balance or not statement.ending_balance:
        print("Cannot reconcile: missing beginning or ending balance")
        return False
    
    calculated_balance = statement.beginning_balance
    for trans in sorted(statement.transactions, key=lambda t: t.date):
        calculated_balance += trans.amount
    
    expected_balance = statement.ending_balance
    difference = abs(calculated_balance - expected_balance)
    
    print(f"\nBalance Reconciliation:")
    print(f"  Beginning Balance: ${statement.beginning_balance:,.2f}")
    print(f"  Calculated Ending: ${calculated_balance:,.2f}")
    print(f"  Expected Ending:   ${expected_balance:,.2f}")
    print(f"  Difference:        ${difference:,.2f}")
    
    if difference < 0.01:  # Allow for rounding
        print("  ✓ Balances match!")
        return True
    else:
        print("  ✗ Balances do not match")
        return False


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python example_integration.py <pdf_file_or_directory>")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if Path(path).is_dir():
        # Process directory
        process_directory(path)
    elif Path(path).is_file():
        # Process single file
        statement = parse_pdf_statement(path)
        
        # Analyze transactions
        analyze_transactions(statement)
        
        # Reconcile balance
        reconcile_balance(statement)
        
        # Export to CSV
        csv_path = Path(path).stem + "_transactions.csv"
        export_to_csv(statement, csv_path)
        
        # Convert to DataFrame
        try:
            df = convert_to_dataframe(statement)
            print(f"\nDataFrame created with {len(df)} rows")
            print(df.head())
        except ImportError:
            print("\nInstall pandas to use DataFrame conversion: pip install pandas")
    else:
        print(f"Error: {path} is not a valid file or directory")
