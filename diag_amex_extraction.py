
from bank_statement_parser import AmexParser
import logging

logging.basicConfig(level=logging.INFO)

def diag_amex():
    parser = AmexParser()
    sample_text = """
    AMERICAN EXPRESS
    Statement Date 01/15/2025
    
    PAYMENTS AND CREDITS
    01/02 PAYMENT RECEIVED - THANK YOU  $1,234.56
    
    PURCHASES
    01/05 AMAZON.COM*123 206-555-0199 WA $45.67
    01/06 STARBUCKS 800-STARBUCKS      $6.78
    
    TOTALS
    New Balance $1,287.01
    """
    
    print("Parsing sample text...")
    statement = parser.parse(sample_text)
    
    print(f"Extracted {len(statement.transactions)} transactions.")
    for tx in statement.transactions:
        print(f"  {tx.date.strftime('%m/%d')} {tx.description}: {tx.amount}")

if __name__ == "__main__":
    diag_amex()
