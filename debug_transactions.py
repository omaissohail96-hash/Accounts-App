"""
Debug script to see actual transaction vendors and descriptions
"""
import pandas as pd
from bank_statement_parser import Transaction

# Load your transactions CSV or data
try:
    df = pd.read_csv("sample_transactions.csv")
    
    print("=== SAMPLE TRANSACTION DATA ===")
    print("\nFirst 20 unique vendors:")
    if 'vendor' in df.columns:
        print(df['vendor'].unique()[:20])
    elif 'Vendor' in df.columns:
        print(df['Vendor'].unique()[:20])
    
    print("\n\nFirst 20 descriptions:")
    if 'description' in df.columns:
        print(df['description'].head(20).tolist())
    elif 'Description' in df.columns:
        print(df['Description'].head(20).tolist())
    
    print("\n\nTransaction types:")
    if 'transaction_type' in df.columns:
        print(df['transaction_type'].value_counts())
    elif 'Type' in df.columns:
        print(df['Type'].value_counts())
        
except Exception as e:
    print(f"Error: {e}")
    print("\nPlease provide the path to your transactions file")
