"""
Report Generator Module
Creates professional P&L reports with deposits and withdrawals summaries
"""
import pandas as pd
from typing import List, Dict
from bank_statement_parser import Transaction
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates professional financial reports"""
    
    def __init__(self):
        pass
    
    def generate_deposits_summary(self, transactions: List[Transaction]) -> pd.DataFrame:
        """Generate deposits summary grouped by source/vendor"""
        deposits = [t for t in transactions if t.transaction_type == 'deposit' and t.amount > 0]
        
        if not deposits:
            return pd.DataFrame(columns=['Source/Vendor', 'Transaction Count', 'Subtotal ($)', 'Transactions'])
        
        # Group by vendor
        vendor_groups = {}
        for trans in deposits:
            vendor = trans.vendor or "Unknown Source"
            if vendor not in vendor_groups:
                vendor_groups[vendor] = []
            vendor_groups[vendor].append(trans)
        
        # Create summary rows
        summary_rows = []
        for vendor, trans_list in sorted(vendor_groups.items()):
            subtotal = sum(t.amount for t in trans_list)
            count = len(trans_list)
            # Create detailed transaction list
            trans_details = []
            for t in sorted(trans_list, key=lambda x: x.date or ''):
                date_str = t.date or 'N/A'
                desc = t.description[:50] if t.description else 'N/A'
                trans_details.append(f"{date_str} | ${t.amount:,.2f} | {desc}")
            
            summary_rows.append({
                'Source/Vendor': vendor,
                'Transaction Count': count,
                'Subtotal ($)': round(subtotal, 2),
                'Transactions': '; '.join(trans_details)
            })
        
        df = pd.DataFrame(summary_rows)
        # Add total row
        total_row = pd.DataFrame([{
            'Source/Vendor': 'TOTAL DEPOSITS',
            'Transaction Count': df['Transaction Count'].sum(),
            'Subtotal ($)': df['Subtotal ($)'].sum(),
            'Transactions': ''
        }])
        df = pd.concat([df, total_row], ignore_index=True)
        
        return df
    
    def generate_withdrawals_summary(self, transactions: List[Transaction]) -> pd.DataFrame:
        """Generate withdrawals summary grouped by category and vendor"""
        withdrawals = [t for t in transactions if t.transaction_type == 'withdrawal' and t.amount < 0]
        
        if not withdrawals:
            return pd.DataFrame(columns=['Category', 'Vendor', 'Transaction Count', 'Subtotal ($)', 'Transactions'])
        
        # Group by category first, then vendor
        category_groups = {}
        for trans in withdrawals:
            category = trans.category or "Uncategorized"
            if category not in category_groups:
                category_groups[category] = {}
            
            vendor = trans.vendor or "Unknown Vendor"
            if vendor not in category_groups[category]:
                category_groups[category][vendor] = []
            category_groups[category][vendor].append(trans)
        
        # Create summary rows
        summary_rows = []
        for category in sorted(category_groups.keys()):
            category_total = 0
            first_in_category = True
            
            for vendor in sorted(category_groups[category].keys()):
                trans_list = category_groups[category][vendor]
                subtotal = abs(sum(t.amount for t in trans_list))
                category_total += subtotal
                count = len(trans_list)
                
                # Create detailed transaction list
                trans_details = []
                for t in sorted(trans_list, key=lambda x: x.date or ''):
                    date_str = t.date or 'N/A'
                    desc = t.description[:50] if t.description else 'N/A'
                    trans_details.append(f"{date_str} | ${abs(t.amount):,.2f} | {desc}")
                
                summary_rows.append({
                    'Category': category if first_in_category else '',
                    'Vendor': vendor,
                    'Transaction Count': count,
                    'Subtotal ($)': round(subtotal, 2),
                    'Transactions': '; '.join(trans_details)
                })
                first_in_category = False
            
            # Add category subtotal row
            summary_rows.append({
                'Category': f"{category} Subtotal",
                'Vendor': '',
                'Transaction Count': sum(len(category_groups[category][v]) for v in category_groups[category]),
                'Subtotal ($)': round(category_total, 2),
                'Transactions': ''
            })
        
        df = pd.DataFrame(summary_rows)
        # Add grand total row
        total_withdrawals = abs(sum(t.amount for t in withdrawals))
        total_row = pd.DataFrame([{
            'Category': 'TOTAL WITHDRAWALS',
            'Vendor': '',
            'Transaction Count': len(withdrawals),
            'Subtotal ($)': round(total_withdrawals, 2),
            'Transactions': ''
        }])
        df = pd.concat([df, total_row], ignore_index=True)
        
        return df
    
    def generate_needs_review_section(self, transactions: List[Transaction]) -> pd.DataFrame:
        """Generate section for transactions that need manual review"""
        needs_review = [t for t in transactions if t.needs_review]
        
        if not needs_review:
            return pd.DataFrame(columns=['Date', 'Description', 'Amount', 'Type', 'Line Number', 'Raw Line'])
        
        rows = []
        for trans in sorted(needs_review, key=lambda x: x.line_number or 0):
            rows.append({
                'Date': trans.date or 'N/A',
                'Description': trans.description[:100] if trans.description else 'N/A',
                'Amount': f"${abs(trans.amount):,.2f}" if trans.amount != 0 else 'N/A',
                'Type': trans.transaction_type.title(),
                'Line Number': trans.line_number or 'N/A',
                'Raw Line': trans.raw_line[:150] if trans.raw_line else 'N/A'
            })
        
        return pd.DataFrame(rows)
    
    def generate_pl_report(self, transactions: List[Transaction]) -> pd.DataFrame:
        """Generate Profit & Loss report"""
        deposits = [t for t in transactions if t.transaction_type == 'deposit' and t.amount > 0]
        withdrawals = [t for t in transactions if t.transaction_type == 'withdrawal' and t.amount < 0]
        
        total_deposits = sum(t.amount for t in deposits)
        total_withdrawals = abs(sum(t.amount for t in withdrawals))
        net_income = total_deposits - total_withdrawals
        
        # Group by category for expenses
        category_expenses = {}
        for trans in withdrawals:
            category = trans.category or "Uncategorized"
            if category not in category_expenses:
                category_expenses[category] = 0
            category_expenses[category] += abs(trans.amount)
        
        # Create report
        rows = []
        
        # Income section
        rows.append({
            'Category': 'INCOME',
            'Amount ($)': total_deposits,
            'Type': 'Income'
        })
        
        # Expense categories
        for category in sorted(category_expenses.keys()):
            rows.append({
                'Category': category,
                'Amount ($)': -category_expenses[category],  # Negative for expenses
                'Type': 'Expense'
            })
        
        # Total expenses
        rows.append({
            'Category': 'TOTAL EXPENSES',
            'Amount ($)': -total_withdrawals,
            'Type': 'Expense'
        })
        
        # Net income
        rows.append({
            'Category': 'NET INCOME',
            'Amount ($)': net_income,
            'Type': 'Net'
        })
        
        df = pd.DataFrame(rows)
        return df
    
    def generate_summary_statistics(self, transactions: List[Transaction]) -> Dict:
        """Generate summary statistics"""
        deposits = [t for t in transactions if t.transaction_type == 'deposit' and t.amount > 0]
        withdrawals = [t for t in transactions if t.transaction_type == 'withdrawal' and t.amount < 0]
        needs_review = [t for t in transactions if t.needs_review]
        
        return {
            'Total Deposits': len(deposits),
            'Total Withdrawals': len(withdrawals),
            'Total Deposit Amount': sum(t.amount for t in deposits),
            'Total Withdrawal Amount': abs(sum(t.amount for t in withdrawals)),
            'Net Income': sum(t.amount for t in deposits) - abs(sum(t.amount for t in withdrawals)),
            'Transactions Needing Review': len(needs_review),
            'Total Transactions': len(transactions)
        }
    
    def export_to_excel(self, transactions: List[Transaction], filename: str = "bank_statement_report.xlsx"):
        """Export all reports to Excel file"""
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Deposits summary
            deposits_df = self.generate_deposits_summary(transactions)
            deposits_df.to_excel(writer, sheet_name='Deposits Summary', index=False)
            
            # Withdrawals summary
            withdrawals_df = self.generate_withdrawals_summary(transactions)
            withdrawals_df.to_excel(writer, sheet_name='Withdrawals Summary', index=False)
            
            # P&L Report
            pl_df = self.generate_pl_report(transactions)
            pl_df.to_excel(writer, sheet_name='P&L Report', index=False)
            
            # Needs Review
            review_df = self.generate_needs_review_section(transactions)
            if not review_df.empty:
                review_df.to_excel(writer, sheet_name='Needs Review', index=False)
            
            # Summary statistics
            stats = self.generate_summary_statistics(transactions)
            stats_df = pd.DataFrame([stats])
            stats_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
        
        return filename

