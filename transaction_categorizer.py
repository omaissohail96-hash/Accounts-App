"""
Transaction Categorizer Module
Categorizes transactions and groups them by vendor/source
"""
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from bank_statement_parser import Transaction
import logging

logger = logging.getLogger(__name__)


class TransactionCategorizer:
    """Categorizes bank transactions and groups by vendor"""
    
    def __init__(self):
        # Categorization rules - vendor keywords mapped to categories
        self.categorization_rules = {
            # Advertising & Marketing
            'Advertising': ['google ads', 'google adwords', 'facebook ads', 'meta ads', 'instagram ads', 
                          'twitter ads', 'linkedin ads', 'advertising', 'marketing', 'promotion'],
            'Shipping': ['fedex', 'ups', 'usps', 'dhl', 'postal', 'shipping', 'delivery', 'courier'],
            'Office Supplies': ['amazon', 'office depot', 'staples', 'office max', 'supplies'],
            'Rent': ['rent', 'lease', 'landlord', 'apartment', 'housing'],
            'Utilities': ['electric', 'gas', 'water', 'utility', 'power', 'energy', 'utility bill'],
            'Internet': ['comcast', 'verizon', 'at&t', 'spectrum', 'internet', 'isp', 'broadband'],
            'Phone': ['verizon', 'at&t', 't-mobile', 'sprint', 'phone', 'mobile', 'cellular'],
            'Bank Fees': ['service charge', 'monthly fee', 'atm fee', 'overdraft', 'nsf', 'bank fee', 
                         'maintenance fee', 'transaction fee', 'service fee'],
            'Travel': ['uber', 'lyft', 'taxi', 'airline', 'hotel', 'booking', 'expedia', 'travel'],
            'Meals': ['starbucks', 'restaurant', 'mcdonald', 'subway', 'pizza', 'food', 'dining', 'cafe'],
            'Groceries': ['walmart', 'target', 'kroger', 'safeway', 'whole foods', 'grocery', 'supermarket'],
            'Fuel': ['shell', 'bp', 'exxon', 'chevron', 'gas station', 'fuel', 'petrol'],
            'Software Subscription': ['paypal', 'stripe', 'subscription', 'software', 'saas', 'platform'],
            'Design Tools': ['canva', 'adobe', 'figma', 'sketch', 'design'],
            'Communication': ['zoom', 'slack', 'teams', 'communication', 'conference'],
            'E-commerce Fees': ['shopify', 'ebay', 'etsy', 'marketplace', 'e-commerce'],
            'Entertainment': ['netflix', 'spotify', 'hulu', 'disney', 'entertainment', 'streaming'],
            'Insurance': ['insurance', 'premium', 'coverage'],
            'Legal & Professional': ['legal', 'attorney', 'lawyer', 'accounting', 'cpa', 'consulting'],
            'Client Payment': ['upwork', 'fiverr', 'client', 'payment received', 'invoice paid'],
        }
        
        # Reverse mapping for quick lookup
        self._build_reverse_map()
    
    def _build_reverse_map(self):
        """Build reverse lookup map for faster categorization"""
        self.vendor_to_category = {}
        for category, keywords in self.categorization_rules.items():
            for keyword in keywords:
                self.vendor_to_category[keyword.lower()] = category
    
    def extract_vendor(self, description: str) -> str:
        """Extract vendor name from transaction description"""
        description_lower = description.lower()
        
        # Common patterns for vendor extraction
        # Try to find known vendors first
        for keyword in self.vendor_to_category.keys():
            if keyword in description_lower:
                # Find the full vendor name context
                idx = description_lower.find(keyword)
                # Extract surrounding words
                words = description.split()
                for i, word in enumerate(words):
                    if keyword in word.lower():
                        # Get vendor name (typically 1-3 words around the keyword)
                        start = max(0, i - 1)
                        end = min(len(words), i + 3)
                        vendor = ' '.join(words[start:end])
                        return vendor[:50]  # Limit length
        
        # If no known vendor, extract first few words as vendor
        words = description.split()
        if len(words) >= 2:
            return ' '.join(words[:3])[:50]
        else:
            return description[:50] if description else "Unknown Vendor"
    
    def categorize_transaction(self, transaction: Transaction) -> str:
        """Categorize a transaction based on description"""
        description_lower = transaction.description.lower()
        
        # Check each keyword
        for keyword, category in self.vendor_to_category.items():
            if keyword in description_lower:
                return category
        
        # Special handling for deposits
        if transaction.transaction_type == 'deposit':
            if any(word in description_lower for word in ['payment', 'received', 'deposit', 'credit', 'refund']):
                # Check if it's from a client
                if any(word in description_lower for word in ['client', 'customer', 'invoice', 'payment from']):
                    return 'Client Payment'
                else:
                    return 'Income'
        
        # Default category
        return 'Uncategorized'
    
    def process_transactions(self, transactions: List[Transaction]) -> List[Transaction]:
        """Process transactions: extract vendors and categorize"""
        for trans in transactions:
            if not trans.vendor:
                trans.vendor = self.extract_vendor(trans.description)
            if not trans.category:
                trans.category = self.categorize_transaction(trans)
        
        return transactions
    
    def group_by_vendor(self, transactions: List[Transaction]) -> Dict[str, List[Transaction]]:
        """Group transactions by vendor"""
        groups = {}
        for trans in transactions:
            vendor = trans.vendor or "Unknown Vendor"
            if vendor not in groups:
                groups[vendor] = []
            groups[vendor].append(trans)
        return groups
    
    def group_by_category(self, transactions: List[Transaction]) -> Dict[str, List[Transaction]]:
        """Group transactions by category"""
        groups = {}
        for trans in transactions:
            category = trans.category or "Uncategorized"
            if category not in groups:
                groups[category] = []
            groups[category].append(trans)
        return groups
    
    def detect_duplicates(self, transactions: List[Transaction]) -> List[Transaction]:
        """Detect and remove duplicate transactions"""
        seen = set()
        unique_transactions = []
        
        for trans in transactions:
            # Create a unique key based on date, amount, and description
            # Allow small variations in description (normalize)
            desc_normalized = re.sub(r'[^a-z0-9]', '', trans.description.lower())[:30]
            key = (
                trans.date,
                round(trans.amount, 2),
                desc_normalized
            )
            
            if key not in seen:
                seen.add(key)
                unique_transactions.append(trans)
            else:
                logger.info(f"Duplicate detected and removed: {trans.description[:50]}")
        
        return unique_transactions

