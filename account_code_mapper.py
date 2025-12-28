"""
Account Code Mapper
Maps Schedule C categories to QuickBooks-style account codes for P&L reporting
"""
import json
from pathlib import Path
from typing import Dict, Optional
from schedule_c_categorizer import ScheduleCCategory


class AccountCodeMapper:
    """
    Maps Schedule C categories to account codes (601, 701, 801, etc.)
    Based on standard chart of accounts structure
    """
    
    def __init__(self):
        # Load keyword-based rules from JSON
        self.keyword_rules = self._load_keyword_rules("account_keywords.json")
        
        # Map Schedule C tax codes to account codes and names
        self.account_code_map = {
            # INCOME (600 series)
            "GROSS": ("601", "SALES"),
            "RETURNS": ("602", "RETURNS & ALLOWANCES"),
            "OTHER_INCOME": ("603", "OTHER INCOME"),
            
            # COGS (700 series)
            "COGS_INVENTORY": ("701", "PURCHASES"),
            "COGS_FREIGHT": ("702", "SHIPPING SUPPLIES"),
            "COGS_MATERIALS": ("703", "DIRECT MATERIALS"),
            "COGS_CUSTOMS": ("704", "CUSTOMS & DUTIES"),
            
            # EXPENSES (800 series)
            "WAGES": ("801", "SALARIES-OFFICERS"),
            "CONTRACT_LABOR": ("807", "TEMPORARY HELP"),
            "OFFSHORE_LABOR": ("808", "OFFSHORE EXP"),  # Special case for offshore/contract labor
            "WAGES_PAYROLL_TAX": ("821", "PAYROLL TAXES"),
            "LEGAL_ACCOUNTING": ("851", "ACCOUNTING & LEGAL"),
            "ADVERTISING": ("854", "ADVERTISEMENT"),
            "BANK_FEES": ("860", "BANK & EBAY CHARGES"),
            "SUBSCRIPTIONS": ("880", "DUES AND SUBSCRIPTION"),
            "RENT": ("928", "RENT"),
            "SUPPLIES": ("935", "STORE SUPPLIES"),
            "UTILITIES": ("941", "TELEPHONE"),  # Default to telephone, can be more specific
            "TRAVEL": ("942", "TRAVEL"),
            "UTILITIES_ELECTRIC": ("946", "UTILITIES - ELECTRICITY"),
            "INSURANCE": ("950", "INSURANCE"),
            "REPAIRS": ("951", "REPAIRS & MAINTENANCE"),
            "VEHICLE": ("952", "CAR & TRUCK EXPENSES"),
            "OTHER_EXPENSES": ("999", "OTHER EXPENSES"),
        }
        
        # Special handling for specific vendors/descriptions
        self.vendor_specific_map = {
            "ebay": ("860", "BANK & EBAY CHARGES"),
            "paypal": ("860", "BANK & EBAY CHARGES"),
            "stripe": ("860", "BANK & EBAY CHARGES"),
            "electric": ("946", "UTILITIES - ELECTRICITY"),
            "electricity": ("946", "UTILITIES - ELECTRICITY"),
            "power": ("946", "UTILITIES - ELECTRICITY"),
            "offshore": ("808", "OFFSHORE EXP"),
            "fiverr": ("808", "OFFSHORE EXP"),
            "upwork": ("807", "TEMPORARY HELP"),
        }
    
    def _load_keyword_rules(self, file_path: str) -> dict:
        """
        Load keyword-based account mapping rules from JSON file.
        
        Args:
            file_path: Path to the JSON file containing keyword rules
            
        Returns:
            Dictionary of account mappings, or empty dict if file not found
        """
        try:
            json_path = Path(file_path)
            if json_path.exists():
                with open(json_path, 'r') as f:
                    return json.load(f)
            return {}
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load keyword rules from {file_path}: {e}")
            return {}
    
    def _match_by_keywords(self, vendor: Optional[str], description: Optional[str]) -> Optional[tuple]:
        """
        Match transaction against keyword rules from JSON file.
        Uses rank-based priority with include/exclude keyword logic.
        
        Algorithm:
        1. Sort categories by rank (1 = highest priority)
        2. For each category:
           - Normalize search text (lowercase, trim)
           - Check exclude_keywords - if any match, skip this category
           - Check include_keywords - if any match, assign this category and stop
        3. Return None if no match (will default to uncategorized)
        
        Args:
            vendor: Vendor name
            description: Transaction description
            
        Returns:
            Tuple of (account_code, account_name) if match found, None otherwise
        """
        if not self.keyword_rules:
            print(f"⚠️ Warning: No keyword rules loaded from JSON")
            return None
        
        # Combine vendor and description for keyword matching
        search_text = ""
        if vendor:
            search_text += vendor.lower().strip() + " "
        if description:
            search_text += description.lower().strip()
        
        search_text = search_text.strip()
        if not search_text:
            return None
        
        # Sort categories by rank (ascending - 1 is highest priority)
        sorted_categories = sorted(
            self.keyword_rules.items(),
            key=lambda x: x[1].get("rank", 999)
        )
        
        # Check each category in rank order
        for account_code, account_data in sorted_categories:
            account_name = account_data.get("name", "UNKNOWN")
            
            # Get include and exclude keywords
            include_keywords = account_data.get("include_keywords", [])
            exclude_keywords = account_data.get("exclude_keywords", [])
            
            # Check exclude keywords first - if any match, skip this category
            exclude_match = False
            for keyword in exclude_keywords:
                keyword_normalized = keyword.lower().strip()
                if keyword_normalized and keyword_normalized in search_text:
                    print(f"⛔ Excluded '{search_text[:50]}...' from {account_code} · {account_name} (exclude: '{keyword}')")
                    exclude_match = True
                    break
            
            if exclude_match:
                continue
            
            # Check include keywords - if any match, assign this category
            for keyword in include_keywords:
                keyword_normalized = keyword.lower().strip()
                if keyword_normalized and keyword_normalized in search_text:
                    print(f"✅ Matched '{search_text[:50]}...' → {account_code} · {account_name} (keyword: '{keyword}')")
                    return (account_code, account_name)
        
        print(f"❌ No keyword match for: '{search_text[:80]}...'")
        return None
    
    def get_account_code(self, vendor: Optional[str] = None, description: Optional[str] = None, category: Optional[ScheduleCCategory] = None, is_income: bool = False) -> tuple:
        """
        Get account code and name based on vendor and description.
        
        Args:
            vendor: Vendor name (for vendor-specific mapping)
            description: Description (for keyword and vendor-specific mapping)
            category: Optional ScheduleCCategory object (for backward compatibility)
            is_income: Whether this is an income transaction (deposit) or expense (withdrawal)
            
        Returns:
            Tuple of (account_code, account_name)
        """
        # Check keyword-based matching first (highest priority)
        keyword_match = self._match_by_keywords(vendor, description)
        if keyword_match:
            return keyword_match
        
        # Check vendor-specific mapping
        if vendor:
            vendor_lower = vendor.lower()
            for key, (code, name) in self.vendor_specific_map.items():
                if key in vendor_lower:
                    return (code, name)
        
        if description:
            desc_lower = description.lower()
            for key, (code, name) in self.vendor_specific_map.items():
                if key in desc_lower:
                    return (code, name)
        
        # Default fallback based on transaction type (no Schedule C logic)
        if is_income:
            return ("601", "SALES")  # Default income to SALES
        return ("999", "OTHER EXPENSES")  # Default expense to OTHER EXPENSES
    
    def get_account_name_display(self, account_code: str, account_name: str) -> str:
        """
        Format account name for display (e.g., "601 · SALES")
        """
        return f"{account_code} · {account_name}"

 
