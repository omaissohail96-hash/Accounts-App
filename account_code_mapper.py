"""
Account Code Mapper
Maps Schedule C categories to QuickBooks-style account codes for P&L reporting
"""
from typing import Dict, Optional
from schedule_c_categorizer import ScheduleCCategory


class AccountCodeMapper:
    """
    Maps Schedule C categories to account codes (601, 701, 801, etc.)
    Based on standard chart of accounts structure
    """
    
    def __init__(self):
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
    
    def get_account_code(self, category: ScheduleCCategory, vendor: Optional[str] = None, description: Optional[str] = None) -> tuple:
        """
        Get account code and name for a Schedule C category.
        
        Args:
            category: ScheduleCCategory object
            vendor: Vendor name (for vendor-specific mapping)
            description: Description (for vendor-specific mapping)
            
        Returns:
            Tuple of (account_code, account_name)
        """
        # Check vendor-specific mapping first
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
        
        # Check tax code mapping
        tax_code = category.tax_code
        
        # Special handling for contract labor vs offshore
        if tax_code == "CONTRACT_LABOR":
            if vendor and any(k in vendor.lower() for k in ["offshore", "fiverr", "overseas"]):
                return ("808", "OFFSHORE EXP")
            if vendor and any(k in vendor.lower() for k in ["upwork", "freelancer"]):
                return ("807", "TEMPORARY HELP")
            return ("807", "TEMPORARY HELP")
        
        # Special handling for utilities
        if tax_code == "UTILITIES":
            if description and any(k in description.lower() for k in ["electric", "electricity", "power"]):
                return ("946", "UTILITIES - ELECTRICITY")
            return ("941", "TELEPHONE")
        
        # Default mapping
        return self.account_code_map.get(tax_code, ("999", "OTHER EXPENSES"))
    
    def get_account_name_display(self, account_code: str, account_name: str) -> str:
        """
        Format account name for display (e.g., "601 · SALES")
        """
        return f"{account_code} · {account_name}"

