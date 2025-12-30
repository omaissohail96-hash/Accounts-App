"""
Account Code Mapper
Maps Schedule C categories to QuickBooks-style account codes for P&L reporting
"""
import json
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


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
            "tiktok": ("601", "SALES"), "shopify": ("601", "SALES"), 
            "wise": ("808", "OFFSHORE EXP"), 
            "irs": ("821", "PAYROLL TAXES"),
            "credit union": ("999", "OTHER EXPENSES")
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
        Now with robust case-insensitive partial string matching.
        
        Algorithm:
        1. Sort categories by rank (1 = highest priority)
        2. For each category:
           - Normalize search text (lowercase, trim, remove extra spaces)
           - Check exclude_keywords - if any match, skip this category
           - Check include_keywords - if any match (case-insensitive partial), assign category and stop
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
        # Normalize: lowercase, strip, remove extra spaces
        search_text = ""
        if vendor:
            search_text += " ".join(vendor.lower().strip().split()) + " "
        if description:
            search_text += " ".join(description.lower().strip().split())
        
        search_text = " ".join(search_text.strip().split())  # Remove extra spaces
        
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
                keyword_normalized = " ".join(keyword.lower().strip().split())
                if keyword_normalized and keyword_normalized in search_text:
                    print(f"⛔ Excluded '{search_text[:50]}...' from {account_code} · {account_name} (exclude: '{keyword}')")
                    exclude_match = True
                    break
            
            if exclude_match:
                continue
            
            # Check include keywords - if any match (case-insensitive partial), assign this category
            for keyword in include_keywords:
                keyword_normalized = " ".join(keyword.lower().strip().split())
                if keyword_normalized and keyword_normalized in search_text:
                    print(f"✅ Matched '{search_text[:50]}...' → {account_code} · {account_name} (keyword: '{keyword}')")
                    return (account_code, account_name)
        
        print(f"❌ No keyword match for: '{search_text[:80]}...'")
        return None
    
    def get_account_code(self,
        vendor: Optional[str] = None,
        description: Optional[str] = None,
        is_income: bool = False,
        transaction_type: Optional[str] = None
    ) -> tuple:
        """
        Direction-first account mapping.

        Rules enforced here:
        1) If transaction direction indicates withdrawal (is_income False) => ALWAYS return an expense
           - Keyword matches that map to income are ignored
           - Default fallback: ("999", "OTHER EXPENSES")
        2) If transaction direction indicates deposit (is_income True) => ALWAYS return an income
           - Keyword matches that map to expenses are ignored
           - Default fallback: ("601", "SALES")
        3) Defensive: if explicit words in the text strongly indicate deposit/withdrawal, they override a wrong is_income flag
        4) Keyword JSON logic is preserved and used only to pick an account within the chosen direction
        """

        text = f"{vendor or ''} {description or ''}".lower()

        # Defensive detection of explicit direction words
        # include multi-word phrases that indicate money leaving the account
        # broadened to catch Zelle/ACH/online payment patterns that are withdrawals
        withdrawal_words = [
            "withdrawal", "withdraw", "payout", "fee", "charge", "debit", "atm",
            "payment to", "payment made to", "paid to", "transfer to", "transfer", "xfer to", "sent to",
            "payment sent", "zelle", "zelle payment", "zelle payment to", "online payment to", "ach payment to",
            "zelle transfer", "sent via zelle"
        ]
        # Avoid overly generic tokens like 'payment' which are ambiguous; keep deposit indicators focused
        deposit_words = ["deposit", "payment received", "payment from", "credit", "received", "refund", "deposit received"]

        forced_direction = None
        if any(w in text for w in withdrawal_words):
            forced_direction = "expense"
        elif any(w in text for w in deposit_words):
            forced_direction = "income"

        # Final direction precedence (DATA-DRIVEN):
        # 1) explicit transaction_type param (HIGHEST PRIORITY - source of truth)
        # 2) forced_direction detected from text phrases (only if transaction_type not provided)
        # 3) provided is_income boolean (fallback)
        if transaction_type:
            # transaction_type is the source of truth - accept common variants
            t = transaction_type.lower().strip()
            # treat any variant containing 'withdraw' as a withdrawal (expense)
            if 'withdraw' in t or t in ('debit', 'debit_card', 'payment_out', 'payment_sent', 'sent'):
                is_income_final = False
            # treat deposit/credit variants as income
            elif 'deposit' in t or t in ('credit', 'credit_card', 'payment_in', 'received'):
                is_income_final = True
            else:
                # Unknown transaction_type, fall back to other indicators
                is_income_final = True if forced_direction == "income" else False if forced_direction == "expense" else bool(is_income)
        else:
            # No transaction_type provided, use forced_direction or is_income
            is_income_final = True if forced_direction == "income" else False if forced_direction == "expense" else bool(is_income)

        def _is_income_code(code: str) -> bool:
            return bool(code) and str(code).strip().startswith("6")

        def _is_expense_code(code: str) -> bool:
            # treat 7xx/8xx/9xx and special 860 as expense buckets
            return bool(code) and (str(code).strip().startswith(("7", "8", "9")) or str(code).strip() == "860")

        # EXPENSE path
        if not is_income_final:
            # No processor-specific hard-coded rules here — direction wins.
            keyword_match = self._match_by_keywords(vendor, description)
            if keyword_match:
                code = keyword_match[0]
                if _is_expense_code(code):
                    logger.debug(f"get_account_code: text='{text[:80]}', keyword matched expense -> {keyword_match}")
                    return keyword_match
                # if matched code is income, ignore it because direction wins
            logger.debug(f"get_account_code: text='{text[:80]}', no expense keyword match -> fallback 999")
            return ("999", "OTHER EXPENSES")

        # INCOME path
        keyword_match = self._match_by_keywords(vendor, description)
        if keyword_match:
            code = keyword_match[0]
            if _is_income_code(code):
                logger.debug(f"get_account_code: text='{text[:80]}', keyword matched income -> {keyword_match}")
                return keyword_match
            # matched expense code - ignore because direction wins

        logger.debug(f"get_account_code: text='{text[:80]}', fallback income -> 601")
        return ("601", "SALES")


    def get_account_name_display(self, account_code: str, account_name: str) -> str:
        """
        Format account name for display (e.g., "601 · SALES")
        """
        return f"{account_code} · {account_name}"

