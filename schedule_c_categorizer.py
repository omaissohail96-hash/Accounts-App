"""
Schedule C Categorizer Module
Categorizes transactions into IRS Schedule C structure for tax preparation.

Schedule C Structure:
- Part I: Income (Gross receipts/sales, Returns & allowances, Other income)
- Part II: Expenses (Advertising, Bank fees, Contract labor, Wages, Rent, Utilities, etc.)
- Part III: Cost of Goods Sold (COGS) - Inventory, freight-in, direct materials, customs, beginning/ending inventory
- Part IV: Vehicle Information (Car & truck expenses - business-related only)
- Part V: Other Expenses (Valid business expenses not fitting above categories)

Exclusion Rules:
- ATM withdrawals → Owner draw (NOT expense)
- Personal spending → Exclude
- Transfers between own accounts → Exclude
"""
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from bank_statement_parser import Transaction
import logging

try:
    import pandas as pd
except ImportError:
    pd = None

logger = logging.getLogger(__name__)


@dataclass
class ScheduleCCategory:
    """Represents a Schedule C category assignment for a transaction"""
    part: str  # "Part I", "Part II", "Part III", "Part IV", "Part V"
    line_number: Optional[str]  # e.g., "Line 1", "Line 8", "Line 27a"
    category_name: str  # e.g., "Gross receipts or sales", "Advertising"
    tax_code: str  # e.g., "GROSS", "ADVERTISING", "COGS_INVENTORY"
    is_excluded: bool = False  # True if transaction should be excluded
    exclusion_reason: Optional[str] = None  # Reason for exclusion if excluded
    is_owner_draw: bool = False  # True if this is an owner draw (ATM withdrawal)


class ScheduleCCategorizer:
    """
    Categorizes transactions into IRS Schedule C structure.
    Uses rule-based and keyword-based logic to match transactions to appropriate categories.
    """
    
    def __init__(self):
        # Build keyword rules for each category
        self._build_categorization_rules()
    
    def _build_categorization_rules(self):
        """Build comprehensive keyword-based categorization rules"""
        
        # PART I - INCOME
        self.income_gross_receipts = [
            # Payment processors
            "shopify", "shopify id", "ebay", "stripe", "paypal", "square", "venmo", "zelle",
            "tiktok", "tiktok shop", "meta pay", "google pay", "apple pay",
            # E-commerce platforms
            "amazon pay", "etsy", "woocommerce", "bigcommerce", "squarespace",
            # Deposits indicating sales
            "payment received", "invoice paid", "customer payment", "sale",
            "revenue", "receipt", "income", "payout", "transfer from",
            # Business income indicators
            "business income", "sales revenue", "gross receipts"
        ]
        
        self.income_returns_allowances = [
            "refund", "return", "chargeback", "reversal", "void",
            "customer refund", "return processing", "returned payment"
        ]
        
        self.income_other = [
            "interest income", "bank interest", "dividend", "royalty",
            "other income", "miscellaneous income"
        ]
        
        # PART III - COST OF GOODS SOLD (COGS)
        self.cogs_inventory_purchases = [
            "inventory", "stock", "product purchase", "merchandise",
            "wholesale", "bulk purchase", "goods purchase", "material purchase",
            "supplier payment", "vendor payment"
        ]
        
        self.cogs_freight_in = [
            "freight", "shipping in", "freight in", "inbound shipping",
            "receiving", "delivery fee", "cargo", "logistics",
            "shipping supplies", "shipping cost", "shipping expense"
        ]
        
        self.cogs_direct_materials = [
            "raw materials", "direct materials", "component", "part",
            "fabric", "ingredient", "supply purchase"
        ]
        
        self.cogs_customs_duties = [
            "customs", "duty", "import tax", "tariff", "import fee",
            "border fee", "import duty"
        ]
        
        # PART II - EXPENSES
        self.expense_advertising = [
            "google ads", "google adwords", "facebook ads", "meta ads",
            "instagram ads", "twitter ads", "linkedin ads", "tiktok ads",
            "youtube ads", "pinterest ads", "snapchat ads", "advertising",
            "marketing", "promotion", "sponsored", "ad spend", "ppc",
            "search ads", "display ads", "retargeting"
        ]
        
        self.expense_bank_fees = [
            "bank fee", "service charge", "monthly fee", "maintenance fee",
            "transaction fee", "processing fee", "nsf fee", "nsf charge", "overdraft",
            "atm fee", "wire transfer fee", "international fee",
            "platform fee", "payment processing fee"
        ]
        
        self.expense_contract_labor = [
            "upwork", "fiverr", "freelancer", "freelance", "contractor",
            "consultant", "offshore", "outsource", "contract labor",
            "independent contractor", "1099", "gig worker"
        ]
        
        self.expense_wages = [
            "payroll", "salary", "wage", "employee", "paycheck",
            "pay stub", "w-2"
        ]
        
        self.expense_payroll_taxes = [
            "payroll tax", "employer tax", "workers comp", "unemployment tax",
            "fica", "medicare tax", "social security tax", "state unemployment",
            "federal unemployment", "suta", "futa"
        ]
        
        self.expense_rent = [
            "rent", "lease", "landlord", "office rent", "warehouse rent",
            "storage rent", "facility rent"
        ]
        
        self.expense_utilities = [
            "electric", "electricity", "power", "gas utility", "water",
            "sewer", "internet", "broadband", "phone", "mobile", "cellular",
            "telephone", "utilities", "utility bill"
        ]
        
        self.expense_supplies = [
            "office supplies", "store supplies", "supplies", "stationery",
            "paper", "ink", "toner", "printer", "office depot", "staples",
            "amazon supplies"
        ]
        
        self.expense_travel = [
            "airline", "flight", "hotel", "lodging", "accommodation",
            "uber trip", "lyft trip", "taxi", "car rental", "travel",
            "conference travel", "business travel"
        ]
        
        self.expense_legal_accounting = [
            "legal", "attorney", "lawyer", "law firm", "legal services",
            "accounting", "cpa", "bookkeeping", "tax prep", "tax preparation",
            "professional services", "consulting"
        ]
        
        self.expense_insurance = [
            "insurance", "premium", "liability insurance", "business insurance",
            "general liability", "professional liability", "health insurance",
            "coverage"
        ]
        
        self.expense_repairs_maintenance = [
            "repair", "maintenance", "fix", "service", "upkeep",
            "equipment repair", "facility maintenance"
        ]
        
        self.expense_subscriptions = [
            "subscription", "saas", "software subscription", "monthly subscription",
            "annual subscription", "membership", "dues", "recurring",
            "netflix", "spotify", "adobe", "microsoft", "zoom", "slack",
            "software", "app subscription"
        ]
        
        # PART IV - VEHICLE EXPENSES (business-related only)
        self.vehicle_expenses = [
            "gas", "fuel", "petrol", "diesel", "gas station",
            "shell", "chevron", "exxon", "bp", "mobil",
            "parking", "toll", "vehicle maintenance", "car repair",
            "auto repair", "vehicle insurance", "car insurance",
            "vehicle registration", "car registration"
        ]
        
        # EXCLUSION KEYWORDS
        self.exclude_personal = [
            "personal", "grocery", "supermarket", "walmart", "target",
            "restaurant", "cafe", "starbucks", "mcdonald", "food",
            "dining", "entertainment", "netflix", "spotify", "hulu",
            "subscription personal", "gym", "fitness"
        ]
        
        self.exclude_transfers = [
            "transfer", "zelle", "quickpay", "internal transfer",
            "move money", "online transfer", "ach transfer",
            "account transfer", "between accounts"
        ]
        
        self.atm_keywords = [
            "atm", "atm withdrawal", "cash withdrawal", "atm fee"
        ]
        
        # Build lookup dictionaries for faster matching
        self._build_lookup_dicts()
    
    def _build_lookup_dicts(self):
        """Build reverse lookup dictionaries for faster categorization"""
        self.vendor_to_category_map = {}
        self.description_to_category_map = {}
        
        # Map all keywords to their categories
        category_mappings = [
            (self.income_gross_receipts, ("Part I", "Line 1", "Gross receipts or sales", "GROSS")),
            (self.income_returns_allowances, ("Part I", "Line 2", "Returns and allowances", "RETURNS")),
            (self.income_other, ("Part I", "Line 6", "Other income", "OTHER_INCOME")),
            (self.cogs_inventory_purchases, ("Part III", "Line 35", "Purchases", "COGS_INVENTORY")),
            (self.cogs_freight_in, ("Part III", "Line 36", "Shipping Supplies", "COGS_FREIGHT")),
            (self.cogs_direct_materials, ("Part III", "Line 38", "Materials and supplies", "COGS_MATERIALS")),
            (self.cogs_customs_duties, ("Part III", "Line 39", "Other costs", "COGS_CUSTOMS")),
            (self.expense_advertising, ("Part II", "Line 8", "Advertising", "ADVERTISING")),
            (self.expense_bank_fees, ("Part II", "Line 18", "Office expense", "BANK_FEES")),
            (self.expense_contract_labor, ("Part II", "Line 11", "Contract labor", "CONTRACT_LABOR")),
            (self.expense_wages, ("Part II", "Line 26", "Wages and salaries", "WAGES")),
            (self.expense_rent, ("Part II", "Line 20", "Rent or lease", "RENT")),
            (self.expense_utilities, ("Part II", "Line 25", "Utilities", "UTILITIES")),
            (self.expense_supplies, ("Part II", "Line 22", "Supplies", "SUPPLIES")),
            (self.expense_travel, ("Part II", "Line 24a", "Travel", "TRAVEL")),
            (self.expense_legal_accounting, ("Part II", "Line 17", "Legal and professional services", "LEGAL_ACCOUNTING")),
            (self.expense_insurance, ("Part II", "Line 15", "Insurance", "INSURANCE")),
            (self.expense_repairs_maintenance, ("Part II", "Line 21", "Repairs and maintenance", "REPAIRS")),
            (self.expense_subscriptions, ("Part II", "Line 27a", "Other expenses", "SUBSCRIPTIONS")),
            (self.vehicle_expenses, ("Part IV", "Line 9", "Car and truck expenses", "VEHICLE")),
        ]
        
        for keywords, (part, line, name, code) in category_mappings:
            for keyword in keywords:
                self.vendor_to_category_map[keyword.lower()] = (part, line, name, code)
                self.description_to_category_map[keyword.lower()] = (part, line, name, code)
    
    def categorize_transaction(self, transaction: Transaction) -> ScheduleCCategory:
        """
        Categorize a single transaction into Schedule C structure.
        
        Args:
            transaction: Transaction object with vendor, description, amount, transaction_type
            
        Returns:
            ScheduleCCategory object with categorization details
        """
        vendor_lower = (transaction.vendor or "").lower()
        desc_lower = (transaction.description or "").lower()
        combined_text = f"{vendor_lower} {desc_lower}".lower()
        
        # STEP 1: Check exclusion rules first
        exclusion_result = self._check_exclusions(transaction, combined_text)
        if exclusion_result:
            return exclusion_result
        
        # STEP 2: Check if this is a payment processor transaction (Shopify, eBay, Amazon, etc.)
        # These should be categorized by keywords, not by transaction direction
        # This ensures ALL Shopify deposits AND withdrawals are properly categorized
        is_payment_processor = any(keyword in combined_text for keyword in [
            "shopify", "ebay", "amazon", "etsy", "tiktok", "stripe", "paypal",
            "square", "venmo", "mercari", "poshmark", "walmart marketplace"
        ])
        
        # STEP 3: Handle deposits OR payment processor transactions (Income)
        # Payment processor transactions are categorized as income regardless of direction
        # (unless they contain fee/charge keywords, which are caught by exclusions in account_keywords.json)
        if transaction.amount > 0 or transaction.transaction_type == "deposit" or is_payment_processor:
            return self._categorize_income(transaction, combined_text)
        
        # STEP 4: Handle withdrawals (Expenses or COGS)
        # Check COGS first (inventory-related costs)
        cogs_result = self._categorize_cogs(transaction, combined_text)
        if cogs_result:
            return cogs_result
        
        # Then check regular expenses
        expense_result = self._categorize_expenses(transaction, combined_text)
        if expense_result:
            return expense_result
        
        # Check vehicle expenses
        vehicle_result = self._categorize_vehicle(transaction, combined_text)
        if vehicle_result:
            return vehicle_result
        
        # Default: Other expenses (Part V)
        return ScheduleCCategory(
            part="Part V",
            line_number="Line 27a",
            category_name="Other expenses",
            tax_code="OTHER_EXPENSES",
            is_excluded=False
        )
    
    def _check_exclusions(self, transaction: Transaction, combined_text: str) -> Optional[ScheduleCCategory]:
        """Check if transaction should be excluded or marked as owner draw"""
        
        # Check transfers FIRST (before personal, as transfers are more specific)
        # BUT: Exempt payment processors and check payments from transfer exclusion
        # (Shopify, eBay, check payments, etc. use "Transfer" in descriptions but are legitimate expenses)
        # Transfers between own accounts → Exclude
        is_payment_processor = any(keyword in combined_text for keyword in [
            "shopify", "ebay", "amazon", "etsy", "tiktok", "stripe", "paypal",
            "square", "venmo", "mercari", "poshmark", "walmart marketplace"
        ])
        
        is_check_payment = any(keyword in combined_text for keyword in [
            "check payment", "chk", "check #", "check number"
        ])
        
        if not is_payment_processor and not is_check_payment and any(keyword in combined_text for keyword in self.exclude_transfers):
            return ScheduleCCategory(
                part="Excluded",
                line_number=None,
                category_name="Account transfer",
                tax_code="TRANSFER",
                is_excluded=True,
                exclusion_reason="Transfer between own accounts - Not income or expense"
            )
        
        # ATM withdrawals → Mark as Other Expenses (will be mapped to 999 in P&L)
        # Note: For Schedule C tax purposes, these would be owner draws,
        # but for P&L accounting, we show them as OTHER EXPENSES
        if any(keyword in combined_text for keyword in self.atm_keywords):
            if transaction.transaction_type == "withdrawal" or transaction.amount < 0:
                return ScheduleCCategory(
                    part="Part V",
                    line_number="Line 27a",
                    category_name="Other expenses",
                    tax_code="OTHER_EXPENSES",
                    is_excluded=False,
                    is_owner_draw=False,  # Changed: Show in P&L as expense, not owner draw
                    exclusion_reason=None
                )
        
        # Personal spending → Exclude (check last, after more specific exclusions)
        if any(keyword in combined_text for keyword in self.exclude_personal):
            # Additional check: if it's clearly personal (restaurant for personal meal, etc.)
            # For now, we'll flag it but the user may want to review
            return ScheduleCCategory(
                part="Excluded",
                line_number=None,
                category_name="Personal expense",
                tax_code="PERSONAL",
                is_excluded=True,
                exclusion_reason="Personal spending - Not a business expense"
            )
        
        return None
    
    def _categorize_income(self, transaction: Transaction, combined_text: str) -> ScheduleCCategory:
        """Categorize income transactions (deposits)"""
        
        # Returns & allowances (negative amounts or refund keywords)
        if any(keyword in combined_text for keyword in self.income_returns_allowances):
            return ScheduleCCategory(
                part="Part I",
                line_number="Line 2",
                category_name="Returns and allowances",
                tax_code="RETURNS"
            )
        
        # Other income
        if any(keyword in combined_text for keyword in self.income_other):
            return ScheduleCCategory(
                part="Part I",
                line_number="Line 6",
                category_name="Other income",
                tax_code="OTHER_INCOME"
            )
        
        # Default: Gross receipts/sales
        return ScheduleCCategory(
            part="Part I",
            line_number="Line 1",
            category_name="Gross receipts or sales",
            tax_code="GROSS"
        )
    
    def _categorize_cogs(self, transaction: Transaction, combined_text: str) -> Optional[ScheduleCCategory]:
        """Categorize Cost of Goods Sold (COGS) expenses"""
        
        # Inventory purchases
        if any(keyword in combined_text for keyword in self.cogs_inventory_purchases):
            return ScheduleCCategory(
                part="Part III",
                line_number="Line 35",
                category_name="Purchases",
                tax_code="COGS_INVENTORY"
            )
        
        # Freight-in
        if any(keyword in combined_text for keyword in self.cogs_freight_in):
            return ScheduleCCategory(
                part="Part III",
                line_number="Line 36",
                category_name="Cost of labor",
                tax_code="COGS_FREIGHT"
            )
        
        # Direct materials
        if any(keyword in combined_text for keyword in self.cogs_direct_materials):
            return ScheduleCCategory(
                part="Part III",
                line_number="Line 38",
                category_name="Materials and supplies",
                tax_code="COGS_MATERIALS"
            )
        
        # Customs/duties
        if any(keyword in combined_text for keyword in self.cogs_customs_duties):
            return ScheduleCCategory(
                part="Part III",
                line_number="Line 39",
                category_name="Other costs",
                tax_code="COGS_CUSTOMS"
            )
        
        return None
    
    def _categorize_expenses(self, transaction: Transaction, combined_text: str) -> Optional[ScheduleCCategory]:
        """Categorize business expenses"""
        
        # Check payments - handle these as OTHER EXPENSES (before bank fees check)
        # This catches check payments that might otherwise be misclassified
        check_keywords = ["check payment", "chk ", " chk", "check #", "check number", "check transaction"]
        if any(keyword in combined_text for keyword in check_keywords):
            return ScheduleCCategory(
                part="Part V",
                line_number="Line 27a",
                category_name="Other expenses",
                tax_code="OTHER_EXPENSES"
            )
        
        # Advertising
        if any(keyword in combined_text for keyword in self.expense_advertising):
            return ScheduleCCategory(
                part="Part II",
                line_number="Line 8",
                category_name="Advertising",
                tax_code="ADVERTISING"
            )
        
        # Bank & platform fees
        if any(keyword in combined_text for keyword in self.expense_bank_fees):
            return ScheduleCCategory(
                part="Part II",
                line_number="Line 18",
                category_name="Office expense",
                tax_code="BANK_FEES"
            )
        
        # Contract labor
        if any(keyword in combined_text for keyword in self.expense_contract_labor):
            return ScheduleCCategory(
                part="Part II",
                line_number="Line 11",
                category_name="Contract labor",
                tax_code="CONTRACT_LABOR"
            )
        
        # Payroll taxes (check before wages)
        if any(keyword in combined_text for keyword in self.expense_payroll_taxes):
            return ScheduleCCategory(
                part="Part II",
                line_number="Line 23",
                category_name="Taxes and licenses",
                tax_code="WAGES_PAYROLL_TAX"
            )
        
        # Wages & payroll
        if any(keyword in combined_text for keyword in self.expense_wages):
            return ScheduleCCategory(
                part="Part II",
                line_number="Line 26",
                category_name="Wages and salaries",
                tax_code="WAGES"
            )
        
        # Rent
        if any(keyword in combined_text for keyword in self.expense_rent):
            return ScheduleCCategory(
                part="Part II",
                line_number="Line 20",
                category_name="Rent or lease",
                tax_code="RENT"
            )
        
        # Utilities
        if any(keyword in combined_text for keyword in self.expense_utilities):
            return ScheduleCCategory(
                part="Part II",
                line_number="Line 25",
                category_name="Utilities",
                tax_code="UTILITIES"
            )
        
        # Supplies
        if any(keyword in combined_text for keyword in self.expense_supplies):
            return ScheduleCCategory(
                part="Part II",
                line_number="Line 22",
                category_name="Supplies",
                tax_code="SUPPLIES"
            )
        
        # Travel
        if any(keyword in combined_text for keyword in self.expense_travel):
            return ScheduleCCategory(
                part="Part II",
                line_number="Line 24a",
                category_name="Travel",
                tax_code="TRAVEL"
            )
        
        # Legal & accounting
        if any(keyword in combined_text for keyword in self.expense_legal_accounting):
            return ScheduleCCategory(
                part="Part II",
                line_number="Line 17",
                category_name="Legal and professional services",
                tax_code="LEGAL_ACCOUNTING"
            )
        
        # Insurance
        if any(keyword in combined_text for keyword in self.expense_insurance):
            return ScheduleCCategory(
                part="Part II",
                line_number="Line 15",
                category_name="Insurance",
                tax_code="INSURANCE"
            )
        
        # Repairs & maintenance
        if any(keyword in combined_text for keyword in self.expense_repairs_maintenance):
            return ScheduleCCategory(
                part="Part II",
                line_number="Line 21",
                category_name="Repairs and maintenance",
                tax_code="REPAIRS"
            )
        
        # Subscriptions & dues
        if any(keyword in combined_text for keyword in self.expense_subscriptions):
            return ScheduleCCategory(
                part="Part II",
                line_number="Line 27a",
                category_name="Other expenses",
                tax_code="SUBSCRIPTIONS"
            )
        
        return None
    
    def _categorize_vehicle(self, transaction: Transaction, combined_text: str) -> Optional[ScheduleCCategory]:
        """Categorize vehicle expenses (business-related only)"""
        
        if any(keyword in combined_text for keyword in self.vehicle_expenses):
            return ScheduleCCategory(
                part="Part IV",
                line_number="Line 9",
                category_name="Car and truck expenses",
                tax_code="VEHICLE"
            )
        
        return None
    
    def categorize_transactions(self, transactions: List[Transaction]) -> List[Tuple[Transaction, ScheduleCCategory]]:
        """
        Categorize a list of transactions.
        
        Args:
            transactions: List of Transaction objects
            
        Returns:
            List of tuples (Transaction, ScheduleCCategory)
        """
        results = []
        for transaction in transactions:
            category = self.categorize_transaction(transaction)
            results.append((transaction, category))
        return results
    
    def generate_schedule_c_summary(self, categorized_transactions: List[Tuple[Transaction, ScheduleCCategory]]) -> Dict:
        """
        Generate a summary grouped by Schedule C categories.
        Suitable for Schedule C automation and bookkeeping reports.
        
        Args:
            categorized_transactions: List of (Transaction, ScheduleCCategory) tuples
            
        Returns:
            Dictionary with summary statistics by category
        """
        summary = {
            "Part I - Income": {},
            "Part II - Expenses": {},
            "Part III - COGS": {},
            "Part IV - Vehicle": {},
            "Part V - Other Expenses": {},
            "Excluded": {},
            "Owner Draws": []
        }
        
        # Map category parts to summary dictionary keys
        part_to_summary_key = {
            "Part I": "Part I - Income",
            "Part II": "Part II - Expenses",
            "Part III": "Part III - COGS",
            "Part IV": "Part IV - Vehicle",
            "Part V": "Part V - Other Expenses",
            "Excluded": "Excluded"
        }
        
        for transaction, category in categorized_transactions:
            # Handle owner draws separately
            if category.is_owner_draw:
                summary["Owner Draws"].append({
                    "date": transaction.date,
                    "vendor": transaction.vendor,
                    "description": transaction.description,
                    "amount": abs(transaction.amount),
                    "reason": category.exclusion_reason
                })
                continue
            
            # Determine which part this belongs to
            part = category.part
            if category.is_excluded:
                part = "Excluded"
            
            # Map part to summary key
            summary_key = part_to_summary_key.get(part, part)
            
            # Skip if no valid part/line number (shouldn't happen, but safety check)
            if not category.line_number:
                continue
            
            # Initialize category if not exists
            category_key = f"{category.line_number} - {category.category_name}"
            if summary_key not in summary:
                summary[summary_key] = {}
            if category_key not in summary[summary_key]:
                summary[summary_key][category_key] = {
                    "tax_code": category.tax_code,
                    "line_number": category.line_number,
                    "category_name": category.category_name,
                    "total_amount": 0.0,
                    "transaction_count": 0,
                    "transactions": []
                }
            
            # Add transaction data
            # For income (positive amounts), use as-is; for expenses (negative amounts), use absolute value
            if transaction.amount > 0:
                # Income
                amount = transaction.amount
            else:
                # Expenses/COGS - use absolute value
                amount = abs(transaction.amount)
            
            summary[summary_key][category_key]["total_amount"] += amount
            summary[summary_key][category_key]["transaction_count"] += 1
            summary[summary_key][category_key]["transactions"].append({
                "date": transaction.date,
                "vendor": transaction.vendor,
                "description": transaction.description,
                "amount": amount
            })
        
        return summary
    
    def generate_schedule_c_report(self, categorized_transactions: List[Tuple[Transaction, ScheduleCCategory]]) -> str:
        """
        Generate a human-readable Schedule C report.
        
        Args:
            categorized_transactions: List of (Transaction, ScheduleCCategory) tuples
            
        Returns:
            Formatted string report
        """
        summary = self.generate_schedule_c_summary(categorized_transactions)
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SCHEDULE C - BUSINESS INCOME AND EXPENSES")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Part I - Income
        report_lines.append("PART I - INCOME")
        report_lines.append("-" * 80)
        income_total = 0.0
        for category_key, data in summary.get("Part I - Income", {}).items():
            amount = data["total_amount"]
            income_total += amount
            report_lines.append(f"  {data['line_number']}: {data['category_name']:40} ${amount:>12,.2f} ({data['transaction_count']} transactions)")
        report_lines.append(f"  {'TOTAL INCOME':>60} ${income_total:>12,.2f}")
        report_lines.append("")
        
        # Part III - COGS
        report_lines.append("PART III - COST OF GOODS SOLD")
        report_lines.append("-" * 80)
        cogs_total = 0.0
        for category_key, data in summary.get("Part III - COGS", {}).items():
            amount = data["total_amount"]
            cogs_total += amount
            report_lines.append(f"  {data['line_number']}: {data['category_name']:40} ${amount:>12,.2f} ({data['transaction_count']} transactions)")
        report_lines.append(f"  {'TOTAL COGS':>60} ${cogs_total:>12,.2f}")
        report_lines.append("")
        
        # Part II - Expenses
        report_lines.append("PART II - EXPENSES")
        report_lines.append("-" * 80)
        expenses_total = 0.0
        for category_key, data in summary.get("Part II - Expenses", {}).items():
            amount = data["total_amount"]
            expenses_total += amount
            report_lines.append(f"  {data['line_number']}: {data['category_name']:40} ${amount:>12,.2f} ({data['transaction_count']} transactions)")
        report_lines.append(f"  {'TOTAL EXPENSES':>60} ${expenses_total:>12,.2f}")
        report_lines.append("")
        
        # Part IV - Vehicle
        report_lines.append("PART IV - VEHICLE EXPENSES")
        report_lines.append("-" * 80)
        vehicle_total = 0.0
        for category_key, data in summary.get("Part IV - Vehicle", {}).items():
            amount = data["total_amount"]
            vehicle_total += amount
            report_lines.append(f"  {data['line_number']}: {data['category_name']:40} ${amount:>12,.2f} ({data['transaction_count']} transactions)")
        if vehicle_total > 0:
            report_lines.append(f"  {'TOTAL VEHICLE EXPENSES':>60} ${vehicle_total:>12,.2f}")
        else:
            report_lines.append("  No vehicle expenses")
        report_lines.append("")
        
        # Part V - Other Expenses
        report_lines.append("PART V - OTHER EXPENSES")
        report_lines.append("-" * 80)
        other_total = 0.0
        for category_key, data in summary.get("Part V - Other Expenses", {}).items():
            amount = data["total_amount"]
            other_total += amount
            report_lines.append(f"  {data['line_number']}: {data['category_name']:40} ${amount:>12,.2f} ({data['transaction_count']} transactions)")
        if other_total > 0:
            report_lines.append(f"  {'TOTAL OTHER EXPENSES':>60} ${other_total:>12,.2f}")
        else:
            report_lines.append("  No other expenses")
        report_lines.append("")
        
        # Summary
        report_lines.append("=" * 80)
        report_lines.append("SUMMARY")
        report_lines.append("-" * 80)
        gross_profit = income_total - cogs_total
        total_expenses = expenses_total + vehicle_total + other_total
        net_income = gross_profit - total_expenses
        
        report_lines.append(f"  Gross Receipts/Sales:                              ${income_total:>12,.2f}")
        report_lines.append(f"  Less: Cost of Goods Sold:                          ${cogs_total:>12,.2f}")
        report_lines.append(f"  Gross Profit:                                       ${gross_profit:>12,.2f}")
        report_lines.append(f"  Less: Total Expenses:                               ${total_expenses:>12,.2f}")
        report_lines.append(f"  NET BUSINESS INCOME:                                ${net_income:>12,.2f}")
        report_lines.append("")
        
        # Owner Draws
        if summary["Owner Draws"]:
            report_lines.append("OWNER DRAWS (Not Expenses)")
            report_lines.append("-" * 80)
            owner_draw_total = sum(draw["amount"] for draw in summary["Owner Draws"])
            for draw in summary["Owner Draws"]:
                report_lines.append(f"  {draw['date']} - {draw['description'][:50]:50} ${draw['amount']:>12,.2f}")
            report_lines.append(f"  {'TOTAL OWNER DRAWS':>60} ${owner_draw_total:>12,.2f}")
            report_lines.append("")
        
        # Excluded
        if summary.get("Excluded"):
            report_lines.append("EXCLUDED TRANSACTIONS (Review Required)")
            report_lines.append("-" * 80)
            for category_key, data in summary["Excluded"].items():
                report_lines.append(f"  {data['category_name']:40} ${data['total_amount']:>12,.2f} ({data['transaction_count']} transactions)")
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def generate_income_statement_report(self, categorized_transactions: List[Tuple[Transaction, ScheduleCCategory]], period: str = "") -> str:
        """
        Generate an income statement-style report similar to QuickBooks/P&L format.
        Shows Income, COGS, Gross Profit, Expenses, and Net Income with percentages.
        
        Args:
            categorized_transactions: List of (Transaction, ScheduleCCategory) tuples
            period: Period label (e.g., "Oct 25" or "Q1 2024")
            
        Returns:
            Formatted string report in income statement format
        """
        summary = self.generate_schedule_c_summary(categorized_transactions)
        
        # Calculate totals
        income_total = sum(
            data["total_amount"]
            for category_key, data in summary.get("Part I - Income", {}).items()
        )
        
        cogs_total = sum(
            data["total_amount"]
            for category_key, data in summary.get("Part III - COGS", {}).items()
        )
        
        expenses_total = sum(
            data["total_amount"]
            for category_key, data in summary.get("Part II - Expenses", {}).items()
        )
        
        vehicle_total = sum(
            data["total_amount"]
            for category_key, data in summary.get("Part IV - Vehicle", {}).items()
        )
        
        other_total = sum(
            data["total_amount"]
            for category_key, data in summary.get("Part V - Other Expenses", {}).items()
        )
        
        total_expenses = expenses_total + vehicle_total + other_total
        gross_profit = income_total - cogs_total
        net_income = gross_profit - total_expenses
        
        # Generate report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"ORDINARY INCOME/EXPENSE - {period}" if period else "ORDINARY INCOME/EXPENSE")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Header
        report_lines.append(f"{'Category':<50} {'Amount':>15} {'% of Income':>15}")
        report_lines.append("-" * 80)
        
        # PART I - INCOME
        report_lines.append("INCOME:")
        income_items = sorted(
            summary.get("Part I - Income", {}).items(),
            key=lambda x: x[1]["total_amount"],
            reverse=True
        )
        for category_key, data in income_items:
            amount = data["total_amount"]
            pct = (amount / income_total * 100) if income_total > 0 else 0
            # Format category name with line number
            category_display = f"{data['line_number']} · {data['category_name']}"
            report_lines.append(f"  {category_display:<48} ${amount:>13,.2f} {pct:>14.1f}%")
        
        report_lines.append(f"  {'Total Income':<50} ${income_total:>13,.2f} {100.0:>14.1f}%")
        report_lines.append("")
        
        # PART III - COGS
        if cogs_total > 0:
            report_lines.append("COST OF GOODS SOLD:")
            cogs_items = sorted(
                summary.get("Part III - COGS", {}).items(),
                key=lambda x: x[1]["total_amount"],
                reverse=True
            )
            for category_key, data in cogs_items:
                amount = data["total_amount"]
                pct = (amount / income_total * 100) if income_total > 0 else 0
                category_display = f"{data['line_number']} · {data['category_name']}"
                report_lines.append(f"  {category_display:<48} ${amount:>13,.2f} {pct:>14.1f}%")
            
            report_lines.append(f"  {'Total COGS':<50} ${cogs_total:>13,.2f} {(cogs_total/income_total*100) if income_total > 0 else 0:>14.1f}%")
            report_lines.append("")
            
            # Gross Profit
            gross_profit_pct = (gross_profit / income_total * 100) if income_total > 0 else 0
            report_lines.append(f"{'Gross Profit':<50} ${gross_profit:>13,.2f} {gross_profit_pct:>14.1f}%")
            report_lines.append("")
        
        # PART II, IV, V - EXPENSES
        report_lines.append("EXPENSE:")
        
        # Combine all expense categories
        all_expenses = []
        
        # Part II Expenses
        for category_key, data in summary.get("Part II - Expenses", {}).items():
            all_expenses.append((data["line_number"], data["category_name"], data["total_amount"], data["tax_code"]))
        
        # Part IV Vehicle
        for category_key, data in summary.get("Part IV - Vehicle", {}).items():
            all_expenses.append((data["line_number"], data["category_name"], data["total_amount"], data["tax_code"]))
        
        # Part V Other
        for category_key, data in summary.get("Part V - Other Expenses", {}).items():
            all_expenses.append((data["line_number"], data["category_name"], data["total_amount"], data["tax_code"]))
        
        # Sort expenses by amount (descending)
        all_expenses.sort(key=lambda x: x[2], reverse=True)
        
        for line_num, category_name, amount, tax_code in all_expenses:
            pct = (amount / income_total * 100) if income_total > 0 else 0
            category_display = f"{line_num} · {category_name.upper()}"
            report_lines.append(f"  {category_display:<48} ${amount:>13,.2f} {pct:>14.1f}%")
        
        if total_expenses > 0:
            total_exp_pct = (total_expenses / income_total * 100) if income_total > 0 else 0
            report_lines.append(f"  {'Total Expense':<50} ${total_expenses:>13,.2f} {total_exp_pct:>14.1f}%")
        else:
            report_lines.append(f"  {'Total Expense':<50} ${0:>13,.2f} {0:>14.1f}%")
        report_lines.append("")
        
        # Net Ordinary Income
        net_income_pct = (net_income / income_total * 100) if income_total > 0 else 0
        report_lines.append(f"{'Net Ordinary Income':<50} ${net_income:>13,.2f} {net_income_pct:>14.1f}%")
        report_lines.append("")
        
        # Net Income
        report_lines.append(f"{'Net Income':<50} ${net_income:>13,.2f} {net_income_pct:>14.1f}%")
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def generate_income_statement_dict(self, categorized_transactions: List[Tuple[Transaction, ScheduleCCategory]]) -> Dict:
        """
        Generate an income statement-style dictionary structure suitable for CSV/Excel export.
        
        Args:
            categorized_transactions: List of (Transaction, ScheduleCCategory) tuples
            
        Returns:
            Dictionary with structured income statement data
        """
        summary = self.generate_schedule_c_summary(categorized_transactions)
        
        # Calculate totals
        income_total = sum(
            data["total_amount"]
            for category_key, data in summary.get("Part I - Income", {}).items()
        )
        
        cogs_total = sum(
            data["total_amount"]
            for category_key, data in summary.get("Part III - COGS", {}).items()
        )
        
        expenses_total = sum(
            data["total_amount"]
            for category_key, data in summary.get("Part II - Expenses", {}).items()
        )
        
        vehicle_total = sum(
            data["total_amount"]
            for category_key, data in summary.get("Part IV - Vehicle", {}).items()
        )
        
        other_total = sum(
            data["total_amount"]
            for category_key, data in summary.get("Part V - Other Expenses", {}).items()
        )
        
        total_expenses = expenses_total + vehicle_total + other_total
        gross_profit = income_total - cogs_total
        net_income = gross_profit - total_expenses
        
        result = {
            "income": [],
            "cogs": [],
            "expenses": [],
            "totals": {
                "income_total": income_total,
                "cogs_total": cogs_total,
                "gross_profit": gross_profit,
                "total_expenses": total_expenses,
                "net_income": net_income
            }
        }
        
        # Income items
        for category_key, data in summary.get("Part I - Income", {}).items():
            amount = data["total_amount"]
            pct = (amount / income_total * 100) if income_total > 0 else 0
            result["income"].append({
                "line_number": data["line_number"],
                "category": data["category_name"],
                "amount": amount,
                "percent_of_income": round(pct, 1),
                "tax_code": data["tax_code"]
            })
        
        # COGS items
        for category_key, data in summary.get("Part III - COGS", {}).items():
            amount = data["total_amount"]
            pct = (amount / income_total * 100) if income_total > 0 else 0
            result["cogs"].append({
                "line_number": data["line_number"],
                "category": data["category_name"],
                "amount": amount,
                "percent_of_income": round(pct, 1),
                "tax_code": data["tax_code"]
            })
        
        # Expense items (combined from Part II, IV, V)
        all_expenses = []
        for category_key, data in summary.get("Part II - Expenses", {}).items():
            all_expenses.append(data)
        for category_key, data in summary.get("Part IV - Vehicle", {}).items():
            all_expenses.append(data)
        for category_key, data in summary.get("Part V - Other Expenses", {}).items():
            all_expenses.append(data)
        
        for data in sorted(all_expenses, key=lambda x: x["total_amount"], reverse=True):
            amount = data["total_amount"]
            pct = (amount / income_total * 100) if income_total > 0 else 0
            result["expenses"].append({
                "line_number": data["line_number"],
                "category": data["category_name"],
                "amount": amount,
                "percent_of_income": round(pct, 1),
                "tax_code": data["tax_code"]
            })
        
        return result
    
    def generate_schedule_c_dataframe(self, categorized_transactions: List[Tuple[Transaction, ScheduleCCategory]]):
        """
        Generate a pandas DataFrame in the format used by the Streamlit UI:
        columns: schedule_c_line, tax_line_code, tax_line_description, raw_total_amount, deductible_amount, count
        
        Args:
            categorized_transactions: List of (Transaction, ScheduleCCategory) tuples
            
        Returns:
            pandas DataFrame with Schedule C data
        """
        if pd is None:
            raise ImportError("pandas is required for generate_schedule_c_dataframe")
        
        rows = []
        
        for transaction, category in categorized_transactions:
            # Skip excluded transactions and owner draws (they're tracked separately)
            if category.is_excluded or category.is_owner_draw:
                continue
            
            # Skip if no line number (shouldn't happen, but safety check)
            if not category.line_number:
                continue
            
            # Get amount (positive for income, absolute value for expenses/COGS)
            if transaction.amount > 0:
                amount = transaction.amount
            else:
                amount = abs(transaction.amount)
            
            rows.append({
                "schedule_c_line": category.line_number,
                "tax_line_code": category.tax_code,
                "tax_line_description": category.category_name,
                "amount": amount
            })
        
        if not rows:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=[
                "schedule_c_line",
                "tax_line_code",
                "tax_line_description",
                "raw_total_amount",
                "deductible_amount",
                "count"
            ])
        
        df = pd.DataFrame(rows)
        
        # Group by schedule_c_line, tax_line_code, tax_line_description
        grouped = df.groupby(
            ["schedule_c_line", "tax_line_code", "tax_line_description"],
            as_index=False
        ).agg(
            raw_total_amount=("amount", "sum"),
            deductible_amount=("amount", "sum"),  # For now, same as raw_total_amount (can be adjusted for partial deductions later)
            count=("amount", "count")
        )
        
        # Sort by line number for better readability
        # Extract numeric part from line number for sorting
        def extract_line_num(line_str):
            match = re.search(r'(\d+)', line_str)
            return float(match.group(1)) if match else 999.0
        
        grouped["line_num_sort"] = grouped["schedule_c_line"].apply(extract_line_num)
        grouped = grouped.sort_values("line_num_sort").drop("line_num_sort", axis=1).reset_index(drop=True)
        
        return grouped
    
    def generate_pl_report_with_account_codes(self, categorized_transactions: List[Tuple[Transaction, ScheduleCCategory]], 
                                               business_name: str = "", period: str = "") -> str:
        """
        Generate a Profit & Loss report with account codes (601, 701, 801, etc.) 
        in the format requested by the client.
        
        Format:
        BUSINESS NAME
        Profit & Loss
        PERIOD
        
        Ordinary Income/Expense
        Income
        601 · SALES          amount    %
        ...
        
        Args:
            categorized_transactions: List of (Transaction, ScheduleCCategory) tuples
            business_name: Business name (e.g., "EAST2WEST TRADING INC")
            period: Period label (e.g., "October 2025")
            
        Returns:
            Formatted string report
        """
        from account_code_mapper import AccountCodeMapper
        
        mapper = AccountCodeMapper()
        summary = self.generate_schedule_c_summary(categorized_transactions)
        
        # Calculate totals
        income_total = sum(
            data["total_amount"]
            for category_key, data in summary.get("Part I - Income", {}).items()
        )
        
        cogs_total = sum(
            data["total_amount"]
            for category_key, data in summary.get("Part III - COGS", {}).items()
        )
        
        expenses_total = sum(
            data["total_amount"]
            for category_key, data in summary.get("Part II - Expenses", {}).items()
        )
        
        vehicle_total = sum(
            data["total_amount"]
            for category_key, data in summary.get("Part IV - Vehicle", {}).items()
        )
        
        other_total = sum(
            data["total_amount"]
            for category_key, data in summary.get("Part V - Other Expenses", {}).items()
        )
        
        total_expenses = expenses_total + vehicle_total + other_total
        gross_profit = income_total - cogs_total
        net_income = gross_profit - total_expenses
        
        # Build report
        report_lines = []
        
        # Header
        if business_name:
            report_lines.append(business_name)
        report_lines.append("")
        report_lines.append("Profit & Loss")
        if period:
            report_lines.append(period)
        report_lines.append("")
        # Extract period label (e.g., "Oct 25" from "October 2025")
        # Format: "Month Day" like "Oct 25"
        period_label = period
        if period:
            month_map = {
                "January": "Jan", "February": "Feb", "March": "Mar", "April": "Apr",
                "May": "May", "June": "Jun", "July": "Jul", "August": "Aug",
                "September": "Sep", "October": "Oct", "November": "Nov", "December": "Dec"
            }
            parts = period.split()
            if len(parts) >= 1 and parts[0] in month_map:
                # Extract month abbreviation
                month_abbr = month_map[parts[0]]
                # Try to extract day from transactions, default to 25
                if categorized_transactions:
                    dates = [tx.date for tx, cat in categorized_transactions if tx.date]
                    if dates:
                        try:
                            from datetime import datetime
                            sample_date = dates[0]
                            if isinstance(sample_date, str):
                                date_obj = datetime.strptime(sample_date, "%Y-%m-%d")
                            else:
                                date_obj = sample_date
                            day = date_obj.strftime("%d").lstrip("0") or "1"
                            period_label = f"{month_abbr} {day}"
                        except:
                            period_label = f"{month_abbr} 25"
                    else:
                        period_label = f"{month_abbr} 25"
                else:
                    period_label = f"{month_abbr} 25"
        
        report_lines.append(f"{period_label} % of Income")
        report_lines.append("")
        report_lines.append("Ordinary Income/Expense")
        report_lines.append("")
        
        # INCOME
        report_lines.append("Income")
        income_items = []
        for category_key, data in summary.get("Part I - Income", {}).items():
            # Get transactions for this category to determine account code
            transactions_for_category = [
                tx for tx, cat in categorized_transactions
                if cat.line_number == data["line_number"] and cat.tax_code == data["tax_code"]
            ]
            vendor = transactions_for_category[0].vendor if transactions_for_category else None
            description = transactions_for_category[0].description if transactions_for_category else None
            
            # Create a temporary category object for mapping
            temp_category = ScheduleCCategory(
                part=data.get("part", "Part I"),
                line_number=data["line_number"],
                category_name=data["category_name"],
                tax_code=data["tax_code"]
            )
            
            account_code, account_name = mapper.get_account_code(vendor, description, is_income=True)
            amount = data["total_amount"]
            pct = (amount / income_total * 100) if income_total > 0 else 0
            income_items.append((account_code, account_name, amount, pct))
        
        # Sort income items by account code
        income_items.sort(key=lambda x: x[0])
        for account_code, account_name, amount, pct in income_items:
            display_name = mapper.get_account_name_display(account_code, account_name)
            report_lines.append(f"{display_name:<40} {amount:>12,.2f} {pct:>6.1f}%")
        
        report_lines.append(f"{'Total Income':<40} {income_total:>12,.2f} {100.0:>6.1f}%")
        report_lines.append("")
        
        # COGS
        if cogs_total > 0:
            report_lines.append("Cost of Goods Sold")
            cogs_items = []
            for category_key, data in summary.get("Part III - COGS", {}).items():
                transactions_for_category = [
                    tx for tx, cat in categorized_transactions
                    if cat.line_number == data["line_number"] and cat.tax_code == data["tax_code"]
                ]
                vendor = transactions_for_category[0].vendor if transactions_for_category else None
                description = transactions_for_category[0].description if transactions_for_category else None
                
                temp_category = ScheduleCCategory(
                    part="Part III",
                    line_number=data["line_number"],
                    category_name=data["category_name"],
                    tax_code=data["tax_code"]
                )
                
                account_code, account_name = mapper.get_account_code(vendor, description, is_income=False)
                amount = data["total_amount"]
                pct = (amount / income_total * 100) if income_total > 0 else 0
                cogs_items.append((account_code, account_name, amount, pct))
            
            cogs_items.sort(key=lambda x: x[0])
            for account_code, account_name, amount, pct in cogs_items:
                display_name = mapper.get_account_name_display(account_code, account_name)
                report_lines.append(f"{display_name:<40} {amount:>12,.2f} {pct:>6.1f}%")
            
            cogs_pct = (cogs_total / income_total * 100) if income_total > 0 else 0
            report_lines.append(f"{'Total COGS':<40} {cogs_total:>12,.2f} {cogs_pct:>6.1f}%")
            report_lines.append("")
            
            # Gross Profit
            gross_profit_pct = (gross_profit / income_total * 100) if income_total > 0 else 0
            report_lines.append(f"{'Gross Profit':<40} {gross_profit:>12,.2f} {gross_profit_pct:>6.1f}%")
            report_lines.append("")
        
        # EXPENSES
        report_lines.append("Expense")
        expense_items = []
        
        # Combine all expenses
        all_expense_data = []
        for category_key, data in summary.get("Part II - Expenses", {}).items():
            all_expense_data.append(data)
        for category_key, data in summary.get("Part IV - Vehicle", {}).items():
            all_expense_data.append(data)
        for category_key, data in summary.get("Part V - Other Expenses", {}).items():
            all_expense_data.append(data)
        
        for data in all_expense_data:
            transactions_for_category = [
                tx for tx, cat in categorized_transactions
                if cat.line_number == data["line_number"] and cat.tax_code == data["tax_code"]
            ]
            vendor = transactions_for_category[0].vendor if transactions_for_category else None
            description = transactions_for_category[0].description if transactions_for_category else None
            
            temp_category = ScheduleCCategory(
                part=data.get("part", "Part II"),
                line_number=data["line_number"],
                category_name=data["category_name"],
                tax_code=data["tax_code"]
            )
            
            account_code, account_name = mapper.get_account_code(vendor, description, is_income=False)
            amount = data["total_amount"]
            pct = (amount / income_total * 100) if income_total > 0 else 0
            expense_items.append((account_code, account_name, amount, pct))
        
        # Sort expenses by account code
        expense_items.sort(key=lambda x: x[0])
        for account_code, account_name, amount, pct in expense_items:
            display_name = mapper.get_account_name_display(account_code, account_name)
            report_lines.append(f"{display_name:<40} {amount:>12,.2f} {pct:>6.1f}%")
        
        total_exp_pct = (total_expenses / income_total * 100) if income_total > 0 else 0
        report_lines.append(f"{'Total Expense':<40} {total_expenses:>12,.2f} {total_exp_pct:>6.1f}%")
        report_lines.append("")
        
        # Net Income
        net_income_pct = (net_income / income_total * 100) if income_total > 0 else 0
        report_lines.append(f"{'Net Ordinary Income':<40} {net_income:>12,.2f} {net_income_pct:>6.1f}%")
        report_lines.append("")
        report_lines.append(f"{'Net Income':<40} {net_income:>12,.2f} {net_income_pct:>6.1f}%")
        report_lines.append("")
        report_lines.append("Page 1")
        
        return "\n".join(report_lines)

