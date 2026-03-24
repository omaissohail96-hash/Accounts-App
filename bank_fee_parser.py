"""
Bank Fee Parser Module
Robust transaction parsing logic that accurately detects bank fees
without misreading total transaction amounts.

Key Features:
- Extracts actual fee amount from description (not transaction total)
- Validates fee amounts are realistic (< $500)
- Uses regex to find amounts near fee keywords
- Handles multiple numbers in descriptions
- Sets needs_review flag when amounts are unclear
"""
import re
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BankFeeResult:
    """Result of bank fee parsing"""
    is_bank_fee: bool
    fee_amount: Optional[float] = None
    needs_review: bool = False
    vendor: str = "Bank Fees"
    reason: Optional[str] = None  # Why it needs review or why amount was chosen


class BankFeeParser:
    """
    Parses bank fees from transaction descriptions with intelligent amount extraction.
    """
    
    # Maximum realistic bank fee amount (for validation)
    MAX_FEE_AMOUNT = 500.0
    
    # Bank fee keywords (must match these to be considered a fee)
    # Using broader matching patterns
    BANK_FEE_KEYWORDS = [
        # Primary keywords
        "fee", "charge", "charged",
        # Specific fee types  
        "service", "monthly", "maintenance", "transaction",
        "processing", "nsf", "overdraft", "atm",
        "wire", "international", "platform", "ach",
        "returned item", "stop payment", "account",
        "initial fee", "withdrawal fee"
    ]
    
    # Regex patterns for extracting fee amounts
    # Pattern 1: "fee" or "charge" followed by amount
    FEE_PATTERN_1 = re.compile(
        r'(?:fee|charge|total\s+fees?|fees?\s+total)[^\d]*?([\d,]+\.?\d{0,2})',
        re.IGNORECASE
    )
    
    # Pattern 2: Amount followed by "fee" or "charge"
    FEE_PATTERN_2 = re.compile(
        r'([\d,]+\.?\d{0,2})[^\d]*?(?:fee|charge)',
        re.IGNORECASE
    )
    
    # Pattern 3: "Qty X amount Total Fees amount" pattern
    QTY_TOTAL_PATTERN = re.compile(
        r'qty\s+(\d+)\s+([\d,]+\.?\d{0,2})\s+total\s+fees?\s+([\d,]+\.?\d{0,2})',
        re.IGNORECASE
    )
    
    # Pattern 4: Generic amount extraction (fallback)
    AMOUNT_PATTERN = re.compile(r'([\d,]+\.\d{2})')
    
    def __init__(self):
        """Initialize the bank fee parser"""
        pass
    
    def parse_bank_fee(self, description: str, transaction_amount: float) -> BankFeeResult:
        """
        Parse a transaction description to detect and extract bank fee amount.
        
        Args:
            description: Transaction description text
            transaction_amount: The full transaction amount (may be different from fee)
            
        Returns:
            BankFeeResult with is_bank_fee, fee_amount, needs_review flags
        """
        if not description:
            return BankFeeResult(is_bank_fee=False)
        
        desc_lower = description.lower()
        
        # Step 1: Check if this is a bank fee at all (using improved matching)
        is_fee = self.is_bank_fee(description)
        
        if not is_fee:
            return BankFeeResult(is_bank_fee=False)
        
        # Step 2: Extract the actual fee amount from description
        fee_amount, reason = self._extract_fee_amount(description, transaction_amount)
        
        # Step 3: Validate the extracted amount
        needs_review = False
        
        if fee_amount is None:
            # Could not extract fee amount from description
            needs_review = True
            reason = "Bank fee detected but amount unclear - needs manual review"
            fee_amount = abs(transaction_amount)  # Use transaction amount as fallback
        
        elif fee_amount > self.MAX_FEE_AMOUNT:
            # Fee amount is suspiciously large
            needs_review = True
            reason = f"Fee amount ${fee_amount:.2f} exceeds typical range (>${self.MAX_FEE_AMOUNT}) - needs review"
        
        elif fee_amount == 0:
            # Zero fee doesn't make sense
            needs_review = True
            reason = "Fee amount is $0.00 - needs review"
        
        # Additional check: if extracted amount equals transaction amount,
        # it might mean we couldn't find a separate fee amount
        elif abs(fee_amount - abs(transaction_amount)) < 0.01:
            # The extracted fee equals the transaction total
            # This is OK for simple fee transactions, but flag multi-amount scenarios
            if len(self.AMOUNT_PATTERN.findall(description)) > 1:
                # Multiple amounts found but we ended up with transaction total
                needs_review = True
                reason = f"Multiple amounts found, but extracted amount matches transaction total - verify correct fee"
        
        # Step 4: Return result
        return BankFeeResult(
            is_bank_fee=True,
            fee_amount=abs(fee_amount) if fee_amount is not None else abs(transaction_amount),
            needs_review=needs_review,
            vendor="Bank Fees",
            reason=reason
        )
    
    def _extract_fee_amount(self, description: str, transaction_amount: float) -> Tuple[Optional[float], str]:
        """
        Extract the actual fee amount from description using multiple strategies.
        
        Returns:
            Tuple of (fee_amount, reason_for_choice)
        """
        # Strategy 1: "Qty X amount Total Fees amount" pattern
        # Example: "Standard ACH Pmnts Initial Fee Qty 3 7.50 Total Fees 7.50"
        qty_match = self.QTY_TOTAL_PATTERN.search(description)
        if qty_match:
            total_fee = self._clean_amount(qty_match.group(3))
            if total_fee is not None:
                return total_fee, "Extracted from 'Total Fees' label"
        
        # Strategy 2: "fee" keyword followed by amount
        # Example: "Service Fee 12.50"
        fee_after_match = self.FEE_PATTERN_1.search(description)
        if fee_after_match:
            amount = self._clean_amount(fee_after_match.group(1))
            if amount is not None and amount < self.MAX_FEE_AMOUNT:
                return amount, "Extracted amount after 'fee' keyword"
        
        # Strategy 3: Amount before "fee" keyword
        # Example: "12.50 Service Fee"
        fee_before_match = self.FEE_PATTERN_2.search(description)
        if fee_before_match:
            amount = self._clean_amount(fee_before_match.group(1))
            if amount is not None and amount < self.MAX_FEE_AMOUNT:
                return amount, "Extracted amount before 'fee' keyword"
        
        # Strategy 4: Find all amounts and choose the smallest realistic one
        # (Fees are typically smaller than transaction totals)
        all_amounts = self.AMOUNT_PATTERN.findall(description)
        if all_amounts:
            cleaned_amounts = []
            for amt_str in all_amounts:
                amt = self._clean_amount(amt_str)
                if amt is not None and 0 < amt < self.MAX_FEE_AMOUNT:
                    cleaned_amounts.append(amt)
            
            if cleaned_amounts:
                # Choose smallest amount (fees are usually small)
                min_amount = min(cleaned_amounts)
                return min_amount, f"Smallest amount from {len(cleaned_amounts)} candidates"
        
        # Strategy 5: No clear fee amount found, return None
        return None, "Could not extract fee amount from description"
    
    def _clean_amount(self, amount_str: str) -> Optional[float]:
        """
        Clean and convert amount string to float.
        
        Args:
            amount_str: String representation of amount (e.g., "1,234.56")
            
        Returns:
            Float value or None if conversion fails
        """
        try:
            # Remove commas and convert to float
            cleaned = amount_str.replace(',', '')
            return float(cleaned)
        except (ValueError, AttributeError):
            return None
    
    def is_bank_fee(self, description: str) -> bool:
        """
        Quick check if a description contains bank fee keywords.
        
        Args:
            description: Transaction description
            
        Returns:
            True if description contains bank fee keywords
        """
        if not description:
            return False
        
        desc_lower = description.lower()
        
        # Exclude patterns that are NOT fees (including balance-related entries)
        exclude_patterns = [
            'transfer to', 'transfer from', 'payment to', 'payment from',
            'zelle', 'quickpay', 'withdrawal to atm', 'deposit',
            'purchase', 'sale', 'refund', 'balance', 'ending balance',
            'opening balance', 'beginning balance', 'closing balance',
            'daily ending', 'average balance'
        ]
        
        if any(pattern in desc_lower for pattern in exclude_patterns):
            return False
        
        # Must contain fee-related words
        fee_indicators = ['fee', 'charge', 'charged']
        has_fee_word = any(indicator in desc_lower for indicator in fee_indicators)
        
        # Or specific service keywords
        service_keywords = [
            'nsf', 'overdraft', 'maintenance', 'service', 
            'platform', 'ach', 'wire'
        ]
        has_service_word = any(keyword in desc_lower for keyword in service_keywords)
        
        return has_fee_word or has_service_word
    
    def get_fee_info_dict(self, description: str, transaction_amount: float) -> Dict:
        """
        Get bank fee information as a dictionary (useful for APIs/logging).
        
        Args:
            description: Transaction description
            transaction_amount: Full transaction amount
            
        Returns:
            Dictionary with fee parsing results
        """
        result = self.parse_bank_fee(description, transaction_amount)
        
        return {
            'is_bank_fee': result.is_bank_fee,
            'fee_amount': result.fee_amount,
            'needs_review': result.needs_review,
            'vendor': result.vendor,
            'reason': result.reason,
            'original_description': description,
            'original_amount': transaction_amount
        }


# Convenience function for quick usage
def parse_bank_fee(description: str, transaction_amount: float) -> BankFeeResult:
    """
    Convenience function to parse bank fee from description.
    
    Example:
        result = parse_bank_fee("Service Fee 12.50 Total USD 5,000", -5000)
        if result.is_bank_fee:
            print(f"Fee: ${result.fee_amount}")
            if result.needs_review:
                print(f"Review needed: {result.reason}")
    
    Args:
        description: Transaction description
        transaction_amount: Full transaction amount
        
    Returns:
        BankFeeResult object
    """
    parser = BankFeeParser()
    return parser.parse_bank_fee(description, transaction_amount)
