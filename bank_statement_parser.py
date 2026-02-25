"""
Multi-Bank Statement Parser Module
Supports parsing of different bank statement formats with automatic bank detection.
"""
import re
import dateparser
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ============================
# UNIFIED DATA MODELS
# ============================

class BankName(Enum):
    """Supported bank identifiers"""
    CHASE = "chase"
    AMEX = "amex"
    BMO = "bmo"
    BOA = "boa"
    FIFTH_THIRD = "fifth_third"
    US_BANK = "us_bank"
    UNKNOWN = "unknown"


class TransactionType(Enum):
    """Transaction type classification"""
    DEBIT = "debit"
    CREDIT = "credit"


class TransactionCategory(Enum):
    """Transaction categories"""
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    CHECK = "check"
    CARD = "card"
    FEE = "fee"
    INTEREST = "interest"
    TRANSFER = "transfer"
    ATM = "atm"
    ACH = "ach"
    UNKNOWN = "unknown"


@dataclass
class Transaction:
    """Unified transaction model for all banks"""
    date: datetime
    description: str
    amount: float  # negative = debit, positive = credit
    type: TransactionType
    bank_name: str
    account_type: Optional[str] = None
    category: Optional[TransactionCategory] = None
    running_balance: Optional[float] = None
    raw_line: str = ""
    check_number: Optional[str] = None
    needs_review: bool = False


@dataclass
class StatementPeriod:
    """Statement period"""
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None


@dataclass
class ParsedStatement:
    """Complete parsed statement with metadata"""
    bank_name: str
    transactions: List[Transaction] = field(default_factory=list)
    account_number: Optional[str] = None
    account_type: Optional[str] = None
    statement_period: Optional[StatementPeriod] = None
    beginning_balance: Optional[float] = None
    ending_balance: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


# ============================
# BANK DETECTION
# ============================

def detect_bank(text: str) -> BankName:
    """
    Detect bank from statement text by inspecting first page.
    
    Args:
        text: Full text from PDF statement (preferably first page)
    
    Returns:
        BankName enum value
    """
    # Get first 3000 characters for detection (typically first page)
    first_page = text[:3000].upper()
    
    # Chase detection
    if "JPMORGAN CHASE" in first_page or "CHASE BUSINESS COMPLETE" in first_page:
        return BankName.CHASE
    
    # Amex detection
    if "AMERICAN EXPRESS" in first_page or "PAYMENT DUE DATE" in first_page:
        return BankName.AMEX
    
    # BMO detection
    if "BMO" in first_page and ("BMO ELITE BUSINESS" in first_page or "BANKING SUMMARY" in first_page):
        return BankName.BMO
    
    # Bank of America detection
    if "BANK OF AMERICA" in first_page and "YOUR CHECKING ACCOUNT" in first_page:
        return BankName.BOA
    
    # Fifth Third detection
    if "FIFTH THIRD BANK" in first_page:
        return BankName.FIFTH_THIRD
    
    # US Bank detection
    if "U.S. BANK SILVER" in first_page or "USBANK" in first_page:
        return BankName.US_BANK
    
    logger.warning("Unable to detect bank from statement text")
    return BankName.UNKNOWN


# ============================
# BANK CONFIGURATION
# ============================

BANK_CONFIG = {
    BankName.CHASE: {
        "sections": [
            "DEPOSITS AND ADDITIONS",
            "CHECKS PAID",
            "ATM & DEBIT CARD WITHDRAWALS",
            "ELECTRONIC WITHDRAWALS",
            "OTHER WITHDRAWALS",
            "FEES"
        ],
        "transaction_regex": r'^(\d{2}/\d{2})\s+(.+?)\s+([\d,]+\.?\d*)\s*$',
        "balance_section": "DAILY LEDGER BALANCES"
    },
    BankName.BMO: {
        "sections": [
            "MONTHLY ACTIVITY DETAILS",
            "MONTHLY ACTIVITY DETAILS (CONT'D)"
        ],
        "transaction_regex": r'^([A-Z][a-z]{2}\s+\d{2})\s+(.+?)\s+(\$?[\d,]+\.\d{2})?\s+(\$?[\d,]+\.\d{2})?\s+(\$?[\d,]+\.\d{2})$',
    },
    BankName.BOA: {
        "sections": [
            "DEPOSITS AND OTHER CREDITS",
            "WITHDRAWALS AND OTHER DEBITS",
            "CHECKS"
        ],
        "transaction_regex": r'^(\d{2}/\d{2})\s+(.+?)\s+([\d,]+\.\d{2})\s*$',
    },
    BankName.FIFTH_THIRD: {
        "sections": [
            "WITHDRAWALS / DEBITS",
            "DEPOSITS / CREDITS"
        ],
        "transaction_regex": r'^(\d{2}/\d{2})\s+(.+?)\s+([\d,]+\.\d{2})\s*$',
    },
    BankName.US_BANK: {
        "sections": [
            "OTHER DEPOSITS",
            "OTHER WITHDRAWALS"
        ],
        "transaction_regex": r'^(\d{2}/\d{2})\s+(.+?)\s+([\d,]+\.\d{2})\s*$',
    },
    BankName.AMEX: {
        "sections": [
            "PAYMENTS AND CREDITS",
            "PURCHASES",
            "FEES"
        ],
        "transaction_regex": r'^(\d{2}/\d{2})\s+(.+?)\s+([\d,]+\.\d{2})\s*$',
    }
}


# ============================
# PARSER INTERFACE
# ============================

class BankStatementParser(ABC):
    """Abstract base class for bank-specific parsers"""
    
    def __init__(self, bank_name: BankName):
        self.bank_name = bank_name
        self.config = BANK_CONFIG.get(bank_name, {})
    
    @abstractmethod
    def parse(self, text: str) -> ParsedStatement:
        """
        Parse statement text into structured data.
        
        Args:
            text: Full text extracted from PDF statement
        
        Returns:
            ParsedStatement with transactions and metadata
        """
        pass
    
    def _parse_date(self, date_str: str, year: Optional[int] = None) -> Optional[datetime]:
        """Parse date string with various formats"""
        try:
            # Handle MM/DD format (need year)
            if re.match(r'^\d{2}/\d{2}$', date_str):
                if year:
                    date_str = f"{date_str}/{year}"
                else:
                    date_str = f"{date_str}/{datetime.now().year}"
            
            # Handle MMM DD format
            if re.match(r'^[A-Z][a-z]{2}\s+\d{2}$', date_str):
                if year:
                    date_str = f"{date_str} {year}"
                else:
                    date_str = f"{date_str} {datetime.now().year}"
            
            parsed = dateparser.parse(date_str)
            return parsed
        except Exception as e:
            logger.warning(f"Failed to parse date '{date_str}': {e}")
            return None
    
    def _parse_amount(self, amount_str: str) -> Optional[float]:
        """Parse amount string, handling various formats"""
        try:
            # Remove currency symbols and commas
            cleaned = amount_str.replace('$', '').replace(',', '').strip()
            
            # Handle parentheses (negative)
            if cleaned.startswith('(') and cleaned.endswith(')'):
                cleaned = '-' + cleaned[1:-1]
            
            return float(cleaned)
        except (ValueError, AttributeError):
            return None
    
    def _clean_description(self, desc: str) -> str:
        """Clean and normalize transaction description"""
        # Remove extra whitespace
        desc = ' '.join(desc.split())
        # Remove common prefixes
        desc = re.sub(r'^(PURCHASE|PAYMENT|DEBIT|CREDIT)\s+', '', desc, flags=re.IGNORECASE)
        return desc.strip()


# ============================
# CHASE PARSER
# ============================

class ChaseParser(BankStatementParser):
    """Parser for Chase Business Complete Checking statements"""
    
    def __init__(self):
        super().__init__(BankName.CHASE)
    
    def parse(self, text: str) -> ParsedStatement:
        """Parse Chase statement"""
        statement = ParsedStatement(bank_name=BankName.CHASE.value)
        lines = text.split('\n')
        
        # Extract statement period and account info
        self._extract_metadata(lines, statement)
        
        # Parse transactions by section
        current_section = None
        in_transaction_section = False
        statement_year = datetime.now().year
        
        # Try to extract year from statement period
        period_match = re.search(r'(\d{4})', text[:1000])
        if period_match:
            statement_year = int(period_match.group(1))
        
        for i, line in enumerate(lines):
            line_upper = line.upper().strip()
            
            # Detect section headers
            if "DEPOSITS AND ADDITIONS" in line_upper:
                current_section = (TransactionCategory.DEPOSIT, TransactionType.CREDIT)
                in_transaction_section = True
                continue
            elif "CHECKS PAID" in line_upper:
                current_section = (TransactionCategory.CHECK, TransactionType.DEBIT)
                in_transaction_section = True
                continue
            elif "ATM & DEBIT CARD WITHDRAWALS" in line_upper or "ATM AND DEBIT CARD" in line_upper:
                current_section = (TransactionCategory.ATM, TransactionType.DEBIT)
                in_transaction_section = True
                continue
            elif "ELECTRONIC WITHDRAWALS" in line_upper:
                current_section = (TransactionCategory.ACH, TransactionType.DEBIT)
                in_transaction_section = True
                continue
            elif "OTHER WITHDRAWALS" in line_upper:
                current_section = (TransactionCategory.WITHDRAWAL, TransactionType.DEBIT)
                in_transaction_section = True
                continue
            elif "FEES" in line_upper and "SERVICE" not in line_upper:
                current_section = (TransactionCategory.FEE, TransactionType.DEBIT)
                in_transaction_section = True
                continue
            elif "DAILY LEDGER BALANCES" in line_upper or "ENDING BALANCE" in line_upper:
                in_transaction_section = False
                continue
            
            # Parse transaction lines
            if in_transaction_section and current_section:
                transaction = self._parse_chase_transaction(line, current_section, statement_year)
                if transaction:
                    statement.transactions.append(transaction)
        
        if not statement.transactions:
            statement.errors.append("no_transactions_parsed")
        
        return statement
    
    def _extract_metadata(self, lines: List[str], statement: ParsedStatement):
        """Extract account number, period, balances from Chase statement"""
        text = '\n'.join(lines[:50])  # Check first 50 lines
        
        # Account number
        acc_match = re.search(r'ACCOUNT\s+NUMBER[:\s]+(\d+)', text, re.IGNORECASE)
        if acc_match:
            statement.account_number = acc_match.group(1)
        
        # Statement period
        period_match = re.search(r'(\d{2}/\d{2}/\d{4})\s+TO\s+(\d{2}/\d{2}/\d{4})', text, re.IGNORECASE)
        if period_match:
            from_date = dateparser.parse(period_match.group(1))
            to_date = dateparser.parse(period_match.group(2))
            statement.statement_period = StatementPeriod(from_date=from_date, to_date=to_date)
        
        # Beginning balance
        begin_match = re.search(r'BEGINNING\s+BALANCE[:\s]+\$?([\d,]+\.\d{2})', text, re.IGNORECASE)
        if begin_match:
            statement.beginning_balance = self._parse_amount(begin_match.group(1))
        
        # Ending balance
        end_match = re.search(r'ENDING\s+BALANCE[:\s]+\$?([\d,]+\.\d{2})', text, re.IGNORECASE)
        if end_match:
            statement.ending_balance = self._parse_amount(end_match.group(1))
    
    def _parse_chase_transaction(self, line: str, section_info: Tuple, year: int) -> Optional[Transaction]:
        """Parse a single Chase transaction line"""
        line = line.strip()
        if not line or len(line) < 10:
            return None
        
        # Skip header and summary lines
        if any(keyword in line.upper() for keyword in ['DATE', 'DESCRIPTION', 'AMOUNT', 'TOTAL', 'CONTINUED']):
            return None
        
        category, trans_type = section_info
        
        # Chase format: MM/DD  Description  Amount
        # Try to match date at start
        date_match = re.match(r'^(\d{2}/\d{2})\s+(.+?)\s+([\d,]+\.?\d*)\s*$', line)
        if not date_match:
            # Try alternative format with check number
            date_match = re.match(r'^(\d{2}/\d{2})\s+(\d+)\s+(.+?)\s+([\d,]+\.?\d*)\s*$', line)
            if date_match:
                date_str = date_match.group(1)
                check_num = date_match.group(2)
                description = date_match.group(3)
                amount_str = date_match.group(4)
            else:
                return None
        else:
            date_str = date_match.group(1)
            description = date_match.group(2)
            amount_str = date_match.group(3)
            check_num = None
        
        # Parse date
        date = self._parse_date(date_str, year)
        if not date:
            return None
        
        # Parse amount
        amount = self._parse_amount(amount_str)
        if amount is None:
            return None
        
        # Apply sign based on transaction type
        if trans_type == TransactionType.DEBIT:
            amount = -abs(amount)
        else:
            amount = abs(amount)
        
        return Transaction(
            date=date,
            description=self._clean_description(description),
            amount=amount,
            type=trans_type,
            bank_name=BankName.CHASE.value,
            category=category,
            raw_line=line,
            check_number=check_num
        )


# ============================
# BMO PARSER
# ============================

class BMOParser(BankStatementParser):
    """Parser for BMO Elite Business Checking statements"""
    
    def __init__(self):
        super().__init__(BankName.BMO)
    
    def parse(self, text: str) -> ParsedStatement:
        """Parse BMO statement"""
        statement = ParsedStatement(bank_name=BankName.BMO.value)
        lines = text.split('\n')
        
        in_transaction_section = False
        statement_year = datetime.now().year
        
        # Try to extract year
        year_match = re.search(r'20\d{2}', text[:1000])
        if year_match:
            statement_year = int(year_match.group(0))
        
        for line in lines:
            line_upper = line.upper().strip()
            
            # Detect transaction section
            if "MONTHLY ACTIVITY DETAILS" in line_upper:
                in_transaction_section = True
                continue
            
            # Skip header row
            if in_transaction_section and "DATE" in line_upper and "DESCRIPTION" in line_upper:
                continue
            
            # End of section
            if in_transaction_section and ("SUMMARY" in line_upper or "TOTAL" in line_upper):
                in_transaction_section = False
                continue
            
            # Parse transaction
            if in_transaction_section:
                transaction = self._parse_bmo_transaction(line, statement_year)
                if transaction:
                    statement.transactions.append(transaction)
        
        if not statement.transactions:
            statement.errors.append("no_transactions_parsed")
        
        return statement
    
    def _parse_bmo_transaction(self, line: str, year: int) -> Optional[Transaction]:
        """Parse BMO transaction line: Date  Description  Withdrawal  Deposit  Balance"""
        line = line.strip()
        if not line or len(line) < 10:
            return None
        
        # BMO format: MMM DD  Description  [Withdrawal]  [Deposit]  Balance
        # Split by multiple spaces to separate columns
        parts = re.split(r'\s{2,}', line)
        
        if len(parts) < 3:
            return None
        
        # First part should be date
        date_str = parts[0].strip()
        if not re.match(r'^[A-Z][a-z]{2}\s+\d{1,2}$', date_str):
            return None
        
        # Parse date
        date = self._parse_date(date_str, year)
        if not date:
            return None
        
        # Description is second part
        description = parts[1].strip()
        
        # Remaining parts are amounts (withdrawal, deposit, balance)
        # Need to figure out which is which
        amounts = []
        for i in range(2, len(parts)):
            amt = self._parse_amount(parts[i].strip())
            if amt is not None:
                amounts.append(amt)
        
        # If we have at least 2 amounts, last one is balance
        # Before that could be withdrawal or deposit
        if len(amounts) >= 2:
            balance = amounts[-1]
            # Check if we have both withdrawal and deposit, or just one
            if len(amounts) == 3:
                # Both withdrawal and deposit columns present
                withdrawal = amounts[0]
                deposit = amounts[1]
                if deposit > 0:
                    amount = deposit
                    trans_type = TransactionType.CREDIT
                else:
                    amount = -abs(withdrawal)
                    trans_type = TransactionType.DEBIT
            elif len(amounts) == 2:
                # Only one amount column (either withdrawal or deposit)
                transaction_amount = amounts[0]
                # Look at description or check balance change to determine type
                desc_lower = description.lower()
                if any(word in desc_lower for word in ['deposit', 'credit', 'payment received', 'transfer in', 'payroll']):
                    amount = abs(transaction_amount)
                    trans_type = TransactionType.CREDIT
                elif any(word in desc_lower for word in ['purchase', 'withdrawal', 'debit', 'fee', 'check', 'payment']):
                    amount = -abs(transaction_amount)
                    trans_type = TransactionType.DEBIT
                else:
                    # Default to credit if positive, debit if seems like expense
                    amount = abs(transaction_amount)
                    trans_type = TransactionType.CREDIT
            else:
                return None
        else:
            return None
        
        return Transaction(
            date=date,
            description=self._clean_description(description),
            amount=amount,
            type=trans_type,
            bank_name=BankName.BMO.value,
            running_balance=balance if len(amounts) >= 2 else None,
            raw_line=line
        )


# ============================
# BANK OF AMERICA PARSER
# ============================

class BoAParser(BankStatementParser):
    """Parser for Bank of America business checking statements"""
    
    def __init__(self):
        super().__init__(BankName.BOA)
    
    def parse(self, text: str) -> ParsedStatement:
        """Parse Bank of America statement"""
        statement = ParsedStatement(bank_name=BankName.BOA.value)
        lines = text.split('\n')
        
        current_section = None
        statement_year = datetime.now().year
        
        # Try to extract year
        year_match = re.search(r'20\d{2}', text[:1000])
        if year_match:
            statement_year = int(year_match.group(0))
        
        for line in lines:
            line_upper = line.upper().strip()
            
            # Detect sections
            if "DEPOSITS AND OTHER CREDITS" in line_upper:
                current_section = (TransactionCategory.DEPOSIT, TransactionType.CREDIT)
                continue
            elif "WITHDRAWALS AND OTHER DEBITS" in line_upper:
                current_section = (TransactionCategory.WITHDRAWAL, TransactionType.DEBIT)
                continue
            elif "CHECKS" in line_upper and current_section is None:
                current_section = (TransactionCategory.CHECK, TransactionType.DEBIT)
                continue
            elif "TOTAL" in line_upper or "ENDING BALANCE" in line_upper:
                current_section = None
                continue
            
            # Parse transaction
            if current_section:
                transaction = self._parse_boa_transaction(line, current_section, statement_year)
                if transaction:
                    statement.transactions.append(transaction)
        
        if not statement.transactions:
            statement.errors.append("no_transactions_parsed")
        
        return statement
    
    def _parse_boa_transaction(self, line: str, section_info: Tuple, year: int) -> Optional[Transaction]:
        """Parse BoA transaction line"""
        line = line.strip()
        if not line or len(line) < 10:
            return None
        
        # Skip headers
        if any(keyword in line.upper() for keyword in ['DATE', 'DESCRIPTION', 'AMOUNT', 'TOTAL']):
            return None
        
        category, trans_type = section_info
        
        # BoA format: MM/DD  Description  Amount
        match = re.match(r'^(\d{2}/\d{2})\s+(.+?)\s+([\d,]+\.\d{2})\s*$', line)
        if not match:
            # Try with check number: MM/DD  CheckNum  Description  Amount
            match = re.match(r'^(\d{2}/\d{2})\s+(\d+)\s+(.+?)\s+([\d,]+\.\d{2})\s*$', line)
            if match:
                date_str = match.group(1)
                check_num = match.group(2)
                description = match.group(3)
                amount_str = match.group(4)
            else:
                return None
        else:
            date_str = match.group(1)
            description = match.group(2)
            amount_str = match.group(3)
            check_num = None
        
        date = self._parse_date(date_str, year)
        if not date:
            return None
        
        amount = self._parse_amount(amount_str)
        if amount is None:
            return None
        
        # Apply sign
        if trans_type == TransactionType.DEBIT:
            amount = -abs(amount)
        else:
            amount = abs(amount)
        
        return Transaction(
            date=date,
            description=self._clean_description(description),
            amount=amount,
            type=trans_type,
            bank_name=BankName.BOA.value,
            category=category,
            raw_line=line,
            check_number=check_num
        )


# ============================
# FIFTH THIRD PARSER
# ============================

class FifthThirdParser(BankStatementParser):
    """Parser for Fifth Third Bank business statements"""
    
    def __init__(self):
        super().__init__(BankName.FIFTH_THIRD)
    
    def parse(self, text: str) -> ParsedStatement:
        """Parse Fifth Third statement"""
        statement = ParsedStatement(bank_name=BankName.FIFTH_THIRD.value)
        lines = text.split('\n')
        
        current_section = None
        statement_year = datetime.now().year
        
        year_match = re.search(r'20\d{2}', text[:1000])
        if year_match:
            statement_year = int(year_match.group(0))
        
        for line in lines:
            line_upper = line.upper().strip()
            
            if "WITHDRAWALS / DEBITS" in line_upper or "WITHDRAWALS/DEBITS" in line_upper:
                current_section = (TransactionCategory.WITHDRAWAL, TransactionType.DEBIT)
                continue
            elif "DEPOSITS / CREDITS" in line_upper or "DEPOSITS/CREDITS" in line_upper:
                current_section = (TransactionCategory.DEPOSIT, TransactionType.CREDIT)
                continue
            elif "TOTAL" in line_upper or "BALANCE" in line_upper:
                current_section = None
                continue
            
            if current_section:
                transaction = self._parse_fifth_third_transaction(line, current_section, statement_year)
                if transaction:
                    statement.transactions.append(transaction)
        
        if not statement.transactions:
            statement.errors.append("no_transactions_parsed")
        
        return statement
    
    def _parse_fifth_third_transaction(self, line: str, section_info: Tuple, year: int) -> Optional[Transaction]:
        """Parse Fifth Third transaction line"""
        line = line.strip()
        if not line or len(line) < 10:
            return None
        
        if any(keyword in line.upper() for keyword in ['DATE', 'DESCRIPTION', 'AMOUNT', 'TOTAL']):
            return None
        
        category, trans_type = section_info
        
        # Format: MM/DD  Description  Amount
        match = re.match(r'^(\d{2}/\d{2})\s+(.+?)\s+([\d,]+\.\d{2})\s*$', line)
        if not match:
            return None
        
        date_str = match.group(1)
        description = match.group(2)
        amount_str = match.group(3)
        
        date = self._parse_date(date_str, year)
        if not date:
            return None
        
        amount = self._parse_amount(amount_str)
        if amount is None:
            return None
        
        if trans_type == TransactionType.DEBIT:
            amount = -abs(amount)
        else:
            amount = abs(amount)
        
        return Transaction(
            date=date,
            description=self._clean_description(description),
            amount=amount,
            type=trans_type,
            bank_name=BankName.FIFTH_THIRD.value,
            category=category,
            raw_line=line
        )


# ============================
# US BANK PARSER
# ============================

class USBankParser(BankStatementParser):
    """Parser for US Bank Silver Business Checking statements"""
    
    def __init__(self):
        super().__init__(BankName.US_BANK)
    
    def parse(self, text: str) -> ParsedStatement:
        """Parse US Bank statement"""
        statement = ParsedStatement(bank_name=BankName.US_BANK.value)
        lines = text.split('\n')
        
        current_section = None
        statement_year = datetime.now().year
        
        year_match = re.search(r'20\d{2}', text[:1000])
        if year_match:
            statement_year = int(year_match.group(0))
        
        for line in lines:
            line_upper = line.upper().strip()
            
            if "OTHER DEPOSITS" in line_upper:
                current_section = (TransactionCategory.DEPOSIT, TransactionType.CREDIT)
                continue
            elif "OTHER WITHDRAWALS" in line_upper:
                current_section = (TransactionCategory.WITHDRAWAL, TransactionType.DEBIT)
                continue
            elif "TOTAL" in line_upper or "ENDING BALANCE" in line_upper:
                current_section = None
                continue
            
            if current_section:
                transaction = self._parse_usbank_transaction(line, current_section, statement_year)
                if transaction:
                    statement.transactions.append(transaction)
        
        if not statement.transactions:
            statement.errors.append("no_transactions_parsed")
        
        return statement
    
    def _parse_usbank_transaction(self, line: str, section_info: Tuple, year: int) -> Optional[Transaction]:
        """Parse US Bank transaction line"""
        line = line.strip()
        if not line or len(line) < 10:
            return None
        
        if any(keyword in line.upper() for keyword in ['DATE', 'DESCRIPTION', 'AMOUNT', 'TOTAL']):
            return None
        
        category, trans_type = section_info
        
        # Format: MM/DD  Description  Amount
        match = re.match(r'^(\d{2}/\d{2})\s+(.+?)\s+([\d,]+\.\d{2})\s*$', line)
        if not match:
            return None
        
        date_str = match.group(1)
        description = match.group(2)
        amount_str = match.group(3)
        
        date = self._parse_date(date_str, year)
        if not date:
            return None
        
        amount = self._parse_amount(amount_str)
        if amount is None:
            return None
        
        if trans_type == TransactionType.DEBIT:
            amount = -abs(amount)
        else:
            amount = abs(amount)
        
        return Transaction(
            date=date,
            description=self._clean_description(description),
            amount=amount,
            type=trans_type,
            bank_name=BankName.US_BANK.value,
            category=category,
            raw_line=line
        )


# ============================
# AMEX PARSER
# ============================

class AmexParser(BankStatementParser):
    """Parser for American Express credit card statements"""
    
    def __init__(self):
        super().__init__(BankName.AMEX)
    
    def parse(self, text: str) -> ParsedStatement:
        """Parse Amex statement"""
        statement = ParsedStatement(bank_name=BankName.AMEX.value, account_type="credit_card")
        lines = text.split('\n')
        
        current_section = None
        statement_year = datetime.now().year
        
        year_match = re.search(r'20\d{2}', text[:1000])
        if year_match:
            statement_year = int(year_match.group(0))
        
        for line in lines:
            line_upper = line.upper().strip()
            
            if "PAYMENTS AND CREDITS" in line_upper:
                current_section = (TransactionCategory.DEPOSIT, TransactionType.CREDIT)
                continue
            elif "PURCHASES" in line_upper and current_section is None:
                current_section = (TransactionCategory.CARD, TransactionType.DEBIT)
                continue
            elif "FEES" in line_upper and "INTEREST" not in line_upper:
                current_section = (TransactionCategory.FEE, TransactionType.DEBIT)
                continue
            elif "TOTAL" in line_upper or "NEW BALANCE" in line_upper:
                current_section = None
                continue
            
            if current_section:
                transaction = self._parse_amex_transaction(line, current_section, statement_year)
                if transaction:
                    statement.transactions.append(transaction)
        
        if not statement.transactions:
            statement.errors.append("no_transactions_parsed")
        
        return statement
    
    def _parse_amex_transaction(self, line: str, section_info: Tuple, year: int) -> Optional[Transaction]:
        """Parse Amex transaction line"""
        line = line.strip()
        if not line or len(line) < 10:
            return None
        
        if any(keyword in line.upper() for keyword in ['DATE', 'DESCRIPTION', 'AMOUNT', 'TOTAL']):
            return None
        
        category, trans_type = section_info
        
        # Format: MM/DD  Description  Amount
        match = re.match(r'^(\d{2}/\d{2})\s+(.+?)\s+([\d,]+\.\d{2})\s*$', line)
        if not match:
            return None
        
        date_str = match.group(1)
        description = match.group(2)
        amount_str = match.group(3)
        
        date = self._parse_date(date_str, year)
        if not date:
            return None
        
        amount = self._parse_amount(amount_str)
        if amount is None:
            return None
        
        # For credit cards, charges are negative (debits), payments are positive (credits)
        if trans_type == TransactionType.DEBIT:
            amount = -abs(amount)
        else:
            amount = abs(amount)
        
        return Transaction(
            date=date,
            description=self._clean_description(description),
            amount=amount,
            type=trans_type,
            bank_name=BankName.AMEX.value,
            category=category,
            raw_line=line
        )


# ============================
# PARSER FACTORY
# ============================

BANK_PARSERS: Dict[BankName, BankStatementParser] = {
    BankName.CHASE: ChaseParser(),
    BankName.BMO: BMOParser(),
    BankName.BOA: BoAParser(),
    BankName.FIFTH_THIRD: FifthThirdParser(),
    BankName.US_BANK: USBankParser(),
    BankName.AMEX: AmexParser(),
}


def get_parser(bank_name: BankName) -> Optional[BankStatementParser]:
    """Get parser for specified bank"""
    return BANK_PARSERS.get(bank_name)


# ============================
# MAIN ENTRY POINT
# ============================

def parse_bank_statement(text: str, bank_name: Optional[BankName] = None) -> ParsedStatement:
    """
    Main entry point for parsing bank statements.
    
    Args:
        text: Full text extracted from PDF statement
        bank_name: Optional bank identifier. If None, will auto-detect.
    
    Returns:
        ParsedStatement with transactions and metadata
    """
    # Auto-detect bank if not specified
    if bank_name is None:
        bank_name = detect_bank(text)
        logger.info(f"Detected bank: {bank_name.value}")
    
    # Get appropriate parser
    parser = get_parser(bank_name)
    
    if parser is None:
        logger.error(f"No parser available for bank: {bank_name.value}")
        return ParsedStatement(
            bank_name=bank_name.value,
            errors=["no_parser_available"]
        )
    
    # Parse statement
    try:
        statement = parser.parse(text)
        logger.info(f"Parsed {len(statement.transactions)} transactions from {bank_name.value} statement")
        return statement
    except Exception as e:
        logger.error(f"Error parsing {bank_name.value} statement: {e}")
        return ParsedStatement(
            bank_name=bank_name.value,
            errors=[f"parse_error: {str(e)}"]
        )


# ============================
# LEGACY COMPATIBILITY
# ============================

class LegacyBankStatementParser:
    """
    Legacy parser interface for backward compatibility.
    Wraps new multi-bank parser to maintain old API.
    """
    
    def parse_statement(self, lines: List[str]) -> Tuple[List[Dict], Dict[str, any]]:
        """
        Legacy parsing function that returns old format.
        
        Returns:
            Tuple of (transactions_list, metadata_dict)
        """
        text = '\n'.join(lines)
        
        # Parse with new system
        parsed = parse_bank_statement(text)
        
        # Convert to old format
        legacy_transactions = []
        for trans in parsed.transactions:
            legacy_transactions.append({
                'date': trans.date.strftime('%Y-%m-%d') if trans.date else None,
                'description': trans.description,
                'amount': trans.amount,
                'transaction_type': 'deposit' if trans.amount > 0 else 'withdrawal',
                'raw_line': trans.raw_line,
                'needs_review': trans.needs_review
            })
        
        metadata = {
            'total_lines': len(lines),
            'total_transactions': len(legacy_transactions),
            'bank_name': parsed.bank_name
        }
        
        return legacy_transactions, metadata
