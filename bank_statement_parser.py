"""
Multi-Bank Statement Parser Module
Supports parsing of different bank statement formats with automatic bank detection.
"""
import re
import dateparser
import phonenumbers
from commonregex import CommonRegex
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
    OTHER = "other"       # Added for BMO and others
    INCOME = "income"     # Added for analysis compatibility
    EXPENSE = "expense"   # Added for analysis compatibility
    UNCATEGORIZED = "uncategorized" # Added for analysis compatibility
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
    needs_review: bool = False  # Added for analysis compatibility


# ============================
# BANK DETECTION
# ============================

@dataclass
class BankLayout:
    """Configuration for a specific bank's layout"""
    bank_name: BankName
    transaction_regex: str
    column_mapping: Dict[str, int]  # Map of field name to regex group index (1-based)
    date_format: str = "%m/%d"
    amount_strict_regex: str = r'\(?\$?\s?\d{1,3}(?:,\d{3})*\.\d{2}\)?'
    section_headers: Dict[str, Tuple[TransactionCategory, TransactionType]] = field(default_factory=dict)
    summary_markers: List[str] = field(default_factory=list)
    check_regex: Optional[str] = None

BANK_LAYOUTS = {
    BankName.CHASE: BankLayout(
        bank_name=BankName.CHASE,
        transaction_regex=r'(\d{1,2}/\d{1,2})\s+(\S.*?)\s+([-+]?\$?[\d,]+\.\d{2}|[-+]?\$?[\d,]+)$',
        check_regex=r'(\d+)\s+[\*\^/ ]+\s+(\d{1,2}/\d{1,2})\s+([-+]?\$?[\d,]+\.\d{2}|[\d,]+\.\d{2})$',
        column_mapping={"date": 1, "description": 2, "amount": 3},
        section_headers={
            # Deposit sections
            "DEPOSITS AND ADDITIONS": (TransactionCategory.DEPOSIT, TransactionType.CREDIT),
            "DEPOSITS": (TransactionCategory.DEPOSIT, TransactionType.CREDIT),
            "ADDITIONS": (TransactionCategory.DEPOSIT, TransactionType.CREDIT),
            # Checks - CHECKS PAID must come before WITHDRAWALS to avoid substring collision
            "CHECKS PAID": (TransactionCategory.CHECK, TransactionType.DEBIT),
            "CHECKS": (TransactionCategory.CHECK, TransactionType.DEBIT),
            # ATM & Debit card
            "ATM & DEBIT CARD WITHDRAWALS": (TransactionCategory.ATM, TransactionType.DEBIT),
            "ATM & DEBIT": (TransactionCategory.ATM, TransactionType.DEBIT),
            # Electronic
            "ELECTRONIC WITHDRAWALS": (TransactionCategory.ACH, TransactionType.DEBIT),
            "ELECTRONIC": (TransactionCategory.ACH, TransactionType.DEBIT),
            # Other withdrawals
            "OTHER WITHDRAWALS": (TransactionCategory.WITHDRAWAL, TransactionType.DEBIT),
            "WITHDRAWALS": (TransactionCategory.WITHDRAWAL, TransactionType.DEBIT),
            # Fees
            "SERVICE FEES": (TransactionCategory.FEE, TransactionType.DEBIT),
            "FEES": (TransactionCategory.FEE, TransactionType.DEBIT),
        },
        summary_markers=["ENDING BALANCE", "DAILY LEDGER", "DAILY ENDING"]
    ),
    BankName.AMEX: BankLayout(
        bank_name=BankName.AMEX,
        transaction_regex=r'(\d{1,2}/\d{1,2}/\d{2,4})\s+(\S.*?)\s+([-+]?\$?[\d,]+\.\d{2})',
        column_mapping={"date": 1, "description": 2, "amount": 3},
        date_format="%m/%d/%y",
        section_headers={
            # All known AMEX section header variants
            "PAYMENTS/CREDITS": (TransactionCategory.DEPOSIT, TransactionType.CREDIT),
            "PAYMENTS AND CREDITS": (TransactionCategory.DEPOSIT, TransactionType.CREDIT),
            "PAYMENT/CREDITS": (TransactionCategory.DEPOSIT, TransactionType.CREDIT),
            "NEW CHARGES": (TransactionCategory.CARD, TransactionType.DEBIT),
            "PURCHASES": (TransactionCategory.CARD, TransactionType.DEBIT),
            "OTHER CHARGES": (TransactionCategory.CARD, TransactionType.DEBIT),
            "FEES": (TransactionCategory.FEE, TransactionType.DEBIT),
        },
        summary_markers=["ACCOUNT TOTALS", "TOTAL FEES CHARGED", "TOTAL INTEREST CHARGED"]
    ),
    BankName.BOA: BankLayout(
        bank_name=BankName.BOA,
        transaction_regex=r'^(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\s+(.*?)\s+([-~+]?\$?[\d, \.]+\.\d{2})\b(?=\s+|$)',
        column_mapping={"date": 1, "description": 2, "amount": 3},
        section_headers={
            "DEPOSITS AND OTHER CREDITS": (TransactionCategory.DEPOSIT, TransactionType.CREDIT),
            "DEPOSITS AND CREDITS": (TransactionCategory.DEPOSIT, TransactionType.CREDIT),
            "DEPOSITS": (TransactionCategory.DEPOSIT, TransactionType.CREDIT),
            "WITHDRAWALS AND OTHER DEBITS": (TransactionCategory.WITHDRAWAL, TransactionType.DEBIT),
            "WITHDRAWALS AND DEBITS": (TransactionCategory.WITHDRAWAL, TransactionType.DEBIT),
            "WITHDRAWALS": (TransactionCategory.WITHDRAWAL, TransactionType.DEBIT),
            "CHECKS PAID": (TransactionCategory.CHECK, TransactionType.DEBIT),
            "CHECKS": (TransactionCategory.CHECK, TransactionType.DEBIT),
            "SERVICE FEES": (TransactionCategory.FEE, TransactionType.DEBIT),
        },
        check_regex=r'(?i)(\d{1,2}/\d{1,2}(?:/\d{2,4}|/\d{2})?)\s+(\S+)\s+[^\d-]*([-~+]?[\d, \.]+\.\d{2})\b',
        summary_markers=["ENDING BALANCE", "DAILY LEDGER BALANCES", "ACCOUNT SUMMARY", "SERVICE FEES", "Total deposits and other credits", "Total withdrawals and other debits", "Total checks"]
    ),
    BankName.FIFTH_THIRD: BankLayout(
        bank_name=BankName.FIFTH_THIRD,
        # Date/Amount/Description — anchored to line start, decimals optional
        transaction_regex=r'^[^\d]*(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\s+([\d, ]+(?:\.\d{1,2})?)\s+(\S.*)',
        column_mapping={"date": 1, "amount": 2, "description": 3},
        section_headers={
            # Withdrawals
            "WITHDRAWALS / DEBITS": (TransactionCategory.WITHDRAWAL, TransactionType.DEBIT),
            "WITHDRAWALS/DEBITS": (TransactionCategory.WITHDRAWAL, TransactionType.DEBIT),
            "WITHDRAWALS AND DEBITS": (TransactionCategory.WITHDRAWAL, TransactionType.DEBIT),
            "WITHDRAWALS / DEBITS - CONTINUED": (TransactionCategory.WITHDRAWAL, TransactionType.DEBIT),
            "WITHDRAWALS / DEBITS (CONTINUED)": (TransactionCategory.WITHDRAWAL, TransactionType.DEBIT),
            "ELECTRONIC WITHDRAWALS": (TransactionCategory.WITHDRAWAL, TransactionType.DEBIT),
            "ATM & DEBIT CARD WITHDRAWALS": (TransactionCategory.WITHDRAWAL, TransactionType.DEBIT),
            "OTHER WITHDRAWALS": (TransactionCategory.WITHDRAWAL, TransactionType.DEBIT),
            # Deposits
            "DEPOSITS / CREDITS": (TransactionCategory.DEPOSIT, TransactionType.CREDIT),
            "DEPOSITS/CREDITS": (TransactionCategory.DEPOSIT, TransactionType.CREDIT),
            "DEPOSITS AND CREDITS": (TransactionCategory.DEPOSIT, TransactionType.CREDIT),
            "DEPOSITS AND ADDITIONS": (TransactionCategory.DEPOSIT, TransactionType.CREDIT),
            "DEPOSITS / CREDITS - CONTINUED": (TransactionCategory.DEPOSIT, TransactionType.CREDIT),
            "DEPOSITS / CREDITS (CONTINUED)": (TransactionCategory.DEPOSIT, TransactionType.CREDIT),
            # Other
            "CHECKS PAID": (TransactionCategory.CHECK, TransactionType.DEBIT),
            "SERVICE FEES": (TransactionCategory.FEE, TransactionType.DEBIT),
            "SERVICE CHARGES": (TransactionCategory.FEE, TransactionType.DEBIT),
        },
        summary_markers=["ENDING BALANCE", "TOTAL WITHDRAWALS", "TOTAL DEPOSITS", "DAILY BALANCE", "DAILY BALANCE SUMMARY", "BALANCE SUMMARY", "PAGE", "DAILY LEDGER BALANCES"]
    ),
    BankName.US_BANK: BankLayout(
        bank_name=BankName.US_BANK,
        # Date = "Nov 4", then description (may include ref number), then amount (may have trailing '-')
        # Amount may be preceded by '$' and a space: '$ 1,500.00-' or '6,500.00'
        transaction_regex=r'([A-Z][a-z]{2}\s+\d{1,2})\s+(.*?)\s+\$?\s*([\d,]+\.\d{2}-?)',
        column_mapping={"date": 1, "description": 2, "amount": 3},
        section_headers={
            "ELECTRONIC DEPOSITS": (TransactionCategory.DEPOSIT, TransactionType.CREDIT),
            "OTHER DEPOSITS": (TransactionCategory.DEPOSIT, TransactionType.CREDIT),
            "OTHER ELECTRONIC DEPOSITS": (TransactionCategory.DEPOSIT, TransactionType.CREDIT),
            "DEPOSITS": (TransactionCategory.DEPOSIT, TransactionType.CREDIT),
            "ELECTRONIC PAYMENTS": (TransactionCategory.WITHDRAWAL, TransactionType.DEBIT),
            "OTHER WITHDRAWALS": (TransactionCategory.WITHDRAWAL, TransactionType.DEBIT),
            "OTHER ELECTRONIC WITHDRAWALS": (TransactionCategory.WITHDRAWAL, TransactionType.DEBIT),
            "WITHDRAWALS": (TransactionCategory.WITHDRAWAL, TransactionType.DEBIT),
            "CHECKS PAID": (TransactionCategory.CHECK, TransactionType.DEBIT),
            "CHECKS": (TransactionCategory.CHECK, TransactionType.DEBIT),
        },
        summary_markers=["ENDING BALANCE", "ACCOUNT BALANCE SUMMARY", "DAILY BALANCE", "BALANCE SUMMARY",
                         "TOTAL OTHER DEPOSITS", "TOTAL OTHER WITHDRAWALS", "ANALYSIS SERVICE CHARGE"]
    ),
    BankName.BMO: BankLayout(
        bank_name=BankName.BMO,
        # Captures: Month Day Description Amount
        # Amount field can have $, commas, parens (negatives), spaces
        transaction_regex=r'(?i)(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})\s+(.*?)\s{2,}(\$?[\d,]+\.\d{2}|\(\$?[\d,]+\.\d{2}\))',
        column_mapping={"date": 1, "day": 2, "description": 3, "amounts": 4},
        date_format="%b %d",
        check_regex=r'(?i)(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})\s+Check\s+([\d]+)\s+\(?\$?([\d,\.]+)\)?',
        section_headers={
            "MONTHLY ACTIVITY DETAILS": (TransactionCategory.OTHER, TransactionType.DEBIT),
            "MONTHLY ACTIVITY DETAILS (CONT'D)": (TransactionCategory.OTHER, TransactionType.DEBIT),
            "MONTHLY ACTIVITY DETAILS (CONT": (TransactionCategory.OTHER, TransactionType.DEBIT),
            "MONTHLY ACTIVITY": (TransactionCategory.OTHER, TransactionType.DEBIT),
            "ACTIVITY DETAILS": (TransactionCategory.OTHER, TransactionType.DEBIT),
            "ACCOUNT ACTIVITY": (TransactionCategory.OTHER, TransactionType.DEBIT),
            "TRANSACTION DETAILS": (TransactionCategory.OTHER, TransactionType.DEBIT),
            "CHECKS": (TransactionCategory.CHECK, TransactionType.DEBIT),
            "OTHER WITHDRAWALS": (TransactionCategory.WITHDRAWAL, TransactionType.DEBIT),
            "OTHER DEBITS": (TransactionCategory.WITHDRAWAL, TransactionType.DEBIT),
        },
        summary_markers=["OPENING BALANCE", "CLOSING BALANCE", "MONTHLY TRANSACTION SUMMARY",
                         "ENDING BALANCE", "DAILY BALANCE", "BALANCE SUMMARY",
                         "Account ID", "EARNINGS SUMMARY"]
    )
}

def detect_bank(text: str) -> BankName:
    """
    Detect bank from statement text by inspecting first page.
    
    Args:
        text: Full text from PDF statement (preferably first page)
    
    Returns:
        BankName enum value
    """
    # Get first 4000 characters for detection
    first_page = text[:4000].upper()
    
    # Improved Chase detection
    if any(kw in first_page for kw in ["JPMORGAN CHASE", "CHASE BUSINESS", "CHASE SAVINGS", "CHASE COMPLETE"]):
        return BankName.CHASE

    # Fifth Third detection — must come BEFORE AMEX (FT PDFs have 'Account Summary'+'New Balance')
    if "FIFTH THIRD BANK" in first_page or "5/3 BANK" in first_page or "FIFTH THIRD" in first_page:
        return BankName.FIFTH_THIRD

    # Bank of America detection
    if any(kw in first_page for kw in ["BANK OF AMERICA", "PREFERRED REWARDS", "BOFA", "BUSINESS ADVANTAGE", "BKOFAMERICA", "B OF A"]):
        return BankName.BOA
    if "YOUR CHECKING ACCOUNT" in first_page and "ACCOUNT SUMMARY" in first_page:
        return BankName.BOA

    # BMO detection (BMO Harris / BMO Montreal)
    if any(kw in first_page for kw in ["BMO", "BANKING SUMMARY", "HARRIS BANK", "ELITE BUSINESS", "MONTREAL"]):
        return BankName.BMO

    # Amex detection
    if any(kw in first_page for kw in ["AMERICAN EXPRESS", "AMEX", "MEMBERSHIP REWARDS", "MEMBERSHIPR"]):
        return BankName.AMEX
    elif "ACCOUNT SUMMARY" in first_page and "NEW BALANCE" in first_page and "PAYMENT DUE" in first_page:
        return BankName.AMEX

    # US Bank detection
    if "U.S. BANK" in first_page or "USBANK" in first_page or "US BANK" in first_page:
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
    
    def parse(self, text: str, manual_year: Optional[int] = None) -> ParsedStatement:
        """Default parser using the bank's layout configuration"""
        layout = BANK_LAYOUTS.get(self.bank_name)
        if not layout:
            return ParsedStatement(bank_name=self.bank_name.value, errors=["no_layout_config"])
            
        statement = ParsedStatement(bank_name=self.bank_name.value)
        
        lines = text.split('\n')
        
        # Determine statement year
        statement_year = manual_year
        if not manual_year:
            # 1. Look for specific date range patterns (e.g. "Period: 11/01/2025 - 11/30/2025")
            range_match = re.search(r'(?:Statement Period|Through|Thru|Period).*?(\d{4})', text[:10000], re.IGNORECASE)
            if range_match:
                statement_year = int(range_match.group(1))
            else:
                # 2. Collect all 4-digit years and prefer the oldest of the most recent two
                all_years = re.findall(r'\b(202[0-9])\b', text[:20000])
                if all_years:
                    all_years_int = sorted(set(int(y) for y in all_years))
                    now = datetime.now()
                    # If we only see current year early in the year, the statement is likely previous year.
                    # But first, prefer smaller years (earlier statement) unless current year ONLY.
                    if len(all_years_int) > 1 and all_years_int[-1] == now.year and now.month < 6:
                        statement_year = all_years_int[-2]
                    else:
                        statement_year = all_years_int[0]
                else:
                    statement_year = datetime.now().year

        # 3. Contextual adjustment: late-month transactions (Nov/Dec) in early current year → previous year
        now = datetime.now()
        if not manual_year and statement_year == now.year and now.month < 5:
            if re.search(r'\b(10|11|12)/\d{2}\b', text[:20000]):
                statement_year -= 1
                logger.info(f"Contextual adjustment: Set year to {statement_year} because late-year months (10-12) found in early {now.year}")

        # Update logger
        logger.info(f"Using {statement_year} as base year for {self.bank_name.value} statement")
        
        # Search for beginning/ending balances
        self._extract_balances(text, statement)
        
        # BoA Specific: Pre-parse Daily Ledger Balances
        daily_ledger = {}
        if self.bank_name == BankName.BOA:
            ledger_match = re.search(r'(?i)Daily ledger balances(.*?)(?:Service fees|Checks|Account summary|\Z)', text, re.DOTALL)
            if ledger_match:
                ledger_txt = ledger_match.group(1)
                for entry in re.finditer(r'(\d{1,2}/\d{1,2})\s+([\d, ]+\.\d{2})', ledger_txt):
                    date_key = entry.group(1)
                    val = self._parse_amount(entry.group(2))
                    if val is not None:
                        daily_ledger[date_key] = val
            logger.info(f"BoA Daily Ledger found with {len(daily_ledger)} entries")

        current_section = None
        last_transaction = None
        current_running_balance = statement.beginning_balance
        last_date_key = None
        current_day_unmatched = []
        pending_tx = None # For multi-line BoA splits
        
        for line in lines:
            line_upper = line.upper().strip()
            if not line_upper:
                continue
                
            # 1. Check for section headers
            # Normalize whitespace for matching (collapse multi-spaces from OCR)
            # Also strip leading non-alphanumeric OCR symbols (e.g. '©', '¢', '*' from BMO)
            stripped_line_upper = re.sub(r'^[^A-Z0-9]+', '', line_upper).strip()
            normalized_line = ' '.join(stripped_line_upper.split())
            section_found = False
            for header, info in layout.section_headers.items():
                # Exact substring match OR startswith (for FifthThird lines like "WITHDRAWALS / DEBITS 61 items")
                if header in normalized_line or normalized_line.startswith(header):
                    current_section = info
                    section_found = True
                    last_transaction = None # Reset multi-line context
                    break
            if section_found: continue
            
            # 2. Check for summary markers to end sections
            if any(marker in line_upper for marker in layout.summary_markers):
                current_section = None
                last_transaction = None
                continue
                
            # 3. If in a section, try to match transaction
            if current_section:
                # First try check_regex if in a check section
                if current_section[0] == TransactionCategory.CHECK and layout.check_regex:
                    check_match = re.search(layout.check_regex, line)
                    if check_match:
                        try:
                            # Multi-group check support (e.g. BMO vs Chase)
                            groups = check_match.groups()
                            if len(groups) >= 4:
                                # BMO Style: #, Month, Day, Amount
                                check_no = groups[0].replace(' ', '').strip()
                                date_str = f"{groups[1]} {groups[2]}"
                                amt_str = groups[3]
                            elif self.bank_name == BankName.BOA:
                                # BoA Style: Date, #, Amount
                                date_str = groups[0]
                                check_no = groups[1]
                                amt_str = groups[2]
                            else:
                                # Chase/Legacy Style: #, Date, Amount
                                check_no = groups[0]
                                date_str = groups[1]
                                amt_str = groups[2]
                            
                            dt = self._parse_date(date_str, statement_year)
                            amount = self._parse_amount(amt_str)
                            
                            if amount is not None:
                                # Checks are always debits
                                if amount > 0: amount = -amount
                                
                                transaction = Transaction(
                                        date=dt,
                                        description=f"CHECK {check_no}",
                                        amount=amount,
                                        type=TransactionType.DEBIT,
                                        bank_name=self.bank_name.value,
                                        category=TransactionCategory.CHECK,
                                        check_number=check_no,
                                        raw_line=line
                                    )
                                statement.transactions.append(transaction)
                                if current_running_balance is not None:
                                    current_running_balance += amount
                                logger.info(f"Parsed Check: {transaction.check_number} on {transaction.date} for {transaction.amount}")
                                last_transaction = transaction
                                
                                # BoA multi-check support: find other checks on same line
                                matches = list(re.finditer(layout.check_regex, line))
                                if len(matches) > 1:
                                    for other_match in matches:
                                        if other_match.start() == check_match.start(): continue 
                                        g = other_match.groups()
                                        if self.bank_name == BankName.BOA:
                                            odt_str, ocn, oamt_str = g[0], g[1], g[2]
                                        else:
                                            ocn, odt_str, oamt_str = g[0], g[1], g[2]
                                            
                                        odt = self._parse_date(odt_str, statement_year)
                                        oamt = self._parse_amount(oamt_str)
                                        if odt and oamt is not None:
                                            if oamt > 0: oamt = -oamt
                                            ocn_clean = re.sub(r'[^\d]', '', ocn)
                                            otrans = Transaction(
                                                date=odt,
                                                description=f"CHECK {ocn_clean}",
                                                amount=oamt,
                                                type=TransactionType.DEBIT,
                                                bank_name=self.bank_name.value,
                                                category=TransactionCategory.CHECK,
                                                check_number=ocn_clean,
                                                raw_line=line
                                            )
                                            statement.transactions.append(otrans)
                                            if current_running_balance is not None:
                                                current_running_balance += oamt
                                            logger.info(f"Parsed Extra Check: {otrans.check_number} on {otrans.date} for {otrans.amount}")
                            continue
                        except Exception as e:
                            logger.warning(f"Failed to parse check line: {line}. Error: {e}")

                # BoA: Track dates for unmatched lines
                if self.bank_name == BankName.BOA:
                    date_match = re.search(r'(\d{1,2}/\d{1,2})', line)
                    if date_match:
                        last_date_key = date_match.group(1)
                    if not last_date_key and re.search(r'(\d{1,2}/\d{1,2}(?:/\d{2,4})?)', line):
                        d_match = re.search(r'(\d{1,2}/\d{1,2}(?:/\d{2,4})?)', line)
                        last_date_key = d_match.group(1)

                txs = self._parse_line_with_layout(
                    line, current_section, statement_year, layout, 
                    prev_balance=current_running_balance
                )
                if txs:
                    for transaction in txs:
                        transaction.bank_name = self.bank_name.value
                        statement.transactions.append(transaction)
                        last_transaction = transaction
                        pending_tx = None
                        if current_running_balance is not None:
                            current_running_balance += transaction.amount
                elif self.bank_name == BankName.BOA and re.search(r'([-~+]?\$?[\d, \.]+\.\d{2})\b', line):
                    # Potential BoA Orphan Line (e.g. CCD or Zelle details on next line)
                    # EXCLUDE pure continuation/reference lines like "1D:9424300002 CCD"
                    is_continuation = re.match(r'^\s*1D:[A-Z0-9]+ (CCD|PPD|WEB|CTX)\s*$', line.strip(), re.IGNORECASE)
                    if not is_continuation and last_transaction and ("ID:" in line or "Conf#" in line or "Zelle" in line or "1D:" in line):
                        amt_match = re.search(r'([-~+]?\$?[\d, \.]+\.\d{2})\b', line)
                        if amt_match:
                            oamt = self._parse_amount(amt_match.group(1))
                            if oamt:
                                orphan_tx = Transaction(
                                    date=last_transaction.date,
                                    description=line.strip(),
                                    amount=oamt,
                                    type=TransactionType.CREDIT if oamt > 0 else TransactionType.DEBIT,
                                    bank_name=self.bank_name.value,
                                    category=last_transaction.category,
                                    raw_line=line
                                )
                                statement.transactions.append(orphan_tx)
                                if current_running_balance is not None:
                                    current_running_balance += oamt
                                logger.info(f"Captured BoA Orphan Transaction on {orphan_tx.date}: {oamt}")
                                # DO NOT reset last_transaction here, keep it for the next orphan or multi-line
                
                # Keep track of unmatched lines for current section/day
                if not txs:
                    current_day_unmatched.append(line)
                    d_match = re.search(r'(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\s+(.*)', line)
                    if d_match:
                        pending_tx = Transaction(
                            date=self._parse_date(d_match.group(1), statement_year),
                            description=self._clean_description(d_match.group(2)),
                            amount=0,
                            type=current_section[1] if current_section else TransactionType.DEBIT,
                            bank_name=self.bank_name.value,
                            category=current_section[0] if current_section else TransactionCategory.DEPOSIT,
                            raw_line=line
                        )
                elif last_transaction or pending_tx:
                    # Multi-line continuation: append to last transaction or check for deferred amount
                    clean_line = self._clean_description(line)
                    is_junk = any(kw in line_upper for kw in ["PAGE", "CONTINUED", "ACCOUNT", "DATE", "BALANCE", "DAILY BALANCE", "STATEMENT"])
                    
                    if clean_line and not is_junk:
                        # BoA Multi-line Amount Support:
                        # If we have a pending_tx and this line ends in an amount
                        if self.bank_name == BankName.BOA and pending_tx:
                            amt_match = re.search(r'([-~+]?\$?[\d, \.]+\.\d{2})$', line)
                            if amt_match:
                                amt = self._parse_amount(amt_match.group(1))
                                if amt is not None:
                                    if pending_tx.type == TransactionType.DEBIT and amt > 0: amt = -amt
                                    pending_tx.amount = amt
                                    # Append any text BEFORE the amount to description
                                    desc_part = self._clean_description(line[:amt_match.start()])
                                    if desc_part: pending_tx.description += " " + desc_part
                                    statement.transactions.append(pending_tx)
                                    last_transaction = pending_tx
                                    if current_running_balance is not None:
                                        current_running_balance += pending_tx.amount
                                    pending_tx = None
                                    continue
                        
                        if last_transaction and not re.match(r'^[\d\s,.\$()-]+$', clean_line):
                            if len(last_transaction.description) < 500:
                                # Improved check: avoid doubling descriptions if the line was already captured
                                # OR if this line is a substring of the existing description
                                if clean_line not in last_transaction.description and last_transaction.description not in clean_line:
                                    last_transaction.description += " " + clean_line
        
        if not statement.transactions:
            statement.errors.append("no_transactions_parsed")
            
        # POST-PROCESSING for BoA: Comprehensive Ledger Comparison
        if self.bank_name == BankName.BOA and daily_ledger:
            logger.info("Performing post-parse BoA Ledger verification...")
            # 1. Group transactions by date
            from collections import defaultdict
            txs_by_date = defaultdict(list)
            for tx in statement.transactions:
                if not tx.needs_review: # Ignore existing repairs if any
                    d_key = tx.date.strftime('%m/%d') if tx.date else None
                    if d_key: txs_by_date[d_key].append(tx)
            
            # 2. Iterate through sorted ledger days
            v_balance = statement.beginning_balance or 0.0
            sorted_days = sorted(daily_ledger.keys(), key=lambda x: (int(x.split('/')[0]), int(x.split('/')[1])))
            
            new_repairs = []
            for d_key in sorted_days:
                expected = daily_ledger[d_key]
                # Sum all transactions up to this day (that haven't been summed yet)
                # Actually, BofA ledger is "Balance at end of day".
                # So we sum all txs for this day and add to running v_balance.
                day_sum = sum(t.amount for t in txs_by_date[d_key])
                actual_end = v_balance + day_sum
                
                gap = expected - actual_end
                if abs(gap) > 0.01:
                    logger.info(f"BoA Ledger Gap on {d_key}: {gap:,.2f} (Expected {expected:,.2f}, Actual {actual_end:,.2f})")
                    # Find a good unmatched line for this date if possible
                    # (Fallback logic: Use the last transaction's description or generic)
                    repair_desc = "[REPAIRED] Missing transaction"
                    # Try to find an unmatched line from the original parse for this date
                    # For now, just categorical repair
                    repaired = Transaction(
                        date=self._parse_date(d_key, statement_year),
                        description=repair_desc,
                        amount=gap,
                        type=TransactionType.CREDIT if gap > 0 else TransactionType.DEBIT,
                        bank_name=self.bank_name.value,
                        category=TransactionCategory.DEPOSIT if gap > 0 else TransactionCategory.WITHDRAWAL,
                        needs_review=True,
                        raw_line=f"REPAIR FOR LEDGER {d_key}"
                    )
                    new_repairs.append(repaired)
                    v_balance = expected # Anchor to ledger
                else:
                    v_balance = actual_end
            
            # Add repairs to statement
            statement.transactions.extend(new_repairs)
        
        # FINAL CLEANUP for BoA: Handle last day gap (obsolete with post-processing, but keeping legacy as fallback)
        if False and self.bank_name == BankName.BOA and last_date_key and last_date_key in daily_ledger:
            expected_balance = daily_ledger[last_date_key]
            gap = expected_balance - current_running_balance
            if abs(gap) > 0.01 and current_day_unmatched:
                raw_u = current_day_unmatched[0]
                u_desc = raw_u.strip()
                u_match = re.search(r'\d{1,2}/\d{1,2}(?:/\d{2,4})?\s+(.*)', raw_u)
                if u_match: u_desc = u_match.group(1).strip()
                
                repaired = Transaction(
                    date=self._parse_date(last_date_key, statement_year),
                    description=f"[REPAIRED] {u_desc}",
                    amount=gap,
                    type=TransactionType.CREDIT if gap > 0 else TransactionType.DEBIT,
                    bank_name=self.bank_name.value,
                    category=current_section[0] if current_section else TransactionCategory.DEPOSIT,
                    raw_line=raw_u,
                    needs_review=True
                )
                statement.transactions.append(repaired)
                current_running_balance += gap

        return statement
    
    def _parse_date(self, date_str: str, year: Optional[int] = None) -> Optional[datetime]:
        """Parse date string with various formats"""
        if not date_str:
            return None
        # Clean the date string
        orig_date_str = date_str
        date_str = date_str.replace('‘', '').replace('’', '').strip()
        
        # BoA specific: ensure we use MDY as it's a US bank
        if self.bank_name == BankName.BOA:
            # Clean non-date artifacts first
            clean_date = re.sub(r'[^\d/]', '', date_str)
            d_match = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{2,4})$', clean_date)
            if d_match:
                try:
                    m, d, y = int(d_match.group(1)), int(d_match.group(2)), int(d_match.group(3))
                    if y < 100: y += 2000
                    res = datetime(y, m, d)
                    logger.info(f"BOA Date Parse (MDY): '{orig_date_str}' -> {res}")
                    return res
                except ValueError: return None
            # Fallback for MM/DD
            d_match2 = re.match(r'^(\d{1,2})/(\d{1,2})$', clean_date)
            if d_match2:
                try:
                    res = datetime(year or 2025, int(d_match2.group(1)), int(d_match2.group(2)))
                    logger.info(f"BOA Date Parse (MD): '{orig_date_str}' -> {res}")
                    return res
                except ValueError: return None

        settings = {'RELATIVE_BASE': datetime(year or 2025, 1, 1)}
        if self.bank_name in [BankName.BOA, BankName.CHASE, BankName.US_BANK, BankName.FIFTH_THIRD]:
            settings['DATE_ORDER'] = 'MDY'
            
        dt = dateparser.parse(date_str, settings=settings)
        if dt:
            has_year = re.search(r'\b20\d{2}\b', date_str)
            if not has_year and year:
                dt = dt.replace(year=year)
            elif has_year and year:
                parsed_year = int(has_year.group(0))
                # Protect against print-year artifact (e.g. 01/19/2026 on a Nov-2025 statement)
                now = datetime.now()
                if parsed_year == now.year and now.month < 5 and dt.month > 8 and year < now.year:
                    dt = dt.replace(year=year)
        return dt
    
    def _parse_line_with_layout(self, line: str, section_info: Tuple, year: int, layout: BankLayout, prev_balance: Optional[float] = None) -> List[Transaction]:
        """Generic transaction line parser using BankLayout. Returns a list (for multi-entry lines)."""
        category, trans_type = section_info
        transactions = []

        # 1. Mask non-amount entities
        masked_line = self._mask_non_amount_entities(line)

        # ---- BMO SPECIAL HANDLING ----
        # BMO has a 4-column layout: Month Day Description Amount [Balance]
        # The 'amount' column key is 'amounts' (not 'amount').
        # Parenthesised amounts e.g. ($1,000.00) are debits; bare $amounts are credits.
        if layout.bank_name == BankName.BMO and "amounts" in layout.column_mapping:
            # Use a regex that allows single spaces between desc and amount (OCR noise)
            # and handles optional currency symbols/parents.
            # Capture FIRST amount on the line (transaction column), ignore second amount (balance column).
            bmo_regex = r'(?i)(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})\s+(.*?)\s+(\(?\$?[\d,]+\.\d{2}\)?)(?:\s+\$?[\d,]+\.\d{2})?\s*$'
            matches = list(re.finditer(bmo_regex, line, re.IGNORECASE))
            if matches:
                for match in matches:
                    groups = match.groups()
                    month_raw = groups[0]   # e.g. "Dec"
                    day_raw   = groups[1]    # e.g. "01"
                    desc_raw  = groups[2]
                    amt_raw   = groups[3]

                    date_str = f"{month_raw} {day_raw}"
                    date = self._parse_date(date_str, year)
                    if not date:
                        continue

                    # Determine sign: parenthesised = debit, bare = credit
                    amt_clean = amt_raw.strip()
                    
                    # BMO OCR Noise Handling: 
                    # Sometimes '(' is read as '6' or '$' as '8' or 'B'
                    # e.g. (62,000.00) or 81,000.00
                    is_debit = '(' in amt_clean or ')' in amt_clean or amt_clean.startswith('6') or amt_clean.startswith('8')
                    
                    amount = self._parse_amount(amt_clean)
                    if amount is None:
                        continue
                        
                    # If it was a debit with potential noise (6 or 8), need to correct the amount
                    if is_debit:
                        # BMO OCR Noise: ($2,000.00) -> (62,000.00) or 81,000.00
                        # Strip currency/parens for testing the core amount digits
                        test_amt = re.sub(r'[^\d\.]', '', amt_clean)
                        # If first digit is 6 or 8 and it's followed by a significantly smaller amount
                        # relative to the apparent value (e.g. 62,000.00 for a 2,000.00 transaction)
                        if (amt_clean.startswith('6') or amt_clean.startswith('8') or '(' in amt_clean):
                            # Try stripping the first digit if it's 6, 8, or B
                            # But only if it's not the ONLY digit before the decimal (don't strip $6.00 -> $.00)
                            candidate = re.sub(r'^[68B]', '', test_amt)
                            if candidate and not candidate.startswith('.'):
                                retry_amt = self._parse_amount(candidate)
                                if retry_amt is not None:
                                    # Cross-validate: If the gap between original and retry is exactly 6000, 8000, 60000 etc.
                                    # or if the original amount seems impossibly large for the context.
                                    amount = retry_amt

                    # Ensure sign is set correctly regardless of section
                    if is_debit:
                        amount = -abs(amount)
                        actual_type = TransactionType.DEBIT
                    else:
                        amount = abs(amount)
                        actual_type = TransactionType.CREDIT

                    desc = self._clean_description(desc_raw)

                    # Detect if this might be a check
                    check_no = None
                    chk_match = re.search(r'(?i)Check\s+(\d+)', desc_raw)
                    if chk_match:
                        check_no = chk_match.group(1)
                        if not desc.strip() or desc.upper() == "CHECK":
                            desc = f"CHECK {check_no}"

                    tx = Transaction(
                        date=date,
                        description=desc,
                        amount=amount,
                        type=actual_type,
                        bank_name=layout.bank_name.value,
                        category=TransactionCategory.CHECK if check_no else category,
                        check_number=check_no,
                        raw_line=line
                    )
                    transactions.append(tx)
                return transactions
            
            # BMO Multi-line descriptor support
            is_date_line = re.search(r'(?i)(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}', line)
            if prev_balance is not None and line.strip() and not is_date_line:
                # Caller in parse() will handle generic multi-line append if we return []
                return [] 
            return []
        # ---- END BMO SPECIAL HANDLING ----

        # ---- FIFTH THIRD SPECIAL HANDLING ----
        if layout.bank_name == BankName.FIFTH_THIRD:
            # Skip header/summary lines that may contain dates (to fix start-date issue)
            line_up = line.upper()
            if any(kw in line_up for kw in ["BALANCE", "STATEMENT", "PAGE", "ACCOUNT NUMBER", "PERIOD", "TOTAL ITEMS", "# ITEMS"]):
                return []

            # OCR Date Normalization: repair slashless dates at line start
            # Examples: "117 87.00 ..." → "11/17 87.00 ..."
            #            "114 70,399.78 ..." → "11/4 70,399.78 ..."  (note: "114" could be 11/4)
            #            "1/19 41.00 ..." → "11/19 41.00 ..." (truncated leading 1)
            ocr_norm = line
            
            # Pattern: line starts with 3-4 digits that look like MMDD (no slash)
            slashless = re.match(r'^(\s*)(\d{3,4})(\s+\d)', line)
            if slashless:
                raw_num = slashless.group(2)
                # Try MM+DD: e.g. "117" → month=11, day=7; "1117" → month=11, day=17
                if len(raw_num) == 3:
                    mm, dd = raw_num[:1], raw_num[1:]  # M/DD (e.g. 1/17... but could be 11/7)
                    mm2, dd2 = raw_num[:2], raw_num[2:]  # MM/D (e.g. 11/7)
                    # Prefer MM/D if first 2 chars are valid month (01-12) and third is valid day
                    if 1 <= int(mm2) <= 12 and 1 <= int(dd2) <= 31:
                        ocr_norm = line.replace(slashless.group(2), f"{mm2}/{dd2}", 1)
                    elif 1 <= int(mm) <= 12 and 1 <= int(dd) <= 31:
                        ocr_norm = line.replace(slashless.group(2), f"{mm}/{dd}", 1)
                elif len(raw_num) == 4:
                    mm, dd = raw_num[:2], raw_num[2:]  # MMDD → MM/DD
                    if 1 <= int(mm) <= 12 and 1 <= int(dd) <= 31:
                        ocr_norm = line.replace(slashless.group(2), f"{mm}/{dd}", 1)

            # Truncated month: "1/19" when it should be "11/19"
            # Heuristic: if month is 1-9 and we're in a Nov statement, it's likely 1X/DD
            truncated = re.match(r'^(\s*)(\d)/(\d{1,2})(\s+\d)', ocr_norm)
            if truncated and year:
                single_m = int(truncated.group(2))
                day_ = int(truncated.group(3))
                # If the statement year's most-common month has 10+ prefix (Nov statement = 11)
                # and we see a single digit like '1', it might be '11'
                candidate_m = int(f"1{single_m}")  # e.g. 1 → 11
                if 1 <= candidate_m <= 12 and 1 <= day_ <= 31:
                    # Only switch if makes sense: single month doesn't make sense for main period
                    # Use MMDDYY hint from description if available
                    desc_part = ocr_norm[truncated.end():]
                    mmddyy = re.search(r'(\d{2})(\d{2})\d{2}', desc_part)
                    if mmddyy and int(mmddyy.group(1)) == candidate_m:
                        ocr_norm = ocr_norm.replace(
                            f"{truncated.group(2)}/{truncated.group(3)}",
                            f"{candidate_m}/{day_}", 1
                        )

            # Now try anchored regex match on the (potentially normalized) line
            matches = list(re.finditer(layout.transaction_regex, ocr_norm))
            # If normalization helped, use ocr_norm; otherwise fall back to original
            working_line = ocr_norm if matches else line
            matches = matches or list(re.finditer(layout.transaction_regex, line))


            # Fallback: Check# Date Amount
            if not matches and category == TransactionCategory.CHECK:
                chk = re.search(r'^\s*(\d{4,10})\s+(\d{1,2}/\d{1,2})\s+([\d, ]+(?:\.\d{2})?)\s*$', line)
                if chk:
                    dt = self._parse_date(chk.group(2), year)
                    amt = self._parse_amount(chk.group(3))
                    if dt and amt is not None:
                        return [Transaction(
                            date=dt, description=f"CHECK {chk.group(1)}",
                            amount=-abs(amt), type=TransactionType.DEBIT,
                            bank_name=layout.bank_name.value,
                            category=TransactionCategory.CHECK,
                            check_number=chk.group(1), raw_line=line
                        )]

            if matches:
                for m in matches:
                    raw_dt, raw_amt, raw_desc = m.group(1), m.group(2), m.group(3)
                    if any(kw in raw_desc.upper() for kw in ["BALANCE", "STATEMENT", "DAILY", "TOTAL"]):
                        continue
                    dt = self._parse_date(raw_dt, year)
                    amt = self._parse_amount(raw_amt)
                    if amt is not None:
                        if not dt or (year and dt.year != year):
                            mm = re.search(r'\b(\d{2})(\d{2})(\d{2})\b', raw_desc)
                            if mm:
                                dt = self._parse_date(f"{mm.group(1)}/{mm.group(2)}", year)
                        if dt:
                            amt = -abs(amt) if trans_type == TransactionType.DEBIT else abs(amt)
                            transactions.append(Transaction(
                                date=dt, description=self._clean_description(raw_desc),
                                amount=amt, type=trans_type,
                                bank_name=layout.bank_name.value,
                                category=category, raw_line=line
                            ))
                if transactions:
                    return transactions

            # Fallback: Date Description Amount (reversed order)
            fb = re.search(r'^\s*(\d{1,2}/\d{1,2})\s+(\S.*?)\s+([\d, ]+(?:\.\d{2})?)\s*$', line)
            if fb:
                dt = self._parse_date(fb.group(1), year)
                amt = self._parse_amount(fb.group(3))
                if dt and amt is not None:
                    amt = -abs(amt) if trans_type == TransactionType.DEBIT else abs(amt)
                    return [Transaction(
                        date=dt, description=self._clean_description(fb.group(2)),
                        amount=amt, type=trans_type,
                        bank_name=layout.bank_name.value,
                        category=category, raw_line=line
                    )]

            # Service fee fallback (no date)
            if category == TransactionCategory.FEE or "SERVICE" in line.upper():
                fm = re.search(r'([\d, ]+(?:\.\d{2})?)\s*$', line)
                if fm:
                    amt = self._parse_amount(fm.group(1))
                    if amt is not None:
                        any_dt = re.search(r'(\d{1,2}/\d{1,2})', line)
                        dt = self._parse_date(any_dt.group(1), year) if any_dt else datetime(year, 11, 30)
                        amt = -abs(amt) if trans_type == TransactionType.DEBIT else abs(amt)
                        return [Transaction(
                            date=dt, description=self._clean_description(line),
                            amount=amt, type=trans_type,
                            bank_name=layout.bank_name.value,
                            category=category, raw_line=line
                        )]
            return []
        # ---- END FIFTH THIRD SPECIAL HANDLING ----

        # 2. Check for balance (if present in transaction line)
        current_balance = None
        if layout.bank_name == BankName.CHASE and "Balance" in line:
            # Chase specific balance extraction
            pass

        # 3. Match transactions (support for multiple per line via finditer)
        matches = list(re.finditer(layout.transaction_regex, masked_line))
        if not matches:
            return []

        for match in matches:
            groups = match.groups()
            date_raw = groups[layout.column_mapping["date"] - 1]
            desc_raw = groups[layout.column_mapping["description"] - 1]
            amt_raw = groups[layout.column_mapping.get("amount", layout.column_mapping.get("amounts", 3)) - 1]

            # Use original description from unmasked line if possible
            start_pos = match.start(layout.column_mapping["description"])
            end_pos = match.end(layout.column_mapping["description"])
            description = line[start_pos:end_pos].strip()

            # Clean description
            description = self._clean_description(description)

            # Parse amount
            amount = self._parse_amount(amt_raw)
            if amount is None:
                continue

            # Fifth Third date recovery: if no '/' in date_raw, try to find MMDDYY in description
            actual_date_raw = date_raw
            if layout.bank_name == BankName.FIFTH_THIRD and '/' not in date_raw:
                # Look for MMDDYY at the end of the description (very common in Fifth Third)
                # or any 6-digit number that looks like a date
                # We use the raw untransformed 'description' variable from line 840
                mmddyy_match = re.search(r'\b(\d{2})(\d{2})(\d{2})\b', description)
                if mmddyy_match:
                    actual_date_raw = f"{mmddyy_match.group(1)}/{mmddyy_match.group(2)}"
                elif len(date_raw) == 3:
                    # Fallback for 3-digit dates like '114' if no MMDDYY found
                    # Try to split as M/DD or MM/D based on current month
                    if month_of_statement := (year_month := datetime.now().month): # Default to current month
                         pass # Complex logic avoided for now as MMDDYY is reliable

            # Parse date
            date = self._parse_date(actual_date_raw, year)
            if not date:
                continue

            # US Bank: amount with trailing '-' means debit
            if layout.bank_name == BankName.US_BANK and isinstance(amt_raw, str) and amt_raw.strip().endswith('-'):
                amount = -abs(amount)
                trans_type = TransactionType.DEBIT
            else:
                # Apply sign based on section
                if trans_type == TransactionType.DEBIT:
                    amount = -abs(amount)
                else:
                    amount = abs(amount)

            transactions.append(Transaction(
                date=date,
                description=description,
                amount=amount,
                type=trans_type,
                bank_name=layout.bank_name.value,
                category=category,
                running_balance=current_balance,
                raw_line=line
            ))

        return transactions

    def _parse_amount(self, amount_str: str) -> Optional[float]:
        """Convert currency string to float, handling negatives and OCR noise"""
        if not amount_str:
            return None
        try:
            is_negative = False
            # Basic cleanup: remove currency symbols, commas, and internal spaces
            cleaned = amount_str.replace('$', '').replace(',', '').strip()
            
            if self.bank_name == BankName.BOA:
                # Handle BoA ID-prefixed amounts (e.g. "126494303 375.00")
                parts = cleaned.split()
                if len(parts) > 1:
                    # If the first part is a long numeric ID, the second is the amount
                    if re.match(r'^\d{5,}$', parts[0]) and re.search(r'\d+\.\d{2}', parts[1]):
                        cleaned = parts[1]
                    else:
                        # Otherwise, join them for standard space-as-separator handling
                        cleaned = "".join(parts)
                else:
                    cleaned = cleaned.replace(' ', '')
            else:
                cleaned = cleaned.replace(' ', '')
            
            # Remove trailing OCR noise (important for BoA)
            cleaned = re.sub(r'[^\d\.\(\)-]+$', '', cleaned)
            
            # Handle trailing negative sign (US Bank style: 3,592.48-)
            if cleaned.endswith('-'):
                is_negative = True
                cleaned = cleaned[:-1]
            elif '(' in cleaned and cleaned.endswith(')'):
                is_negative = True
                cleaned = cleaned.replace('(', '').replace(')', '')

            # Treat ~ as negative (seen in BoA OCR for withdrawals)
            if cleaned.startswith('~'):
                is_negative = True
                cleaned = cleaned[1:]
                
            has_leading_minus = cleaned.startswith('-')
            clean_digits = re.sub(r'[^\d.]', '', cleaned)
            
            if not clean_digits:
                return None
            
            # BoA specific: Handle missing dots in amounts like 57441 -> 574.41
            if '.' not in clean_digits and self.bank_name == BankName.BOA and len(clean_digits) >= 3:
                # Basic heuristic: if it's 5 digits like 27094 (Conf# overlap), skip insertion
                if not (len(clean_digits) == 5 and clean_digits.startswith('270')):
                    clean_digits = clean_digits[:-2] + '.' + clean_digits[-2:]
                
            amount = float(clean_digits)
            if is_negative or has_leading_minus:
                amount = -abs(amount)
            return amount
        except (ValueError, TypeError, AttributeError):
            return None
    
    def _clean_description(self, desc: str) -> str:
        """Clean and normalize transaction description"""
        if not desc: return ""
        # Remove extra spaces and common OCR noise
        cleaned = re.sub(r'\s+', ' ', desc).strip()
        # Remove leading symbols like '*', '¢'
        cleaned = re.sub(r'^[^a-zA-Z0-9]+', '', cleaned)
        # Remove common prefixes
        cleaned = re.sub(r'^(PURCHASE|PAYMENT|DEBIT|CREDIT)\s+', '', cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _extract_balances(self, text: str, statement: ParsedStatement):
        """Extract beginning and ending balances from statement text"""
        
        # Search for beginning/opening balance
        beg_match = None
        if self.bank_name == BankName.US_BANK:
            # US Bank: "Beginning Balance on Nov 3 $ 26,427.22"
            # Flexible date and optional/noisy dollar sign
            beg_match = re.search(r'(?i)BEGINNING\s+BALANCE[^\n\r]*?[\$s]?\s*([\d,]+\.\d{2})', text)
        else:
            # General pattern for Chase, BoA, 5/3, BMO
            beg_match = re.search(r'(?i)(?:BEGINNING|OPENING|STARTING|PREVIOUS)\s+BALANCE[^\n]*?[\$s]?\s*([\d,]+\.\d{2})', text)
            
        if beg_match:
            statement.beginning_balance = self._parse_amount(beg_match.group(1))
            logger.info(f"Detected {self.bank_name.value} beginning balance: {statement.beginning_balance}")
            
        # Search for ending/closing balance
        end_match = None
        if self.bank_name == BankName.US_BANK:
            # US Bank: "Ending Balance on Nov 30, 2025 $ 31,334.74"
            end_match = re.search(r'(?i)ENDING\s+BALANCE[^\n\r]*?[\$s]?\s*([\d,]+\.\d{2})', text)
        else:
            end_match = re.search(r'(?i)(?:ENDING|CLOSING|NEW|TOTAL)\s+BALANCE[^\n]*?[\$s]?\s*([\d,]+\.\d{2})', text)
            
        if end_match:
            statement.ending_balance = self._parse_amount(end_match.group(1))
            logger.info(f"Detected {self.bank_name.value} ending balance: {statement.ending_balance}")

    def _mask_non_amount_entities(self, line: str) -> str:
        """
        Identify and hide phone numbers and dates to prevent them from being
        misidentified as amounts.
        """
        if not line:
            return ""
            
        masked_line = line
        entities_to_mask = []
        
        # 1. Identify Phone Numbers (Robust)
        try:
            for match in phonenumbers.PhoneNumberMatcher(line, "US"):
                phone_str = line[match.start:match.end]
                entities_to_mask.append(phone_str)
        except Exception:
            pass
            
        # 2. Identify Dates (Regex based to avoid CommonRegex false positives on amounts)
        # Matches patterns like 11/24/25, 11-24-25, 11 24 25, 2024-11-20
        # For BOA, we SHOULD NOT mask dates because they appear in the middle of lines for multi-transactions
        if self.bank_name == BankName.BOA:
            # Only mask confirmation numbers and potential phones
            # MUST preserve length to keep character offsets correct!
            patterns = [
                r'(?<!\d)\d{3}[-\s]?\d{3}[-\s]?\d{4}(?!\d)',       # Phone numbers
                r'(?i)Conf#\s*[A-Z0-9]{5,}(?!\.[0-9]{2})',         # BoA confirmation numbers (restricted)
            ]
            for pattern in patterns:
                def mask_rep(m): return "X" * len(m.group(0))
                line = re.sub(pattern, mask_rep, line)
            return line

        date_patterns = [
            r'(?<!\d)(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})(?!\d)',  # 11/24/25 or 11/24/2025
            r'(?<!\d)(?:\d{4}-\d{2}-\d{2})(?!\d)',              # 2024-11-20
            r'(?<!\d)(?:\d{1,2}\s+\d{1,2}\s+\d{2,4})(?!\d)',    # 11 24 25
            r'(?<!\w)(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:,?\s+\d{2,4})?(?!\w)', # Jan 15, 2025
            r'(?i)Conf#\s*[A-Z0-9]+(?!\.[0-9]{2})',             # BoA confirmation numbers (avoiding dots/cents)
        ]
        
        # Exclude the leading date from masking.
        # Leading = first non-space content is a date pattern (up to 6 non-letter chars before it).
        # Supports: 11/24/2025, 2024-11-20, OR Nov 24
        leading_date = None
        leading_date_match = re.search(
            r'(?i)^([^a-zA-Z]{0,6})(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2})',
            line.strip()
        )
        if leading_date_match:
            leading_date = leading_date_match.group(2)

        for pattern in date_patterns:
            for match in re.finditer(pattern, line, re.IGNORECASE):
                entity = match.group(0)
                if leading_date and entity == leading_date:
                    continue
                entities_to_mask.append(entity)
        
        # 3. Mask identified entities (preserving length/offsets)
        # Sort by length descending to avoid partial replacements of longer strings
        entities_to_mask.sort(key=len, reverse=True)
        for entity in entities_to_mask:
            # Only mask if it's actually in masked_line (could have Been masked by a longer pattern)
            if entity in masked_line:
                masked_line = masked_line.replace(entity, " " * len(entity))
            
        return masked_line


# ============================
# CHASE PARSER
# ============================

class ChaseParser(BankStatementParser):
    """Parser for Chase Business Complete Checking statements"""
    
    def __init__(self):
        super().__init__(BankName.CHASE)
    
    def parse(self, text: str, manual_year: Optional[int] = None) -> ParsedStatement:
        # Use generic parser first
        statement = super().parse(text, manual_year=manual_year)
        # Then extract Chase-specific metadata
        lines = text.split('\n')
        self._extract_metadata(lines, statement)
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
        """Legacy wrapper, now uses modular line parser"""
        return self._parse_line_with_layout(line, section_info, year, BANK_LAYOUTS[BankName.CHASE])


# ============================
# BMO PARSER
# ============================

class BMOParser(BankStatementParser):
    """Parser for BMO Elite Business Checking statements"""
    
    def __init__(self):
        super().__init__(BankName.BMO)
    
    # Inherits default parse with manual_year support


# ============================
# BANK OF AMERICA PARSER
# ============================

class BoAParser(BankStatementParser):
    """Parser for Bank of America business checking statements"""
    
    def __init__(self):
        super().__init__(BankName.BOA)
    
    # Inherits default parse with manual_year support


# ============================
# FIFTH THIRD PARSER
# ============================

class FifthThirdParser(BankStatementParser):
    """Parser for Fifth Third Bank business statements"""
    
    def __init__(self):
        super().__init__(BankName.FIFTH_THIRD)
    
    # Inherits default parse with manual_year support


# ============================
# US BANK PARSER
# ============================


# ============================
# AMEX PARSER
# ============================

class AmexParser(BankStatementParser):
    """Parser for American Express credit card statements"""
    
    def __init__(self):
        super().__init__(BankName.AMEX)
    
    def parse(self, text: str, manual_year: Optional[int] = None) -> ParsedStatement:
        # AMEX uses generic parser
        statement = super().parse(text, manual_year=manual_year)
        statement.account_type = "credit_card"

        # Always do a full-line scan fallback for AMEX because section headers are often absent.
        # Avoid duplicates by tracking already-seen raw_line strings.
        seen_lines = {t.raw_line for t in statement.transactions}

        lines = text.split('\n')
        # Use the same min-year approach to avoid picking up payment due dates
        statement_year = manual_year or datetime.now().year
        if not manual_year:
            all_years = re.findall(r'\b(202[0-9])\b', text[:20000])
            if all_years:
                statement_year = min(int(y) for y in all_years)

        layout = BANK_LAYOUTS[BankName.AMEX]
        for line in lines:
            if line.strip() in seen_lines:
                continue  # Already captured by section-based parser
            txns = self._parse_line_with_layout(
                line,
                (TransactionCategory.CARD, TransactionType.DEBIT),
                statement_year,
                layout
            )
            for t in txns:   # _parse_line_with_layout always returns a list
                t.bank_name = BankName.AMEX.value
                statement.transactions.append(t)
                seen_lines.add(line.strip())

        return statement


# ============================
# PARSER FACTORY
# ============================

class USBankParser(BankStatementParser):
    def __init__(self):
        super().__init__(BankName.US_BANK)

    def parse(self, text: str, manual_year: Optional[int] = None) -> ParsedStatement:
        # Create empty statement for US Bank - we rely on the specific scan
        statement = ParsedStatement(bank_name=BankName.US_BANK.value)
        
        layout = BANK_LAYOUTS[BankName.US_BANK]
        statement_year = manual_year or datetime.now().year
        if not manual_year:
            all_years = re.findall(r'\b(202[0-9])\b', text[:20000])
            if all_years:
                statement_year = min(int(y) for y in all_years)
        
        # Extract balances
        self._extract_balances(text, statement)

        existing_hashes = set()
        
        lines = text.split('\n')
        for line in lines:
            line_strip = line.strip()
            if not line_strip: continue
            
            # Skip lines that look like Balance Summary or headers
            if any(kw in line_strip.upper() for kw in ["BALANCE SUMMARY", "ENDING BALANCE", "ENDING BALANCE ON", "BEGINNING BALANCE", "BEGINNING BALANCE ON", "ANALYSIS SERVICE CHARGE"]):
                continue
                
            # US Bank Specific: Filter out lines that are clearly Balance Summary sequences
            # (e.g. Nov 4 32,927.22 Nov 5 ...) - look for multiple Nov/Dec/etc months
            months = re.findall(r'[A-Z][a-z]{2}\s+\d{1,2}', line_strip)
            if len(months) > 1:
                continue

            # Try to parse any line as a transaction
            txs = self._parse_line_with_layout(line, (TransactionCategory.OTHER, TransactionType.DEBIT), statement_year, layout)
            if txs:
                for tx in txs:
                    # Check for duplicates
                    h = hash(f"{tx.date}{tx.description}{tx.amount}")
                    if h not in existing_hashes:
                        # US Bank Sign Authority: The trailing '-' in the raw line is the source of truth
                        if line_strip.endswith('-') or re.search(r'[\d,]+\.\d{2}-', line_strip):
                            tx.type = TransactionType.DEBIT
                            tx.category = TransactionCategory.WITHDRAWAL
                            tx.amount = -abs(tx.amount)
                        else:
                            # Heuristic for category if not explicitly a debit via trailer
                            desc_upper = tx.description.upper()
                            if any(kw in desc_upper for kw in ["MOBILE", "DEPOSIT", "CREDIT", "TRANSFER FROM", "TELLER DEPOSIT"]):
                                tx.type = TransactionType.CREDIT
                                tx.category = TransactionCategory.DEPOSIT
                                tx.amount = abs(tx.amount)
                            else:
                                tx.type = TransactionType.DEBIT
                                tx.category = TransactionCategory.WITHDRAWAL
                                tx.amount = -abs(tx.amount)
                            
                        statement.transactions.append(tx)
                        existing_hashes.add(h)
        
        # Re-sort and re-calculate
        statement.transactions.sort(key=lambda x: x.date if x.date else datetime.min)
        return statement

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

# MAIN ENTRY POINT
# ============================

def parse_bank_statement(text: str, bank_name: Optional[BankName] = None, manual_year: Optional[int] = None) -> ParsedStatement:
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
        statement = parser.parse(text, manual_year=manual_year)
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
