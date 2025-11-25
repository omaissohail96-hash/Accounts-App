"""
Bank Statement Parser Module
Handles parsing of different bank statement formats with flexible detection
"""
import re
import dateparser
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Transaction:
    """Represents a single transaction from a bank statement"""
    date: Optional[str]
    description: str
    amount: float
    transaction_type: str  # 'deposit' or 'withdrawal'
    vendor: Optional[str] = None
    category: Optional[str] = None
    needs_review: bool = False
    raw_line: str = ""
    line_number: Optional[int] = None


class BankStatementParser:
    """Parses bank statements from various banks with flexible format detection"""
    
    def __init__(self):
        # Common section headers that indicate deposits
        self.deposit_headers = [
            r'deposits?',
            r'credits?',
            r'amount\s+in',
            r'additions?',
            r'deposits?\s+and\s+credits?',
            r'money\s+in',
            r'income',
            r'received',
        ]
        
        # Common section headers that indicate withdrawals
        self.withdrawal_headers = [
            r'withdrawals?',
            r'debits?',
            r'amount\s+out',
            r'checks?\s+paid',
            r'atm\s+&\s+debit',
            r'debit\s+card',
            r'electronic\s+withdrawals?',
            r'fees?',
            r'service\s+charges?',
            r'money\s+out',
            r'payments?',
            r'purchases?',
        ]
        
        # Common date patterns
        self.date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY or MM-DD-YYYY
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',   # YYYY/MM/DD
            r'\d{1,2}\s+\w{3}\s+\d{2,4}',      # DD MMM YYYY
            r'\w{3}\s+\d{1,2},?\s+\d{2,4}',   # MMM DD, YYYY
        ]
        
        # Amount patterns (handles parentheses, negatives, currency symbols)
        self.amount_patterns = [
            r'[$€£¥]?\s*-?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?',  # Standard currency
            r'\(\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*\)',      # Parentheses for negative
        ]
    
    def detect_sections(self, lines: List[str]) -> Dict[str, List[int]]:
        """
        Detect deposit and withdrawal sections in the statement
        
        Returns:
            Dict with 'deposits' and 'withdrawals' keys containing line number ranges
        """
        sections = {
            'deposits': [],
            'withdrawals': [],
            'other': []
        }
        
        current_section = None
        section_start = None
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Check for deposit headers
            for pattern in self.deposit_headers:
                if re.search(pattern, line_lower):
                    if current_section and section_start is not None:
                        # Close previous section
                        if current_section in sections:
                            sections[current_section].append((section_start, i - 1))
                    current_section = 'deposits'
                    section_start = i
                    logger.info(f"Found deposit section at line {i}: {line[:50]}")
                    break
            
            # Check for withdrawal headers
            for pattern in self.withdrawal_headers:
                if re.search(pattern, line_lower):
                    if current_section and section_start is not None:
                        # Close previous section
                        if current_section in sections:
                            sections[current_section].append((section_start, i - 1))
                    current_section = 'withdrawals'
                    section_start = i
                    logger.info(f"Found withdrawal section at line {i}: {line[:50]}")
                    break
            
            # Check for section end indicators (summary lines, totals)
            if re.search(r'total|summary|ending\s+balance|beginning\s+balance', line_lower):
                if current_section and section_start is not None:
                    if current_section in sections:
                        sections[current_section].append((section_start, i))
                    current_section = None
                    section_start = None
        
        # Close last section if still open
        if current_section and section_start is not None:
            if current_section in sections:
                sections[current_section].append((section_start, len(lines) - 1))
        
        return sections
    
    def extract_amount(self, text: str) -> Optional[float]:
        """Extract monetary amount from text, handling various formats"""
        # Remove commas and try to find amount
        text_clean = text.replace(',', '')
        
        # Try parentheses format first (negative)
        paren_match = re.search(r'\(\s*(\d+(?:\.\d{2})?)\s*\)', text_clean)
        if paren_match:
            return -float(paren_match.group(1))
        
        # Try negative sign
        neg_match = re.search(r'-\s*(\d+(?:\.\d{2})?)', text_clean)
        if neg_match:
            return -float(neg_match.group(1))
        
        # Try standard positive amount
        pos_match = re.search(r'(\d+(?:\.\d{2})?)', text_clean)
        if pos_match:
            amount = float(pos_match.group(1))
            # Check if there's a negative indicator nearby
            text_before = text[:text.find(pos_match.group(1))].lower()
            if any(neg_word in text_before[-20:] for neg_word in ['debit', 'withdrawal', 'payment', 'charge', 'fee']):
                return -amount
            return amount
        
        return None
    
    def extract_date(self, text: str) -> Optional[str]:
        """Extract date from text using various patterns"""
        for pattern in self.date_patterns:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(0)
                try:
                    parsed_date = dateparser.parse(date_str)
                    if parsed_date:
                        return parsed_date.strftime('%Y-%m-%d')
                except:
                    pass
        return None
    
    def parse_transaction_line(self, line: str, transaction_type: str, line_num: int) -> Optional[Transaction]:
        """Parse a single transaction line"""
        if not line.strip():
            return None
        
        # Skip header lines, total lines, and summary lines
        line_lower = line.lower()
        header_keywords = ['date', 'description', 'amount']
        summary_keywords = [
            'total deposits', 'total additions', 'total credits', 'total withdrawals',
            'total debits', 'total checks', 'total atm', 'total fees', 'total charges',
            'total payments', 'total amount', 'total for this period', 'total ending balance',
            'ending balance', 'beginning balance', 'closing balance', 'average ledger balance',
            'average available balance', 'interest paid', 'summary'
        ]

        if any(keyword in line_lower for keyword in summary_keywords):
            return None

        if line_lower.strip().startswith('total '):
            tokens = line_lower.split()
            if len(tokens) >= 2:
                summary_tokens = {
                    'deposit', 'deposits', 'withdrawal', 'withdrawals', 'debit', 'debits',
                    'credit', 'credits', 'fees', 'charges', 'amount', 'payments', 'checks',
                    'atm', 'ach', 'fees', 'service', 'interest', 'balance', 'ending', 'beginning'
                }
                second = tokens[1].strip(',:')
                numeric = second.replace(',', '').replace('.', '').isdigit()
                if numeric or second in summary_tokens:
                    return None

        if any(word in line_lower for word in header_keywords):
            if 'date' in line_lower and 'description' in line_lower:
                return None  # Header row
        
        # Extract date
        date = self.extract_date(line)
        
        # Extract amount
        amount = self.extract_amount(line)
        if amount is None:
            # Can't find amount, mark for review
            return Transaction(
                date=date,
                description=line.strip(),
                amount=0.0,
                transaction_type=transaction_type,
                needs_review=True,
                raw_line=line,
                line_number=line_num
            )
        
        # Ensure withdrawals are negative
        if transaction_type == 'withdrawal' and amount > 0:
            amount = -abs(amount)
        elif transaction_type == 'deposit' and amount < 0:
            amount = abs(amount)
        
        # Extract description (everything except date and amount)
        description = line
        # Remove date from description
        if date:
            date_original = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}\s+\w{3}\s+\d{2,4}', description)
            if date_original:
                description = description.replace(date_original.group(0), '').strip()
        
        # Remove amount from description
        amount_patterns_clean = [
            r'\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?',
            r'-?\d{1,3}(?:,\d{3})*(?:\.\d{2})?',
            r'\(\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*\)',
        ]
        for pattern in amount_patterns_clean:
            description = re.sub(pattern, '', description).strip()
        
        description = ' '.join(description.split())  # Normalize whitespace
        
        if not description or len(description) < 3:
            needs_review = True
            description = line.strip()[:100]  # Use raw line if no description extracted
        else:
            needs_review = False
        
        return Transaction(
            date=date,
            description=description,
            amount=amount,
            transaction_type=transaction_type,
            raw_line=line,
            line_number=line_num,
            needs_review=needs_review
        )
    
    def parse_statement(self, lines: List[str]) -> Tuple[List[Transaction], Dict[str, any]]:
        """
        Main parsing function
        
        Returns:
            Tuple of (transactions_list, metadata_dict)
        """
        transactions = []
        metadata = {
            'total_lines': len(lines),
            'deposit_sections': 0,
            'withdrawal_sections': 0,
        }
        
        # Detect sections
        sections = self.detect_sections(lines)
        
        # Parse deposit sections
        for start, end in sections['deposits']:
            metadata['deposit_sections'] += 1
            for i in range(start + 1, min(end, len(lines))):  # Skip header line
                transaction = self.parse_transaction_line(lines[i], 'deposit', i)
                if transaction and abs(transaction.amount) > 0:
                    transactions.append(transaction)
        
        # Parse withdrawal sections
        for start, end in sections['withdrawals']:
            metadata['withdrawal_sections'] += 1
            for i in range(start + 1, min(end, len(lines))):  # Skip header line
                transaction = self.parse_transaction_line(lines[i], 'withdrawal', i)
                if transaction and abs(transaction.amount) > 0:
                    transactions.append(transaction)
        
        # If no sections detected, try to parse all lines (fallback)
        if not transactions:
            logger.warning("No sections detected, attempting to parse all lines...")
            for i, line in enumerate(lines):
                # Try to detect transaction by presence of amount
                amount = self.extract_amount(line)
                if amount and abs(amount) > 1.0:  # Minimum transaction amount
                    # Guess type based on sign or keywords
                    line_lower = line.lower()
                    if amount < 0 or any(word in line_lower for word in ['debit', 'withdrawal', 'payment', 'charge', 'fee', 'purchase']):
                        trans_type = 'withdrawal'
                    else:
                        trans_type = 'deposit'
                    
                    transaction = self.parse_transaction_line(line, trans_type, i)
                    if transaction:
                        transactions.append(transaction)
        
        metadata['total_transactions'] = len(transactions)
        return transactions, metadata

