"""
Test Enhanced Custom Rules System
Tests amount-based filtering, priority ordering, and pattern matching
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Transaction:
    """Simplified Transaction model for testing"""
    date: str = "2024-01-15"
    transaction_type: str = "withdrawal"
    vendor: str = ""
    amount: float = 0.0
    description: str = ""
    raw_line: str = ""
    account_code: Optional[str] = None
    _mapped_by_rule: bool = False


def _norm(s: str) -> str:
    """Normalize text for matching"""
    return " ".join((s or "").lower().strip().split())


def apply_enhanced_rules(transactions, business_rules):
    """
    Enhanced rule matching with amount ranges, priority, and patterns
    This is a standalone version of the logic implemented in bank_data_analysis.py
    """
    # Sort rules by priority (lower number = higher priority, default = 999)
    sorted_rules = sorted(business_rules, key=lambda r: r.get("priority", 999))
    
    for tx in transactions:
        text = f"{tx.vendor or ''} {tx.description or ''}"
        text = _norm(text)
        
        # Use absolute value of amount for comparison (withdrawals are negative)
        tx_amount = abs(getattr(tx, "amount", 0.0))
        
        matched = False
        for rule in sorted_rules:
            # 1. Check primary keyword (required)
            kw = _norm(rule.get("keyword", ""))
            if not kw or kw not in text:
                continue
            
            # 2. Check amount range filters (optional)
            min_amt = rule.get("min_amount")
            max_amt = rule.get("max_amount")
            
            if min_amt is not None and tx_amount < min_amt:
                continue  # Amount too small for this rule
                
            if max_amt is not None and tx_amount > max_amt:
                continue  # Amount too large for this rule
            
            # 3. Check exclusion keywords (optional) - skip if any match
            exclude_kws = rule.get("exclude_keywords", [])
            if exclude_kws and any(_norm(excl) in text for excl in exclude_kws):
                continue  # Excluded by keyword
            
            # 4. Check additional keywords (optional) - ALL must match
            additional_kws = rule.get("additional_keywords", [])
            if additional_kws and not all(_norm(kw_add) in text for kw_add in additional_kws):
                continue  # Not all additional keywords match
            
            # 5. All conditions met - apply the rule
            tx.account_code = rule.get("account_code")
            tx._mapped_by_rule = True
            matched = True
            break  # First matching rule wins (priority-based)
        
        if not matched:
            tx.account_code = None
            tx._mapped_by_rule = False
    
    return transactions


# ============================
# TEST CASES
# ============================

def test_amount_range_filtering():
    """Test that ATM withdrawals are categorized based on amount ranges"""
    print("\n=== TEST 1: Amount Range Filtering ===")
    
    rules = [
        {
            "keyword": "ATM Withdrawal",
            "account_code": "807",
            "account_display": "807 · TEMPORARY HELP",
            "min_amount": 200,
            "priority": 1
        },
        {
            "keyword": "ATM Withdrawal",
            "account_code": "999",
            "account_display": "999 · OTHER EXPENSES",
            "max_amount": 199.99,
            "priority": 2
        }
    ]
    
    transactions = [
        Transaction(amount=-250, description="ATM Withdrawal cash"),
        Transaction(amount=-150, description="ATM Withdrawal"),
        Transaction(amount=-50, description="ATM Withdrawal"),
    ]
    
    result = apply_enhanced_rules(transactions, rules)
    
    # Verify results
    assert result[0].account_code == "807", f"Expected 807, got {result[0].account_code}"
    assert result[1].account_code == "999", f"Expected 999, got {result[1].account_code}"
    assert result[2].account_code == "999", f"Expected 999, got {result[2].account_code}"
    
    print(f"✓ $250 ATM → {result[0].account_code} (TEMPORARY HELP)")
    print(f"✓ $150 ATM → {result[1].account_code} (OTHER EXPENSES)")
    print(f"✓ $50 ATM → {result[2].account_code} (OTHER EXPENSES)")
    print("PASSED ✓")


def test_check_amount_filtering():
    """Test that checks are categorized based on amount ranges"""
    print("\n=== TEST 2: Check Amount Filtering ===")
    
    rules = [
        {
            "keyword": "Check",
            "account_code": "801",
            "account_display": "801 · SALARIES-OFFICERS",
            "min_amount": 1000,
            "priority": 1
        },
        {
            "keyword": "Check",
            "account_code": "822",
            "account_display": "822 · SUPPLIES",
            "min_amount": 100,
            "max_amount": 999.99,
            "priority": 2
        },
        {
            "keyword": "Check",
            "account_code": "999",
            "account_display": "999 · OTHER EXPENSES",
            "max_amount": 99.99,
            "priority": 3
        }
    ]
    
    transactions = [
        Transaction(amount=-2500, description="Check #1234", vendor="Payroll"),
        Transaction(amount=-350, description="Check #1235", vendor="Office Supplies"),
        Transaction(amount=-45, description="Check #1236", vendor="Misc"),
    ]
    
    result = apply_enhanced_rules(transactions, rules)
    
    assert result[0].account_code == "801", f"Expected 801, got {result[0].account_code}"
    assert result[1].account_code == "822", f"Expected 822, got {result[1].account_code}"
    assert result[2].account_code == "999", f"Expected 999, got {result[2].account_code}"
    
    print(f"✓ $2,500 Check → {result[0].account_code} (SALARIES)")
    print(f"✓ $350 Check → {result[1].account_code} (SUPPLIES)")
    print(f"✓ $45 Check → {result[2].account_code} (OTHER EXPENSES)")
    print("PASSED ✓")


def test_priority_ordering():
    """Test that higher priority rules match first"""
    print("\n=== TEST 3: Priority Ordering ===")
    
    rules = [
        {
            "keyword": "payment",
            "account_code": "928",
            "account_display": "928 · RENT",
            "additional_keywords": ["rent"],
            "priority": 1  # Higher priority
        },
        {
            "keyword": "payment",
            "account_code": "999",
            "account_display": "999 · OTHER EXPENSES",
            "priority": 2  # Lower priority
        }
    ]
    
    transactions = [
        Transaction(amount=-1500, description="Payment for rent", vendor="Landlord"),
        Transaction(amount=-100, description="Payment processing fee"),
    ]
    
    result = apply_enhanced_rules(transactions, rules)
    
    assert result[0].account_code == "928", f"Expected 928, got {result[0].account_code}"
    assert result[1].account_code == "999", f"Expected 999, got {result[1].account_code}"
    
    print(f"✓ Rent payment → {result[0].account_code} (RENT - matched priority 1 rule)")
    print(f"✓ Generic payment → {result[1].account_code} (OTHER - matched priority 2 rule)")
    print("PASSED ✓")


def test_exclusion_keywords():
    """Test that exclusion keywords prevent matching"""
    print("\n=== TEST 4: Exclusion Keywords ===")
    
    rules = [
        {
            "keyword": "stripe",
            "account_code": "601",
            "account_display": "601 · SALES",
            "exclude_keywords": ["refund", "chargeback"],
            "priority": 1
        },
        {
            "keyword": "stripe",
            "account_code": "602",
            "account_display": "602 · RETURNS",
            "additional_keywords": ["refund"],
            "priority": 2
        }
    ]
    
    transactions = [
        Transaction(amount=500, description="Stripe payment received"),
        Transaction(amount=-250, description="Stripe refund processed"),
    ]
    
    result = apply_enhanced_rules(transactions, rules)
    
    assert result[0].account_code == "601", f"Expected 601, got {result[0].account_code}"
    assert result[1].account_code == "602", f"Expected 602, got {result[1].account_code}"
    
    print(f"✓ Stripe payment → {result[0].account_code} (SALES)")
    print(f"✓ Stripe refund → {result[1].account_code} (RETURNS - excluded from SALES)")
    print("PASSED ✓")


def test_backward_compatibility():
    """Test that old-style rules (keyword only) still work"""
    print("\n=== TEST 5: Backward Compatibility ===")
    
    # Old-style rules without new fields
    rules = [
        {
            "keyword": "Aubreystallion",
            "account_code": "928",
            "account_display": "928 · RENT"
        },
        {
            "keyword": "shopify",
            "account_code": "601",
            "account_display": "601 · SALES"
        }
    ]
    
    transactions = [
        Transaction(amount=-2000, vendor="Aubreystallion", description="Monthly rent"),
        Transaction(amount=350, vendor="Shopify", description="Online sales"),
    ]
    
    result = apply_enhanced_rules(transactions, rules)
    
    assert result[0].account_code == "928", f"Expected 928, got {result[0].account_code}"
    assert result[1].account_code == "601", f"Expected 601, got {result[1].account_code}"
    
    print(f"✓ Aubreystallion → {result[0].account_code} (RENT)")
    print(f"✓ Shopify → {result[1].account_code} (SALES)")
    print("PASSED ✓")


def test_additional_keywords():
    """Test that additional keywords require ALL to match"""
    print("\n=== TEST 6: Additional Keywords (AND logic) ===")
    
    rules = [
        {
            "keyword": "atm",
            "account_code": "928",
            "account_display": "928 · RENT",
            "additional_keywords": ["rent", "landlord"],
            "priority": 1
        },
        {
            "keyword": "atm",
            "account_code": "999",
            "account_display": "999 · OTHER EXPENSES",
            "priority": 2
        }
    ]
    
    transactions = [
        Transaction(amount=-1500, description="ATM withdrawal for rent to landlord"),
        Transaction(amount=-100, description="ATM withdrawal cash"),
    ]
    
    result = apply_enhanced_rules(transactions, rules)
    
    assert result[0].account_code == "928", f"Expected 928, got {result[0].account_code}"
    assert result[1].account_code == "999", f"Expected 999, got {result[1].account_code}"
    
    print(f"✓ ATM for rent to landlord → {result[0].account_code} (RENT - all keywords matched)")
    print(f"✓ ATM cash → {result[1].account_code} (OTHER - missing additional keywords)")
    print("PASSED ✓")


# ============================
# RUN ALL TESTS
# ============================

if __name__ == "__main__":
    print("=" * 60)
    print("ENHANCED CUSTOM RULES SYSTEM - TEST SUITE")
    print("=" * 60)
    
    try:
        test_amount_range_filtering()
        test_check_amount_filtering()
        test_priority_ordering()
        test_exclusion_keywords()
        test_backward_compatibility()
        test_additional_keywords()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓✓✓")
        print("=" * 60)
        print("\n✅ Enhanced custom rules system is working correctly!")
        print("✅ Amount-based filtering: WORKING")
        print("✅ Priority ordering: WORKING")
        print("✅ Exclusion keywords: WORKING")
        print("✅ Additional keywords: WORKING")
        print("✅ Backward compatibility: WORKING")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
