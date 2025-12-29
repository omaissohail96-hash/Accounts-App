#!/usr/bin/env python3
"""
Test if 'Shopify ID' keyword is matching correctly
"""

from account_code_mapper import AccountCodeMapper

def test_shopify_variations():
    mapper = AccountCodeMapper()
    
    test_cases = [
        ("Shopify", "Payment from Shopify"),
        ("Shopify ID", "Shopify ID payment"),
        ("SHOPIFY", "SHOPIFY PAYMENT"),
        ("shopify", "shopify payout"),
        ("Shopify Id", "Payment received from Shopify Id"),
        ("Orig CO Name:Shopify", "Descr:Xxxxxxxxxxsec:Web"),
        ("Shopify ID", "Shopify ID - Monthly Subscription Fee"),
        ("Shopify ID", "Transfer Sec:Web Ind ID:Shopify Ind Name:Newengland Tack"),
    ]
    
    print(f"{'='*80}")
    print("TESTING SHOPIFY KEYWORD MATCHING")
    print(f"{'='*80}\n")
    
    for vendor, description in test_cases:
        account_code = mapper.get_account_code(
            vendor=vendor,
            description=description,
            is_income=True
        )
        
        print(f"Vendor: '{vendor}'")
        print(f"Description: '{description}'")
        print(f"→ Account Code: {account_code}")
        print()
        
        # Check if it matched 601 SALES or 860 BANK FEES
        if account_code == "601 · SALES":
            print(f"   ✅ CORRECT - Matched SALES")
        elif account_code == "860 · BANK & EBAY CHARGES":
            print(f"   ✅ CORRECT - Matched BANK FEES (contains 'fee')")
        else:
            print(f"   ❌ UNEXPECTED - Got {account_code}")
        print()

if __name__ == "__main__":
    test_shopify_variations()
