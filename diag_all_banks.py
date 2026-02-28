"""
Per-bank diagnostics - saves output to files to avoid truncation.
"""
import sys, os, logging, json
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, os.path.dirname(__file__))

from document_parser import DocumentParser
from bank_statement_parser import parse_bank_statement, BankName, detect_bank

PDF_DIR = os.path.join(os.path.dirname(__file__), "bank statments")
PDFS = {
    "Chase_Dec":    ("20 Dec Chase.pdf",    None),
    "Chase_List":   ("2CED020D-078D-423B-A7BE-CFE61B680C1D-list.pdf", None),
    "BMO":          ("BMO.pdf",             None),
    "BOA":          ("BoA.pdf",             None),
    "FifthThird":   ("FifthThird 53.pdf",   None),
    "US_Bank":      ("US Bank.pdf",         None),
    "AMEX":         ("AMEX.pdf",            None),
}

dp = DocumentParser()
results = {}

for label, (filename, manual_bank) in PDFS.items():
    path = os.path.join(PDF_DIR, filename)
    if not os.path.exists(path):
        print(f"[SKIP] {label}: not found")
        continue
    try:
        lines, is_readable, unreadable = dp.parse_pdf(path)
        text = "\n".join(lines)
        detected = detect_bank(text)
        result = parse_bank_statement(text)
        tx = result.transactions
        deposits = [t for t in tx if t.amount > 0]
        debits   = [t for t in tx if t.amount < 0]
        zeros    = [t for t in tx if t.amount == 0]
        needs_review = [t for t in tx if t.needs_review]
        no_date  = [t for t in tx if not t.date]
        
        summary = {
            "bank": label,
            "file": filename,
            "detected": detected.value,
            "is_readable": is_readable,
            "unreadable_pages": unreadable,
            "total_lines": len(lines),
            "total_tx": len(tx),
            "deposits": len(deposits),
            "debits": len(debits),
            "zeros": len(zeros),
            "needs_review": len(needs_review),
            "no_date": len(no_date),
            "beg_balance": result.beginning_balance,
            "end_balance": result.ending_balance,
            "errors": result.errors,
        }
        results[label] = summary
        
        print(f"\n{'='*70}")
        print(f"BANK: {label}  |  file: {filename}")
        print(f"  Detected: {detected.value}")
        print(f"  Lines: {len(lines)}  |  Unreadable pages: {unreadable}")
        print(f"  Transactions: {len(tx)}  (deposits: {len(deposits)}, debits: {len(debits)}, zero: {len(zeros)})")
        print(f"  Needs-review: {len(needs_review)}  |  No-date: {len(no_date)}")
        print(f"  Beg Balance: {result.beginning_balance}   End Balance: {result.ending_balance}")
        if result.errors: print(f"  ERRORS: {result.errors}")
        
        if tx:
            print(f"\n  --- First 5 ---")
            for t in tx[:5]:
                ds = t.date.strftime('%m/%d/%Y') if t.date else 'NO-DATE'
                cat = t.category.value if t.category else '?'
                print(f"    {ds} | {cat:14} | {t.amount:>10.2f} | {t.description[:55]}")
            if len(tx) > 10:
                print(f"  --- Last 5 ---")
                for t in tx[-5:]:
                    ds = t.date.strftime('%m/%d/%Y') if t.date else 'NO-DATE'
                    cat = t.category.value if t.category else '?'
                    print(f"    {ds} | {cat:14} | {t.amount:>10.2f} | {t.description[:55]}")
        
        # Show deposit amounts for BMO/BOA to see if key amounts appear
        if label in ("BMO", "BOA"):
            print(f"\n  --- All Deposits ---")
            for t in sorted(deposits, key=lambda x: x.amount, reverse=True)[:20]:
                ds = t.date.strftime('%m/%d/%Y') if t.date else 'NO-DATE'
                print(f"    {ds} | {t.amount:>10.2f} | {t.description[:60]}")
    except Exception as e:
        import traceback
        print(f"\n{'='*70}")
        print(f"[{label}] EXCEPTION: {e}")
        traceback.print_exc()

print("\n\nSUMMARY TABLE:")
print(f"{'Bank':<15} {'Detected':<12} {'Total TX':<10} {'Deposits':<10} {'Debits':<10} {'No TX?'}")
print("-"*70)
for label, s in results.items():
    err = "ERR" if s.get('errors') else ""
    no_tx = "NO TRANSACTIONS" if s['total_tx'] == 0 else ""
    print(f"{label:<15} {s['detected']:<12} {s['total_tx']:<10} {s['deposits']:<10} {s['debits']:<10} {no_tx}{err}")
